from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    denom = torch.clamp(torch.norm(x, dim=dim, keepdim=True), min=eps)
    return x / denom


@dataclass
class FusionLossConfig:
    temperature: float
    hard_k: int
    weight_hard: float
    weight_res_skt: float
    weight_res_txt: float
    weight_orth: float
    weight_rec: float
    hard_negative_mode: str = "dual-source"


class SketchEncoder(nn.Module):
    """ResNet sketch encoder used as frozen extractor after loading pretrained weights."""

    def __init__(self, backbone: str = "resnet50", output_dim: int = 512, pretrained: Union[bool, str] = True):
        super().__init__()
        if backbone != "resnet50":
            raise ValueError(f"Unsupported sketch backbone: {backbone}")

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained is True else None
        resnet = models.resnet50(weights=weights)
        in_features = resnet.fc.in_features
        resnet.fc = nn.Identity()

        if isinstance(pretrained, str) and pretrained:
            checkpoint = torch.load(pretrained, map_location="cpu")
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model_dict = resnet.state_dict()
            filtered = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            resnet.load_state_dict(filtered, strict=False)

        self.backbone = resnet
        self.projection = nn.Linear(in_features, output_dim)
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone(x)
        return self.projection(features)


class CrossAttentionStack(nn.Module):
    def __init__(self, dim: int, heads: int = 8, layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "attn": nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True),
                        "ln1": nn.LayerNorm(dim),
                        "ffn": nn.Sequential(
                            nn.Linear(dim, dim * 4),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(dim * 4, dim),
                        ),
                        "ln2": nn.LayerNorm(dim),
                    }
                )
            )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = query
        for block in self.layers:
            attn_out, _ = block["attn"](x, context, context)
            x = block["ln1"](x + attn_out)
            x = block["ln2"](x + block["ffn"](x))
        return x


class FusionPromptModule(nn.Module):
    """Learnable class-aware template prompts (num_classes x vector)."""

    def __init__(
        self,
        dim: int,
        num_template_prompts: int = 125,
        heads: int = 8,
        layers: int = 3,
        dropout: float = 0.1,
        decouple_dim: int = 256,
    ):
        super().__init__()
        self.template_prompt = nn.Parameter(torch.randn(1, num_template_prompts, dim) * 0.02)
        self.sketch_to_template = CrossAttentionStack(dim, heads=heads, layers=layers, dropout=dropout)
        self.template_to_text = CrossAttentionStack(dim, heads=heads, layers=layers, dropout=dropout)
        self.output_ln = nn.LayerNorm(dim)

        hidden = max(int(decouple_dim), 64)
        self.shared_encoder = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )
        self.sketch_private_encoder = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )
        self.text_private_encoder = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )

        self.sketch_decoder = nn.Sequential(
            nn.Linear(hidden * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.text_decoder = nn.Sequential(
            nn.Linear(hidden * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.prompt_reconstructor = nn.Sequential(
            nn.Linear(hidden * 3, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, sketch_feat: torch.Tensor, text_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b = sketch_feat.size(0)
        sketch_token = sketch_feat.unsqueeze(1)
        template = self.template_prompt.expand(b, -1, -1)
        embedded_template = self.sketch_to_template(query=template, context=sketch_token)

        text_token = text_feat.unsqueeze(1)
        fused_template = self.template_to_text(query=embedded_template, context=text_token)
        fusion_prompt = self.output_ln(fused_template.mean(dim=1))
        return embedded_template, fusion_prompt

    def decouple_from_prompt(
        self,
        fusion_prompt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shared = self.shared_encoder(fusion_prompt)
        sketch_private = self.sketch_private_encoder(fusion_prompt)
        text_private = self.text_private_encoder(fusion_prompt)

        sketch_decoupled = self.sketch_decoder(torch.cat([shared, sketch_private], dim=-1))
        text_decoupled = self.text_decoder(torch.cat([shared, text_private], dim=-1))
        reconstructed_prompt = self.prompt_reconstructor(torch.cat([shared, sketch_private, text_private], dim=-1))
        return sketch_decoupled, text_decoupled, shared, sketch_private, text_private, reconstructed_prompt


class SketchyFusionModel(nn.Module):
    """Fusion model: freeze pretrained extractors and train fusion prompt module only."""

    def __init__(
        self,
        clip_model: nn.Module,
        feature_dim: int = 512,
        sketch_backbone: str = "resnet50",
        sketch_pretrained: Union[bool, str] = True,
        num_template_prompts: int = 125,
        fusion_heads: int = 8,
        fusion_layers: int = 3,
        decouple_dim: int = 256,
        use_fusion_prompt: bool = True,
        use_disentanglement: bool = True,
    ) -> None:
        super().__init__()
        self.clip = clip_model
        self.feature_dim = feature_dim
        self.use_fusion_prompt = bool(use_fusion_prompt)
        self.use_disentanglement = bool(use_disentanglement)

        embed_dim = self._infer_clip_dim(self.clip)
        # Frozen extractor adapters copied from pretrained PT.
        self.image_base_adapter = nn.Linear(embed_dim, feature_dim)
        self.text_base_adapter = nn.Linear(embed_dim, feature_dim)
        nn.init.normal_(self.image_base_adapter.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.image_base_adapter.bias)
        nn.init.normal_(self.text_base_adapter.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.text_base_adapter.bias)

        self.sketch_encoder = SketchEncoder(backbone=sketch_backbone, output_dim=feature_dim, pretrained=sketch_pretrained)

        self.fusion = FusionPromptModule(
            dim=feature_dim,
            num_template_prompts=num_template_prompts,
            heads=fusion_heads,
            layers=fusion_layers,
            decouple_dim=decouple_dim,
        )

    @staticmethod
    def _infer_clip_dim(clip_model: nn.Module) -> int:
        if hasattr(clip_model, "embed_dim"):
            return int(clip_model.embed_dim)
        if hasattr(clip_model, "visual") and hasattr(clip_model.visual, "output_dim"):
            return int(clip_model.visual.output_dim)
        if hasattr(clip_model, "text_projection"):
            return int(clip_model.text_projection.shape[1])
        return 512

    def freeze_extractors(self) -> None:
        self.clip.eval()
        self.sketch_encoder.eval()
        self.image_base_adapter.eval()
        self.text_base_adapter.eval()

        for p in self.clip.parameters():
            p.requires_grad = False
        for p in self.sketch_encoder.parameters():
            p.requires_grad = False
        for p in self.image_base_adapter.parameters():
            p.requires_grad = False
        for p in self.text_base_adapter.parameters():
            p.requires_grad = False

    def unfreeze_fusion(self) -> None:
        self.fusion.train()
        for p in self.fusion.parameters():
            p.requires_grad = True

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.clip.encode_image(images).float()
            feats = self.image_base_adapter(feats)
        return l2_normalize(feats)

    def encode_texts(self, tokens: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.clip.encode_text(tokens).float()
            feats = self.text_base_adapter(feats)
        return l2_normalize(feats)

    def encode_sketches(self, sketches: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feats = self.sketch_encoder(sketches).float()
        return l2_normalize(feats)

    def build_fusion_query(self, text_feat: torch.Tensor, sketch_feat: torch.Tensor) -> torch.Tensor:
        if not self.use_fusion_prompt:
            return l2_normalize((text_feat + sketch_feat) * 0.5)
        _, fusion_prompt = self.fusion(sketch_feat=sketch_feat, text_feat=text_feat)
        return l2_normalize(fusion_prompt)

    def build_fusion_query_with_decoupled(
        self,
        text_feat: torch.Tensor,
        sketch_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.use_fusion_prompt:
            _, fusion_prompt = self.fusion(sketch_feat=sketch_feat, text_feat=text_feat)
            fusion_prompt = l2_normalize(fusion_prompt)
        else:
            fusion_prompt = l2_normalize((text_feat + sketch_feat) * 0.5)

        if self.use_disentanglement:
            sketch_from_fusion, text_from_fusion, shared, sketch_private, text_private, reconstructed_prompt = (
                self.fusion.decouple_from_prompt(fusion_prompt)
            )
        else:
            sketch_from_fusion = fusion_prompt
            text_from_fusion = fusion_prompt
            shared = torch.zeros_like(fusion_prompt)
            sketch_private = torch.zeros_like(fusion_prompt)
            text_private = torch.zeros_like(fusion_prompt)
            reconstructed_prompt = fusion_prompt
        return (
            fusion_prompt,
            l2_normalize(sketch_from_fusion),
            l2_normalize(text_from_fusion),
            shared,
            sketch_private,
            text_private,
            reconstructed_prompt,
        )

    def forward_train(self, batch: Dict[str, torch.Tensor], loss_cfg: FusionLossConfig) -> Dict[str, torch.Tensor]:
        images = batch["images"]
        texts = batch["texts"]
        sketches = batch["sketches_resnet"]
        neg_images = batch["neg_images"]

        image_feat = self.encode_images(images)
        text_feat = self.encode_texts(texts)
        sketch_feat = self.encode_sketches(sketches)

        (
            query_feat,
            sketch_from_fusion,
            text_from_fusion,
            shared_feat,
            sketch_private_feat,
            text_private_feat,
            reconstructed_prompt,
        ) = self.build_fusion_query_with_decoupled(text_feat=text_feat, sketch_feat=sketch_feat)
        neg_img_feat, neg_img_mask = self._encode_negative_images(neg_images)

        lhard = self._hard_info_nce(
            anchor=query_feat,
            positive=image_feat,
            negatives=neg_img_feat,
            neg_mask=neg_img_mask,
            temperature=loss_cfg.temperature,
            hard_k=loss_cfg.hard_k,
            mode=loss_cfg.hard_negative_mode,
        )

        lres_skt = self._residual_loss(pred_feat=sketch_from_fusion, ref_feat=sketch_feat)
        lres_txt = self._residual_loss(pred_feat=text_from_fusion, ref_feat=text_feat)
        lres = lres_skt + lres_txt
        lorth = self._orthogonality_loss(shared_feat, sketch_private_feat, text_private_feat)
        lrec = self._reconstruction_loss(reconstructed_prompt, query_feat)

        total = (
            loss_cfg.weight_hard * lhard
            + loss_cfg.weight_res_skt * lres_skt
            + loss_cfg.weight_res_txt * lres_txt
            + loss_cfg.weight_orth * lorth
            + loss_cfg.weight_rec * lrec
        )

        return {
            "loss": total,
            "lhard": lhard,
            "lres_skt": lres_skt,
            "lres_txt": lres_txt,
            "lres": lres,
            "lorth": lorth,
            "lrec": lrec,
            "features": {
                "query": query_feat,
                "image": image_feat,
                "text": text_feat,
                "sketch": sketch_feat,
                "sketch_from_fusion": sketch_from_fusion,
                "text_from_fusion": text_from_fusion,
                "shared": shared_feat,
                "sketch_private": sketch_private_feat,
                "text_private": text_private_feat,
            },
        }

    def forward_train_cached(
        self,
        sketches: torch.Tensor,
        image_feat: torch.Tensor,
        text_feat: torch.Tensor,
        neg_img_feat: torch.Tensor,
        neg_img_mask: torch.Tensor,
        loss_cfg: FusionLossConfig,
    ) -> Dict[str, torch.Tensor]:
        image_feat = l2_normalize(image_feat.float())
        text_feat = l2_normalize(text_feat.float())
        sketch_feat = self.encode_sketches(sketches)

        (
            query_feat,
            sketch_from_fusion,
            text_from_fusion,
            shared_feat,
            sketch_private_feat,
            text_private_feat,
            reconstructed_prompt,
        ) = self.build_fusion_query_with_decoupled(text_feat=text_feat, sketch_feat=sketch_feat)

        lhard = self._hard_info_nce(
            anchor=query_feat,
            positive=image_feat,
            negatives=neg_img_feat.float(),
            neg_mask=neg_img_mask,
            temperature=loss_cfg.temperature,
            hard_k=loss_cfg.hard_k,
            mode=loss_cfg.hard_negative_mode,
        )

        lres_skt = self._residual_loss(pred_feat=sketch_from_fusion, ref_feat=sketch_feat)
        lres_txt = self._residual_loss(pred_feat=text_from_fusion, ref_feat=text_feat)
        lres = lres_skt + lres_txt
        lorth = self._orthogonality_loss(shared_feat, sketch_private_feat, text_private_feat)
        lrec = self._reconstruction_loss(reconstructed_prompt, query_feat)

        total = (
            loss_cfg.weight_hard * lhard
            + loss_cfg.weight_res_skt * lres_skt
            + loss_cfg.weight_res_txt * lres_txt
            + loss_cfg.weight_orth * lorth
            + loss_cfg.weight_rec * lrec
        )

        return {
            "loss": total,
            "lhard": lhard,
            "lres_skt": lres_skt,
            "lres_txt": lres_txt,
            "lres": lres,
            "lorth": lorth,
            "lrec": lrec,
            "features": {
                "query": query_feat,
                "image": image_feat,
                "text": text_feat,
                "sketch": sketch_feat,
                "sketch_from_fusion": sketch_from_fusion,
                "text_from_fusion": text_from_fusion,
                "shared": shared_feat,
                "sketch_private": sketch_private_feat,
                "text_private": text_private_feat,
            },
        }

    def _encode_negative_images(self, negatives: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if negatives.dim() == 4:
            negatives = negatives.unsqueeze(1)
        if negatives.size(1) == 0:
            zeros = negatives.new_zeros((negatives.size(0), 0, self.feature_dim))
            mask = negatives.new_zeros((negatives.size(0), 0), dtype=torch.bool)
            return zeros, mask
        b, k = negatives.size(0), negatives.size(1)
        flat = negatives.reshape(b * k, *negatives.shape[2:])
        feats = self.encode_images(flat).view(b, k, -1)
        mask = negatives.reshape(b, k, -1).abs().sum(dim=-1) > 0
        return feats, mask

    def _hard_info_nce(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
        neg_mask: torch.Tensor,
        temperature: float,
        hard_k: int,
        mode: str,
    ) -> torch.Tensor:
        mode_norm = str(mode).strip().lower()
        if mode_norm not in {"dual-source", "in-batch-only", "hard-pool-only"}:
            raise ValueError(
                "loss.hard_negative_mode must be one of: dual-source, in-batch-only, hard-pool-only"
            )

        total = anchor.new_tensor(0.0)
        valid = 0
        for i in range(anchor.size(0)):
            local_neg = negatives[i]
            local_mask = neg_mask[i] if neg_mask.numel() > 0 else None
            if local_mask is not None:
                local_neg = local_neg[local_mask]

            inbatch_mask = torch.ones(anchor.size(0), dtype=torch.bool, device=anchor.device)
            inbatch_mask[i] = False
            inbatch_neg = positive[inbatch_mask]

            if mode_norm == "in-batch-only":
                all_negs = inbatch_neg
            elif mode_norm == "hard-pool-only":
                all_negs = local_neg
            else:
                all_negs = torch.cat([local_neg, inbatch_neg], dim=0) if local_neg.numel() else inbatch_neg

            if all_negs.size(0) == 0:
                continue

            sims = torch.matmul(anchor[i].unsqueeze(0), all_negs.t()).squeeze(0)
            k = min(max(int(hard_k), 1), sims.numel())
            hard_idx = torch.topk(sims, k=k, largest=True).indices
            hard_negs = all_negs[hard_idx]

            logits = torch.cat([positive[i].unsqueeze(0), hard_negs], dim=0)
            logits = torch.matmul(anchor[i].unsqueeze(0), logits.t()).squeeze(0) / temperature
            labels = torch.zeros(1, dtype=torch.long, device=anchor.device)
            total = total + F.cross_entropy(logits.unsqueeze(0), labels)
            valid += 1

        if valid == 0:
            return anchor.new_tensor(0.0)
        return total / valid

    @staticmethod
    def _residual_loss(pred_feat: torch.Tensor, ref_feat: torch.Tensor) -> torch.Tensor:
        fusion_unit = F.normalize(pred_feat, dim=-1)
        ref_unit = F.normalize(ref_feat, dim=-1)
        return (1.0 - F.cosine_similarity(fusion_unit, ref_unit, dim=-1)).mean()

    @staticmethod
    def _orthogonality_loss(
        shared_feat: torch.Tensor,
        sketch_private_feat: torch.Tensor,
        text_private_feat: torch.Tensor,
    ) -> torch.Tensor:
        def cross_corr_penalty(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            a = F.normalize(a, dim=-1)
            b = F.normalize(b, dim=-1)
            corr = torch.matmul(a.t(), b) / max(a.size(0), 1)
            return corr.pow(2).mean()

        return (
            cross_corr_penalty(shared_feat, sketch_private_feat)
            + cross_corr_penalty(shared_feat, text_private_feat)
            + cross_corr_penalty(sketch_private_feat, text_private_feat)
        )

    @staticmethod
    def _reconstruction_loss(reconstructed_prompt: torch.Tensor, target_prompt: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(reconstructed_prompt, target_prompt)


__all__ = ["SketchyFusionModel", "FusionLossConfig", "l2_normalize"]
