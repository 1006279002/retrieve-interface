"""
STFusionIR 融合检索模型
========================
实现 Sketch + Text 双模态融合的图像检索模型。

架构概览:
  ┌─────────────┐   ┌──────────────────┐   ┌─────────────┐
  │  草图 (PNG)  │   │  文本 (自然语言)   │   │  图像 (图库)  │
  └──────┬──────┘   └────────┬─────────┘   └──────┬──────┘
         │                   │                    │
    ┌────▼────┐         ┌────▼────┐          ┌────▼────┐
    │ ResNet50│         │  CLIP   │          │  CLIP   │
    │(冻结)   │         │(冻结)   │          │(冻结)   │
    └────┬────┘         └────┬────┘          └────┬────┘
         │                   │                    │
    sketch_feat         text_feat            image_feat
         │                   │
         └───────┬───────────┘
                 │
         ┌───────▼────────┐
         │ FusionPrompt   │  ← 可学习的类感知模板 prompts
         │   Module       │     通过 Cross-Attention 融合双模态
         │  (可训练)       │
         └───────┬────────┘
                 │
            fusion_query ──── cosine sim ────► Top-K 排名

核心组件:
  - SketchEncoder:     冻结的 ResNet-50 草图编码器
  - CrossAttentionStack: 多层交叉注意力模块
  - FusionPromptModule: 可学习的模板 prompt + 双模态融合 + 解耦
  - SketchyFusionModel: 顶层模型，协调编码器与融合模块
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from torchvision.models import ResNet50_Weights


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """L2 归一化：x / ||x||_2，eps 防止除零。"""
    denom = torch.clamp(torch.norm(x, dim=dim, keepdim=True), min=eps)
    return x / denom


@dataclass
class FusionLossConfig:
    """融合训练损失函数的超参数配置。

    loss = weight_hard * L_hard     (硬负样本 InfoNCE)
         + weight_res_skt * L_res   (草图残差：融合解码后的草图特征应与原始草图一致)
         + weight_res_txt * L_res   (文本残差)
         + weight_orth * L_orth     (正交损失：共享/私有特征应相互正交)
         + weight_rec * L_rec       (重构损失：解码后能重构回原始 prompt)
    """
    temperature: float       # InfoNCE 温度系数
    hard_k: int              # 硬负样本数量
    weight_hard: float       # 硬负样本损失权重
    weight_res_skt: float    # 草图残差损失权重
    weight_res_txt: float    # 文本残差损失权重
    weight_orth: float       # 正交性损失权重
    weight_rec: float        # 重构损失权重
    hard_negative_mode: str = "dual-source"  # 硬负样本来源: "dual-source" | "in-batch-only" | "hard-pool-only"


# ============================================================
# SketchEncoder — 冻结的草图编码器
# ============================================================
# 基于 ResNet-50 主干，可选加载 QuickDraw 预训练权重（手绘草图分类任务）。
# 推理时 backbone 冻结（no_grad），仅 projection 层可参与训练。

class SketchEncoder(nn.Module):
    """ResNet-50 草图编码器。加载预训练权重后冻结 backbone 用作特征提取器。"""

    def __init__(self, backbone: str = "resnet50", output_dim: int = 512, pretrained: Union[bool, str] = True):
        super().__init__()
        if backbone != "resnet50":
            raise ValueError(f"Unsupported sketch backbone: {backbone}")

        # 加载 ImageNet 预训练 ResNet-50，替换最后的 FC 为 Identity
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained is True else None
        resnet = models.resnet50(weights=weights)
        in_features = resnet.fc.in_features
        resnet.fc = nn.Identity()

        # 如果提供了自定义预训练权重路径（如 QuickDraw 预训练），则加载
        if isinstance(pretrained, str) and pretrained:
            checkpoint = torch.load(pretrained, map_location="cpu")
            # 兼容多种 checkpoint 格式
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint
            # 移除 DataParallel 的 "module." 前缀
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            model_dict = resnet.state_dict()
            filtered = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            resnet.load_state_dict(filtered, strict=False)

        self.backbone = resnet
        # 投影层：将 ResNet 特征映射到统一维度
        self.projection = nn.Linear(in_features, output_dim)
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：backbone 提取特征（冻结，无梯度）→ projection 映射。"""
        with torch.no_grad():
            features = self.backbone(x)
        return self.projection(features)


# ============================================================
# CrossAttentionStack — 多层交叉注意力模块
# ============================================================
# 每一层: MultiheadAttention(query, context, context) → LayerNorm → FFN → LayerNorm
# 用于将一种模态的信息注入到另一种模态的表示中。

class CrossAttentionStack(nn.Module):
    """多层交叉注意力堆栈：query 通过关注 context 来更新自身表示。"""

    def __init__(self, dim: int, heads: int = 8, layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "attn": nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True),
                        "ln1": nn.LayerNorm(dim),         # 注意力后 LayerNorm
                        "ffn": nn.Sequential(               # 前馈网络 (dim → 4*dim → dim)
                            nn.Linear(dim, dim * 4),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(dim * 4, dim),
                        ),
                        "ln2": nn.LayerNorm(dim),         # FFN 后 LayerNorm
                    }
                )
            )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """query 通过交叉注意力关注 context，逐层更新。"""
        x = query
        for block in self.layers:
            # 交叉注意力：query 关注 context
            attn_out, _ = block["attn"](x, context, context)
            x = block["ln1"](x + attn_out)       # 残差连接 + LayerNorm
            x = block["ln2"](x + block["ffn"](x)) # FFN + 残差连接 + LayerNorm
        return x


# ============================================================
# FusionPromptModule — 核心融合模块
# ============================================================
# 双阶段交叉注意力融合 + 特征解耦（disentanglement）:
#
#   Stage 1: Sketch → Template
#     模板 prompts 通过 Cross-Attention 关注草图特征 → embedded_template
#
#   Stage 2: Template → Text
#     embedded_template 通过 Cross-Attention 关注文本特征 → fused_template
#     平均池化 → fusion_prompt（融合查询向量）
#
#   解耦模块 (Disentanglement):
#     将 fusion_prompt 分解为:
#       - shared:          共享语义（草图+文本共同信息）
#       - sketch_private:  草图独有信息
#       - text_private:    文本独有信息
#     然后从各部分重构:
#       - sketch_decoupled: shared + sketch_private → 应与原始草图特征一致
#       - text_decoupled:   shared + text_private  → 应与原始文本特征一致
#       - reconstructed:    shared + sketch_private + text_private → 应能还原 fusion_prompt

class FusionPromptModule(nn.Module):
    """可学习的类感知模板 prompts + 双模态融合 + 解耦。"""

    def __init__(
        self,
        dim: int,
        num_template_prompts: int = 125,  # 模板数量（通常 = 类别数）
        heads: int = 8,
        layers: int = 3,
        dropout: float = 0.1,
        decouple_dim: int = 256,
    ):
        super().__init__()
        # 可学习的模板提示向量 [1, num_templates, dim]
        self.template_prompt = nn.Parameter(torch.randn(1, num_template_prompts, dim) * 0.02)

        # 双阶段交叉注意力
        self.sketch_to_template = CrossAttentionStack(dim, heads=heads, layers=layers, dropout=dropout)
        self.template_to_text = CrossAttentionStack(dim, heads=heads, layers=layers, dropout=dropout)
        self.output_ln = nn.LayerNorm(dim)

        # ---- 解耦模块 ----
        hidden = max(int(decouple_dim), 64)
        # 共享编码器：提取草图+文本共同语义
        self.shared_encoder = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
        )
        # 草图私有编码器：提取草图独有信息
        self.sketch_private_encoder = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
        )
        # 文本私有编码器：提取文本独有信息
        self.text_private_encoder = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
        )

        # 解码器：从共享+私有重构各模态特征
        self.sketch_decoder = nn.Sequential(
            nn.Linear(hidden * 2, dim), nn.GELU(), nn.Linear(dim, dim), nn.LayerNorm(dim),
        )
        self.text_decoder = nn.Sequential(
            nn.Linear(hidden * 2, dim), nn.GELU(), nn.Linear(dim, dim), nn.LayerNorm(dim),
        )
        # Prompt 重构器：从 shared + sketch_private + text_private 重构原始 prompt
        self.prompt_reconstructor = nn.Sequential(
            nn.Linear(hidden * 3, dim), nn.GELU(), nn.Linear(dim, dim), nn.LayerNorm(dim),
        )

    def forward(self, sketch_feat: torch.Tensor, text_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向融合：
        Returns:
          embedded_template: [B, num_templates, dim] — 融合后的模板表示
          fusion_prompt:     [B, dim] — 最终融合查询向量（模板平均池化）
        """
        b = sketch_feat.size(0)
        sketch_token = sketch_feat.unsqueeze(1)  # [B, 1, dim] — 草图作为单个 token
        template = self.template_prompt.expand(b, -1, -1)  # [B, num_templates, dim]

        # Stage 1: 模板关注草图 → 草图信息注入模板
        embedded_template = self.sketch_to_template(query=template, context=sketch_token)

        # Stage 2: 融合模板关注文本 → 文本信息进一步注入
        text_token = text_feat.unsqueeze(1)  # [B, 1, dim]
        fused_template = self.template_to_text(query=embedded_template, context=text_token)

        # 平均池化得到最终融合查询向量
        fusion_prompt = self.output_ln(fused_template.mean(dim=1))
        return embedded_template, fusion_prompt

    def decouple_from_prompt(
        self,
        fusion_prompt: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """从融合 prompt 中解耦出共享/私有特征，并重构各模态表示。

        Returns:
          sketch_decoupled:  从共享+草图私有重构的草图特征
          text_decoupled:    从共享+文本私有重构的文本特征
          shared:            共享语义特征
          sketch_private:    草图独有特征
          text_private:      文本独有特征
          reconstructed_prompt: 从三部分重构的 prompt（应与 fusion_prompt 一致）
        """
        shared = self.shared_encoder(fusion_prompt)
        sketch_private = self.sketch_private_encoder(fusion_prompt)
        text_private = self.text_private_encoder(fusion_prompt)

        # [shared, sketch_private] → 草图特征
        sketch_decoupled = self.sketch_decoder(torch.cat([shared, sketch_private], dim=-1))
        # [shared, text_private] → 文本特征
        text_decoupled = self.text_decoder(torch.cat([shared, text_private], dim=-1))
        # [shared, sketch_private, text_private] → 重构 prompt
        reconstructed_prompt = self.prompt_reconstructor(torch.cat([shared, sketch_private, text_private], dim=-1))
        return sketch_decoupled, text_decoupled, shared, sketch_private, text_private, reconstructed_prompt


# ============================================================
# SketchyFusionModel — 顶层融合检索模型
# ============================================================
# 协调三个编码器（CLIP 图像、CLIP 文本、ResNet 草图）和一个可训练的 FusionPromptModule。
# 训练策略：冻结所有特征提取器，仅训练 FusionPromptModule 的融合参数。

class SketchyFusionModel(nn.Module):
    """融合检索模型：冻结预训练提取器 + 训练融合 prompt 模块。"""

    def __init__(
        self,
        clip_model: nn.Module,          # CLIP 模型（ViT-B/32 等）
        feature_dim: int = 512,          # 统一特征维度
        sketch_backbone: str = "resnet50",
        sketch_pretrained: Union[bool, str] = True,
        num_template_prompts: int = 125, # 模板 prompt 数量
        fusion_heads: int = 8,           # 交叉注意力头数
        fusion_layers: int = 3,          # 交叉注意力层数
        decouple_dim: int = 256,         # 解耦模块隐藏维度
        use_fusion_prompt: bool = True,  # 是否启用融合 prompt（关闭则退化为简单平均）
        use_disentanglement: bool = True,# 是否启用解耦模块
    ) -> None:
        super().__init__()
        self.clip = clip_model
        self.feature_dim = feature_dim
        self.use_fusion_prompt = bool(use_fusion_prompt)
        self.use_disentanglement = bool(use_disentanglement)

        embed_dim = self._infer_clip_dim(self.clip)
        # 线性适配器：将 CLIP 输出维度映射到统一的 feature_dim
        self.image_base_adapter = nn.Linear(embed_dim, feature_dim)
        self.text_base_adapter = nn.Linear(embed_dim, feature_dim)
        nn.init.normal_(self.image_base_adapter.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.image_base_adapter.bias)
        nn.init.normal_(self.text_base_adapter.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.text_base_adapter.bias)

        # 草图编码器（ResNet-50 + QuickDraw 预训练权重）
        self.sketch_encoder = SketchEncoder(backbone=sketch_backbone, output_dim=feature_dim, pretrained=sketch_pretrained)

        # 融合 prompt 模块（唯一可训练的部分）
        self.fusion = FusionPromptModule(
            dim=feature_dim,
            num_template_prompts=num_template_prompts,
            heads=fusion_heads,
            layers=fusion_layers,
            decouple_dim=decouple_dim,
        )

    @staticmethod
    def _infer_clip_dim(clip_model: nn.Module) -> int:
        """自动推断 CLIP 模型的输出维度。"""
        if hasattr(clip_model, "embed_dim"):
            return int(clip_model.embed_dim)
        if hasattr(clip_model, "visual") and hasattr(clip_model.visual, "output_dim"):
            return int(clip_model.visual.output_dim)
        if hasattr(clip_model, "text_projection"):
            return int(clip_model.text_projection.shape[1])
        return 512  # 默认 ViT-B/32 维度

    def freeze_extractors(self) -> None:
        """冻结所有特征提取器（CLIP、Sketch Encoder、适配器），仅保留 fusion 模块可训练。"""
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
        """解冻 fusion 模块（训练时使用）。"""
        self.fusion.train()
        for p in self.fusion.parameters():
            p.requires_grad = True

    # ---------- 编码器接口 ----------

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """CLIP 编码图像 → adapter → L2 归一化。"""
        with torch.no_grad():
            feats = self.clip.encode_image(images).float()
            feats = self.image_base_adapter(feats)
        return l2_normalize(feats)

    def encode_texts(self, tokens: torch.Tensor) -> torch.Tensor:
        """CLIP 编码文本 → adapter → L2 归一化。"""
        with torch.no_grad():
            feats = self.clip.encode_text(tokens).float()
            feats = self.text_base_adapter(feats)
        return l2_normalize(feats)

    def encode_sketches(self, sketches: torch.Tensor) -> torch.Tensor:
        """ResNet 编码草图 → L2 归一化。"""
        with torch.no_grad():
            feats = self.sketch_encoder(sketches).float()
        return l2_normalize(feats)

    # ---------- 融合查询构建 ----------

    def build_fusion_query(self, text_feat: torch.Tensor, sketch_feat: torch.Tensor) -> torch.Tensor:
        """构建融合查询向量（推理时使用）。
        若未启用 fusion_prompt，退化为 (text + sketch) / 2 的简单平均。
        """
        if not self.use_fusion_prompt:
            return l2_normalize((text_feat + sketch_feat) * 0.5)
        _, fusion_prompt = self.fusion(sketch_feat=sketch_feat, text_feat=text_feat)
        return l2_normalize(fusion_prompt)

    def build_fusion_query_with_decoupled(
        self,
        text_feat: torch.Tensor,
        sketch_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """构建融合查询 + 解耦特征（训练时使用，需要计算解耦损失）。

        Returns:
          fusion_prompt, sketch_decoupled, text_decoupled, shared, sketch_private, text_private, reconstructed_prompt
        """
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
            # 未启用解耦时返回占位零向量
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

    # ============================================================
    # 训练前向传播
    # ============================================================

    def forward_train(self, batch: Dict[str, torch.Tensor], loss_cfg: FusionLossConfig) -> Dict[str, torch.Tensor]:
        """训练时前向传播（从原始图像/文本/草图开始）。
        计算所有损失分量并返回字典。
        """
        images = batch["images"]
        texts = batch["texts"]
        sketches = batch["sketches_resnet"]
        neg_images = batch["neg_images"]

        # 编码三模态
        image_feat = self.encode_images(images)
        text_feat = self.encode_texts(texts)
        sketch_feat = self.encode_sketches(sketches)

        # 融合 + 解耦
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

        # ---- 计算各损失分量 ----
        # 硬负样本 InfoNCE 损失（核心检索损失）
        lhard = self._hard_info_nce(
            anchor=query_feat,
            positive=image_feat,
            negatives=neg_img_feat,
            neg_mask=neg_img_mask,
            temperature=loss_cfg.temperature,
            hard_k=loss_cfg.hard_k,
            mode=loss_cfg.hard_negative_mode,
        )

        # 残差损失：解耦出的草图/文本特征应与原始编码特征一致
        lres_skt = self._residual_loss(pred_feat=sketch_from_fusion, ref_feat=sketch_feat)
        lres_txt = self._residual_loss(pred_feat=text_from_fusion, ref_feat=text_feat)
        lres = lres_skt + lres_txt

        # 正交损失：共享特征与私有特征应相互正交（减少信息冗余）
        lorth = self._orthogonality_loss(shared_feat, sketch_private_feat, text_private_feat)

        # 重构损失：解耦后的各部分应能重构回原始 prompt
        lrec = self._reconstruction_loss(reconstructed_prompt, query_feat)

        # 加权总损失
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
        """训练时前向传播（使用预缓存的 CLIP 特征，加速训练）。
        与 forward_train 的区别：跳过 CLIP 编码，直接使用预计算的特征。
        """
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

    # ============================================================
    # 损失函数
    # ============================================================

    def _encode_negative_images(self, negatives: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """编码负样本图像：支持 [B, K, C, H, W] 或 [B, C, H, W] 格式。
        Returns: (features [B, K, D], mask [B, K])
        """
        if negatives.dim() == 4:
            negatives = negatives.unsqueeze(1)  # 添加 K 维度
        if negatives.size(1) == 0:
            zeros = negatives.new_zeros((negatives.size(0), 0, self.feature_dim))
            mask = negatives.new_zeros((negatives.size(0), 0), dtype=torch.bool)
            return zeros, mask
        b, k = negatives.size(0), negatives.size(1)
        flat = negatives.reshape(b * k, *negatives.shape[2:])
        feats = self.encode_images(flat).view(b, k, -1)
        # mask: 非零图像为有效负样本
        mask = negatives.reshape(b, k, -1).abs().sum(dim=-1) > 0
        return feats, mask

    def _hard_info_nce(
        self,
        anchor: torch.Tensor,        # 查询向量 [B, D]
        positive: torch.Tensor,      # 正样本（ground-truth 图像）[B, D]
        negatives: torch.Tensor,     # 负样本池 [B, K, D]
        neg_mask: torch.Tensor,      # 负样本有效性 mask [B, K]
        temperature: float,          # 温度系数
        hard_k: int,                 # 选取的硬负样本数量
        mode: str,                   # 硬负样本来源模式
    ) -> torch.Tensor:
        """硬负样本 InfoNCE 损失。

        对每个样本：
          1. 从负样本池 + batch 内其他样本中选取 top-K 最难的负样本
          2. 用 (anchor, positive, hard_negatives) 计算 InfoNCE 损失

        mode 选项:
          - "dual-source":    同时使用显式负样本池 + batch 内负样本
          - "in-batch-only":  仅使用 batch 内其他样本作为负样本
          - "hard-pool-only": 仅使用显式负样本池
        """
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
                local_neg = local_neg[local_mask]  # 过滤无效负样本

            # batch 内负样本：除自身外所有正样本
            inbatch_mask = torch.ones(anchor.size(0), dtype=torch.bool, device=anchor.device)
            inbatch_mask[i] = False
            inbatch_neg = positive[inbatch_mask]

            # 根据模式合并负样本来源
            if mode_norm == "in-batch-only":
                all_negs = inbatch_neg
            elif mode_norm == "hard-pool-only":
                all_negs = local_neg
            else:
                all_negs = torch.cat([local_neg, inbatch_neg], dim=0) if local_neg.numel() else inbatch_neg

            if all_negs.size(0) == 0:
                continue

            # 选取 top-K 最难的负样本（与 anchor 相似度最高）
            sims = torch.matmul(anchor[i].unsqueeze(0), all_negs.t()).squeeze(0)
            k = min(max(int(hard_k), 1), sims.numel())
            hard_idx = torch.topk(sims, k=k, largest=True).indices
            hard_negs = all_negs[hard_idx]

            # InfoNCE: -log(exp(anchor·pos/τ) / Σ exp(anchor·neg/τ))
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
        """残差损失：1 - cosine_similarity(预测特征, 参考特征)。
        鼓励解耦后的草图/文本特征与原始编码特征保持一致。
        """
        fusion_unit = F.normalize(pred_feat, dim=-1)
        ref_unit = F.normalize(ref_feat, dim=-1)
        return (1.0 - F.cosine_similarity(fusion_unit, ref_unit, dim=-1)).mean()

    @staticmethod
    def _orthogonality_loss(
        shared_feat: torch.Tensor,
        sketch_private_feat: torch.Tensor,
        text_private_feat: torch.Tensor,
    ) -> torch.Tensor:
        """正交性损失：共享特征与私有特征之间、草图私有与文本私有之间应相互正交。
        通过最小化交叉相关矩阵的平方均值来实现。
        """
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
        """重构损失：解耦后的各部分应能重构回原始融合 prompt。使用 MSE。"""
        return F.mse_loss(reconstructed_prompt, target_prompt)


__all__ = ["SketchyFusionModel", "FusionLossConfig", "l2_normalize"]
