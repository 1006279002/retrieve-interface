from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .inference import clip
from .inference.cstbir_fusion_model import SketchyFusionModel


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
ASSET_ROOT = Path(os.environ.get("STFUSION_ASSET_ROOT", WORKSPACE_ROOT / "stfusion_assets")).expanduser().resolve()
DEFAULT_CACHE_DIR = WORKSPACE_ROOT / ".stfusion_cache"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

StatusCallback = Optional[Callable[[str, Optional[int], Optional[int]], None]]


@dataclass(frozen=True)
class DatasetDefinition:
    key: str
    label: str
    config_path: Path
    checkpoint_path: Path
    text_index_path: Path
    image_root: Path
    sketch_root: Path
    mode: str


@dataclass
class QuerySample:
    id: str
    label: str
    category: str
    virtual_class: str
    text: str
    image_path: Path
    sketch_paths: List[Path]


@dataclass
class RetrievalHit:
    rank: int
    score: float
    image_path: Path


@dataclass
class RetrievalResult:
    dataset_key: str
    sample: QuerySample
    sketch_path: Path
    query_text: str
    hits: List[RetrievalHit]
    ground_truth_rank: Optional[int]


@dataclass
class AssetCheck:
    name: str
    path: Optional[Path]
    exists: bool
    required: bool
    note: str = ""


DATASETS: Dict[str, DatasetDefinition] = {
    "sketchy": DatasetDefinition(
        key="sketchy",
        label="Sketchy",
        config_path=ASSET_ROOT / "configs" / "config_sketchy_fusion.yaml",
        checkpoint_path=ASSET_ROOT / "checkpoints" / "sketchy_fusion_best.pt",
        text_index_path=WORKSPACE_ROOT / "datasets" / "sketchy_test.txt",
        image_root=WORKSPACE_ROOT / "datasets" / "256x256" / "photo",
        sketch_root=WORKSPACE_ROOT / "datasets" / "256x256" / "sketch",
        mode="sketchy",
    ),
    "chair": DatasetDefinition(
        key="chair",
        label="Chair",
        config_path=ASSET_ROOT / "configs" / "config_chair_fusion.yaml",
        checkpoint_path=ASSET_ROOT / "checkpoints" / "chair_fusion_best.pt",
        text_index_path=WORKSPACE_ROOT / "datasets" / "chair_test.txt",
        image_root=WORKSPACE_ROOT / "datasets" / "datasets" / "ChairV2" / "testB",
        sketch_root=WORKSPACE_ROOT / "datasets" / "datasets" / "ChairV2" / "testA",
        mode="paired",
    ),
    "shoe": DatasetDefinition(
        key="shoe",
        label="Shoe",
        config_path=ASSET_ROOT / "configs" / "config_shoe_fusion.yaml",
        checkpoint_path=ASSET_ROOT / "checkpoints" / "shoe_fusion_best.pt",
        text_index_path=WORKSPACE_ROOT / "datasets" / "shoe_test.txt",
        image_root=WORKSPACE_ROOT / "datasets" / "datasets" / "ShoeV2" / "testB",
        sketch_root=WORKSPACE_ROOT / "datasets" / "datasets" / "ShoeV2" / "testA",
        mode="paired",
    ),
}


def list_dataset_definitions() -> List[DatasetDefinition]:
    return [DATASETS[key] for key in ("sketchy", "chair", "shoe")]


def _emit_status(callback: StatusCallback, message: str, current: Optional[int] = None, total: Optional[int] = None) -> None:
    if callback is not None:
        callback(message, current, total)


def _load_yaml(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_path(raw_path: str | Path, base_paths: Sequence[Path]) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate
    for base_path in base_paths:
        resolved = (base_path / candidate).resolve()
        if resolved.exists():
            return resolved
    return (base_paths[0] / candidate).resolve()


def _select_device(preferred: Optional[str] = None) -> str:
    env_override = os.environ.get("STFUSION_DEVICE")
    if env_override:
        preferred = env_override

    if preferred:
        value = preferred.lower()
        if value.startswith("cuda") and torch.cuda.is_available():
            return preferred
        if value == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return preferred
        if value == "cpu":
            return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _default_sketch_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _load_image(path: Path, preprocess) -> torch.Tensor:
    with Image.open(path) as img:
        return preprocess(img.convert("RGB"))


def _load_sketch(path: Path, transform) -> torch.Tensor:
    with Image.open(path) as img:
        return transform(img.convert("RGB"))


def _read_text_index(path: Path) -> List[tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError("Text index not found: {}".format(path))

    rows: List[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or "\t" not in line:
                continue
            left, right = line.split("\t", 1)
            rows.append((left.strip(), right.strip()))
    return rows


def _trailing_number_key(path: Path) -> tuple[str, int]:
    match = re.search(r"(\d+)$", path.stem)
    if match:
        return path.stem[: match.start(1)], int(match.group(1))
    return path.stem, -1


def _sorted_sketches(paths: Iterable[Path]) -> List[Path]:
    return sorted(paths, key=lambda item: _trailing_number_key(item))


def _sample_label(index: int, primary: str, secondary: str, text: str) -> str:
    snippet = " ".join(text.split())[:28]
    return "{:04d} | {} | {} | {}".format(index + 1, primary, secondary, snippet)


def _build_sketchy_samples(definition: DatasetDefinition) -> List[QuerySample]:
    samples: List[QuerySample] = []
    for index, (relative_image, raw_text) in enumerate(_read_text_index(definition.text_index_path)):
        rel_path = Path(relative_image)
        if len(rel_path.parts) < 3:
            continue
        transform_dir, category, filename = rel_path.parts[0], rel_path.parts[1], rel_path.parts[-1]
        image_path = (definition.image_root / rel_path).resolve()
        stem = Path(filename).stem
        sketch_glob = definition.sketch_root / transform_dir / category / "{}-*.png".format(stem)
        sketch_paths = _sorted_sketches(sketch_glob.parent.glob(sketch_glob.name))
        text = raw_text.replace("<sketch>", category.replace("_", " ")).strip()
        samples.append(
            QuerySample(
                id="{}:{}".format(category, stem),
                label=_sample_label(index, category, stem, text),
                category=category,
                virtual_class=category,
                text=text,
                image_path=image_path,
                sketch_paths=[path.resolve() for path in sketch_paths if path.exists()],
            )
        )
    return samples


def _build_paired_samples(definition: DatasetDefinition) -> List[QuerySample]:
    dataset_name = definition.label.lower()
    samples: List[QuerySample] = []
    for index, (filename, text) in enumerate(_read_text_index(definition.text_index_path)):
        image_path = (definition.image_root / filename).resolve()
        stem = Path(filename).stem
        sketch_paths = _sorted_sketches(definition.sketch_root.glob("{}_*.png".format(stem)))
        samples.append(
            QuerySample(
                id="{}:{}".format(dataset_name, stem),
                label=_sample_label(index, stem, dataset_name, text),
                category=dataset_name,
                virtual_class=stem,
                text=text,
                image_path=image_path,
                sketch_paths=[path.resolve() for path in sketch_paths if path.exists()],
            )
        )
    return samples


def _build_samples(definition: DatasetDefinition) -> List[QuerySample]:
    if definition.mode == "sketchy":
        return _build_sketchy_samples(definition)
    if definition.mode == "paired":
        return _build_paired_samples(definition)
    raise ValueError("Unsupported dataset mode: {}".format(definition.mode))


def _resolve_required_files(definition: DatasetDefinition) -> List[AssetCheck]:
    config = _load_yaml(definition.config_path) if definition.config_path.exists() else {}
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}

    checks = [
        AssetCheck("{} config".format(definition.label), definition.config_path, definition.config_path.exists(), True),
        AssetCheck("{} checkpoint".format(definition.label), definition.checkpoint_path, definition.checkpoint_path.exists(), True),
        AssetCheck("{} text index".format(definition.label), definition.text_index_path, definition.text_index_path.exists(), True),
        AssetCheck("{} image root".format(definition.label), definition.image_root, definition.image_root.exists(), True),
        AssetCheck("{} sketch root".format(definition.label), definition.sketch_root, definition.sketch_root.exists(), True),
    ]

    sketch_pretrained = model_cfg.get("sketch_pretrained")
    if isinstance(sketch_pretrained, str):
        sketch_pretrained_path = _resolve_path(sketch_pretrained, [ASSET_ROOT, WORKSPACE_ROOT])
        checks.append(
            AssetCheck(
                "{} sketch backbone weight".format(definition.label),
                sketch_pretrained_path,
                sketch_pretrained_path.exists(),
                True,
            )
        )

    clip_model_path = model_cfg.get("clip_model_path")
    if isinstance(clip_model_path, str) and clip_model_path.strip():
        clip_path = _resolve_path(clip_model_path, [ASSET_ROOT, WORKSPACE_ROOT])
        checks.append(
            AssetCheck(
                "{} CLIP weight".format(definition.label),
                clip_path,
                clip_path.exists(),
                True,
            )
        )
    else:
        checks.append(
            AssetCheck(
                "{} CLIP weight".format(definition.label),
                None,
                True,
                False,
                note="Not bundled in workspace; loaded from ~/.cache/clip or downloaded on first run.",
            )
        )

    return checks


def workspace_asset_report() -> Dict[str, object]:
    datasets = {}
    missing_required = []
    for definition in list_dataset_definitions():
        checks = _resolve_required_files(definition)
        datasets[definition.key] = [
            {
                "name": check.name,
                "path": str(check.path) if check.path else None,
                "exists": check.exists,
                "required": check.required,
                "note": check.note,
            }
            for check in checks
        ]
        for check in checks:
            if check.required and not check.exists:
                missing_required.append(check.name)

    app_checks = [
        AssetCheck("GUI launcher", WORKSPACE_ROOT / "run_stfusion_gui.py", (WORKSPACE_ROOT / "run_stfusion_gui.py").exists(), True),
        AssetCheck("GUI package", WORKSPACE_ROOT / "stfusion_gui", (WORKSPACE_ROOT / "stfusion_gui").exists(), True),
        AssetCheck("Local inference package", WORKSPACE_ROOT / "stfusion_gui" / "inference", (WORKSPACE_ROOT / "stfusion_gui" / "inference").exists(), True),
        AssetCheck("Asset root", ASSET_ROOT, ASSET_ROOT.exists(), True),
    ]

    for check in app_checks:
        if check.required and not check.exists:
            missing_required.append(check.name)

        return {
            "workspace_root": str(WORKSPACE_ROOT),
            "asset_root": str(ASSET_ROOT),
            "required_ready": not missing_required,
            "missing_required": missing_required,
        "app": [
            {
                "name": check.name,
                "path": str(check.path) if check.path else None,
                "exists": check.exists,
                "required": check.required,
                "note": check.note,
            }
            for check in app_checks
        ],
        "datasets": datasets,
    }


class FusionDatasetSession:
    def __init__(self, definition: DatasetDefinition, cache_dir: Optional[Path] = None) -> None:
        self.definition = definition
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.config = _load_yaml(self.definition.config_path)
        self.samples = _build_samples(self.definition)
        self.gamma = float(self.config.get("retrieval", {}).get("gamma", 0.6))

        self.device_name: Optional[str] = None
        self.device: Optional[torch.device] = None
        self.preprocess = None
        self.sketch_transform = None
        self.model: Optional[SketchyFusionModel] = None
        self.gallery_paths: List[Path] = []
        self.gallery_matrix: Optional[torch.Tensor] = None
        self._gallery_index: Dict[Path, int] = {}
        self._ready = False

    def _resolve_num_template_prompts(self) -> int:
        model_cfg = self.config.get("model", {})
        explicit = model_cfg.get("num_template_prompts")
        if explicit is not None:
            return max(1, int(explicit))
        return max(1, len({sample.virtual_class for sample in self.samples}))

    def _cache_path(self) -> Path:
        return self.cache_dir / "{}_gallery.pt".format(self.definition.key)

    def _build_model(self, status_callback: StatusCallback = None) -> None:
        training_cfg = self.config.get("training", {})
        preferred_device = training_cfg.get("device")
        self.device_name = _select_device(str(preferred_device) if preferred_device else None)
        self.device = torch.device(self.device_name)
        _emit_status(status_callback, "Loading CLIP and retrieval model")

        model_cfg = self.config.get("model", {})
        clip_source = model_cfg.get("clip_model_path") or model_cfg.get("clip_model", "ViT-B/32")
        if isinstance(clip_source, str):
            clip_path_candidate = _resolve_path(clip_source, [ASSET_ROOT, WORKSPACE_ROOT])
            if clip_path_candidate.exists():
                clip_source = str(clip_path_candidate)

        sketch_pretrained = model_cfg.get("sketch_pretrained", True)
        if isinstance(sketch_pretrained, str):
            sketch_pretrained_path = _resolve_path(sketch_pretrained, [ASSET_ROOT, WORKSPACE_ROOT])
            sketch_pretrained = str(sketch_pretrained_path)

        clip_model, preprocess = clip.load(clip_source, device=self.device, jit=False)

        model = SketchyFusionModel(
            clip_model=clip_model,
            feature_dim=int(model_cfg.get("feature_dim", 512)),
            sketch_backbone=str(model_cfg.get("sketch_backbone", "resnet50")),
            sketch_pretrained=sketch_pretrained,
            num_template_prompts=self._resolve_num_template_prompts(),
            fusion_heads=int(model_cfg.get("fusion_heads", 8)),
            fusion_layers=int(model_cfg.get("fusion_layers", 3)),
            decouple_dim=int(model_cfg.get("decouple_dim", 256)),
            use_fusion_prompt=bool(model_cfg.get("use_fusion_prompt", True)),
            use_disentanglement=bool(model_cfg.get("use_disentanglement", True)),
        ).to(self.device)

        checkpoint_path = self.definition.checkpoint_path
        if not checkpoint_path.exists():
            raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))

        state = torch.load(str(checkpoint_path), map_location="cpu")
        model_state = state.get("model", state)
        model.load_state_dict(model_state, strict=False)
        model.eval()
        model.freeze_extractors()

        self.model = model
        self.preprocess = preprocess
        self.sketch_transform = _default_sketch_transform(int(self.config.get("data", {}).get("image_size", 224)))

    def _load_gallery_cache(self) -> bool:
        cache_path = self._cache_path()
        if not cache_path.exists() or self.device is None:
            return False

        payload = torch.load(str(cache_path), map_location="cpu")
        expected = {
            "dataset_key": self.definition.key,
            "config_path": str(self.definition.config_path.resolve()),
            "config_mtime_ns": self.definition.config_path.stat().st_mtime_ns,
            "checkpoint_path": str(self.definition.checkpoint_path.resolve()),
            "checkpoint_mtime_ns": self.definition.checkpoint_path.stat().st_mtime_ns,
            "text_index_path": str(self.definition.text_index_path.resolve()),
            "text_index_mtime_ns": self.definition.text_index_path.stat().st_mtime_ns,
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                return False

        image_paths = [Path(path) for path in payload.get("image_paths", [])]
        image_matrix = payload.get("image_matrix")
        if not image_paths or image_matrix is None:
            return False

        self.gallery_paths = [path.resolve() for path in image_paths]
        self.gallery_matrix = image_matrix.to(self.device)
        self._gallery_index = {path: index for index, path in enumerate(self.gallery_paths)}
        return True

    def _save_gallery_cache(self) -> None:
        if self.gallery_matrix is None:
            return

        payload = {
            "dataset_key": self.definition.key,
            "config_path": str(self.definition.config_path.resolve()),
            "config_mtime_ns": self.definition.config_path.stat().st_mtime_ns,
            "checkpoint_path": str(self.definition.checkpoint_path.resolve()),
            "checkpoint_mtime_ns": self.definition.checkpoint_path.stat().st_mtime_ns,
            "text_index_path": str(self.definition.text_index_path.resolve()),
            "text_index_mtime_ns": self.definition.text_index_path.stat().st_mtime_ns,
            "image_paths": [str(path) for path in self.gallery_paths],
            "image_matrix": self.gallery_matrix.detach().cpu(),
        }
        torch.save(payload, str(self._cache_path()))

    def _compute_gallery_matrix(self, status_callback: StatusCallback = None) -> None:
        if self.model is None or self.preprocess is None or self.device is None:
            raise RuntimeError("Model session is not initialized.")

        unique_images = sorted({sample.image_path.resolve() for sample in self.samples if sample.image_path.exists()})
        if not unique_images:
            raise ValueError("No gallery images found for dataset '{}'.".format(self.definition.key))

        batch_size = int(self.config.get("feature_cache", {}).get("image_batch_size", 32))
        current_batch_size = max(1, batch_size)
        cudnn_enabled = torch.backends.cudnn.enabled

        def encode_once(batch_value: int) -> torch.Tensor:
            chunks: List[torch.Tensor] = []
            total = len(unique_images)
            with torch.no_grad():
                for start in range(0, total, batch_value):
                    batch_paths = unique_images[start : start + batch_value]
                    images = torch.stack([_load_image(path, self.preprocess) for path in batch_paths]).to(self.device)
                    features = self.model.encode_images(images).cpu()
                    chunks.append(features)
                    _emit_status(status_callback, "Encoding gallery images", min(start + len(batch_paths), total), total)
            return torch.cat(chunks, dim=0)

        try:
            while current_batch_size >= 1:
                try:
                    self.gallery_paths = unique_images
                    self.gallery_matrix = encode_once(current_batch_size).to(self.device)
                    self._gallery_index = {path: index for index, path in enumerate(self.gallery_paths)}
                    self._save_gallery_cache()
                    return
                except RuntimeError as exc:
                    message = str(exc)
                    cudnn_error = "Unable to find a valid cuDNN algorithm" in message or "cuDNN" in message
                    if self.device.type != "cuda" or not cudnn_error:
                        raise
                    torch.cuda.empty_cache()
                    if current_batch_size > 1:
                        current_batch_size = max(1, current_batch_size // 2)
                        continue
                    if torch.backends.cudnn.enabled:
                        torch.backends.cudnn.enabled = False
                        continue
                    raise
        finally:
            torch.backends.cudnn.enabled = cudnn_enabled

    def ensure_ready(self, status_callback: StatusCallback = None) -> None:
        if self._ready:
            return
        self._build_model(status_callback=status_callback)
        _emit_status(status_callback, "Preparing gallery cache")
        if not self._load_gallery_cache():
            self._compute_gallery_matrix(status_callback=status_callback)
        _emit_status(status_callback, "Model is ready")
        self._ready = True

    def retrieve(
        self,
        sample: QuerySample,
        sketch_path: Path,
        text: str,
        top_k: int = 5,
    ) -> RetrievalResult:
        if not self._ready:
            self.ensure_ready()
        if self.model is None or self.sketch_transform is None or self.device is None or self.gallery_matrix is None:
            raise RuntimeError("Model is not ready.")
        if not sketch_path.exists():
            raise FileNotFoundError("Sketch not found: {}".format(sketch_path))

        top_k = max(1, min(int(top_k), len(self.gallery_paths)))

        with torch.no_grad():
            tokens = clip.tokenize([text], truncate=True).to(self.device)
            text_feat = self.model.encode_texts(tokens).squeeze(0)

            sketch_tensor = _load_sketch(sketch_path, self.sketch_transform).unsqueeze(0).to(self.device)
            sketch_feat = self.model.encode_sketches(sketch_tensor).squeeze(0)

            query_fusion = self.model.build_fusion_query(
                text_feat=text_feat.unsqueeze(0),
                sketch_feat=sketch_feat.unsqueeze(0),
            ).squeeze(0)
            base_signal = F.normalize((sketch_feat + text_feat) * 0.5, dim=0)
            query = F.normalize(self.gamma * query_fusion + (1.0 - self.gamma) * base_signal, dim=0)

            scores = torch.matmul(query.unsqueeze(0), self.gallery_matrix.t()).squeeze(0)
            sorted_indices = torch.argsort(scores, descending=True)

        hits: List[RetrievalHit] = []
        for rank, index_value in enumerate(sorted_indices[:top_k].tolist(), start=1):
            hits.append(
                RetrievalHit(
                    rank=rank,
                    score=float(scores[index_value].item()),
                    image_path=self.gallery_paths[index_value],
                )
            )

        ground_truth_rank = None
        target_index = self._gallery_index.get(sample.image_path.resolve())
        if target_index is not None:
            ground_truth_rank = int((sorted_indices == target_index).nonzero(as_tuple=False).item()) + 1

        return RetrievalResult(
            dataset_key=self.definition.key,
            sample=sample,
            sketch_path=sketch_path,
            query_text=text,
            hits=hits,
            ground_truth_rank=ground_truth_rank,
        )


class FusionWorkspace:
    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self._sessions: Dict[str, FusionDatasetSession] = {}

    def list_datasets(self) -> List[DatasetDefinition]:
        return list_dataset_definitions()

    def get_session(self, dataset_key: str) -> FusionDatasetSession:
        if dataset_key not in DATASETS:
            raise KeyError("Unknown dataset: {}".format(dataset_key))
        if dataset_key not in self._sessions:
            self._sessions[dataset_key] = FusionDatasetSession(DATASETS[dataset_key], cache_dir=self.cache_dir)
        return self._sessions[dataset_key]

    def asset_report(self) -> Dict[str, object]:
        return workspace_asset_report()

    def export_dataset_summary(self) -> str:
        summary = []
        for definition in self.list_datasets():
            session = self.get_session(definition.key)
            summary.append(
                json.dumps(
                    {
                        "key": definition.key,
                        "label": definition.label,
                        "num_samples": len(session.samples),
                        "config": str(definition.config_path),
                        "checkpoint": str(definition.checkpoint_path),
                        "text_index": str(definition.text_index_path),
                    },
                    ensure_ascii=False,
                )
            )
        return "\n".join(summary)
