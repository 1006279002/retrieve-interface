"""
STFusionIR 后端模块
==================
负责模型加载、特征提取、图库索引缓存和跨模态检索的核心逻辑。
不依赖任何 GUI 框架，可被命令行或 GUI 前端复用。

核心类:
  - DatasetDefinition: 数据集元信息（路径、模式等）
  - QuerySample:      单条查询样本（图片 + 多张草图 + 文本描述）
  - RetrievalHit:     检索结果中的单条命中
  - RetrievalResult:  一次完整检索的结果
  - FusionDatasetSession: 单个数据集的检索会话（管理模型、图库特征矩阵）
  - FusionWorkspace:      多数据集工作区（管理多个 session）
"""

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

# ---- 路径常量 ----
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent  # 项目根目录
ASSET_ROOT = Path(os.environ.get("STFUSION_ASSET_ROOT", WORKSPACE_ROOT / "stfusion_assets")).expanduser().resolve()  # 模型/配置资源目录
DEFAULT_CACHE_DIR = WORKSPACE_ROOT / ".stfusion_cache"  # 图库特征缓存目录

# ImageNet 标准归一化参数（用于草图预处理）
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# 状态回调类型：用于向 GUI 报告加载进度
# 参数: (消息文本, 当前进度, 总数)
StatusCallback = Optional[Callable[[str, Optional[int], Optional[int]], None]]


# ============================================================
# 数据模型
# ============================================================

@dataclass(frozen=True)
class DatasetDefinition:
    """数据集定义：包含路径、模式和元信息。frozen=True 保证不可变。"""
    key: str                     # 数据集唯一标识（如 "sketchy", "chair", "shoe"）
    label: str                   # 显示名称
    config_path: Path            # YAML 配置文件路径
    checkpoint_path: Path        # 融合模型权重 .pt 文件
    text_index_path: Path        # 文本索引文件（每行: image_path\ttext_description）
    image_root: Path             # 照片图库根目录
    sketch_root: Path            # 草图根目录
    mode: str                    # 数据集模式: "sketchy"（多类别）或 "paired"（配对草图-照片）


@dataclass
class QuerySample:
    """单条查询样本：包含参考照片、候选草图列表和文本描述。"""
    id: str                      # 样本唯一 ID
    label: str                   # 列表显示标签
    category: str                # 类别名（如 "airplane", "chair"）
    virtual_class: str           # 虚拟类别（用于 prompt 模板数量推断）
    text: str                    # 自然语言描述
    image_path: Path             # 参考（ground-truth）照片路径
    sketch_paths: List[Path]     # 该样本的所有候选草图路径


@dataclass
class RetrievalHit:
    """检索结果中的单条命中记录。"""
    rank: int                    # 排名（1-based）
    score: float                 # 余弦相似度分数
    image_path: Path             # 命中图片路径


@dataclass
class RetrievalResult:
    """一次完整检索的结果。"""
    dataset_key: str             # 数据集标识
    sample: QuerySample          # 查询样本
    sketch_path: Path            # 实际使用的草图路径
    query_text: str              # 实际使用的文本查询
    hits: List[RetrievalHit]     # Top-K 命中列表
    ground_truth_rank: Optional[int]  # GT 图片的排名（用于评估，None 表示不在图库中）


@dataclass
class AssetCheck:
    """资源文件检查结果。"""
    name: str                    # 资源名称
    path: Optional[Path]         # 文件路径
    exists: bool                 # 是否存在
    required: bool               # 是否必需
    note: str = ""               # 额外说明


# ============================================================
# 数据集注册表
# ============================================================
# 三个预定义数据集，各自使用独立的配置文件、checkpoint 和文本索引。
# mode 决定样本构建逻辑：
#   "sketchy" — 多类别手绘草图数据集（125 类），样本按类别组织
#   "paired"  — ChairV2 / ShoeV2 配对数据集，草图与照片一一对应

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


# ============================================================
# 工具函数
# ============================================================

def list_dataset_definitions() -> List[DatasetDefinition]:
    """返回所有已注册数据集的列表。"""
    return [DATASETS[key] for key in ("sketchy", "chair", "shoe")]


def _emit_status(callback: StatusCallback, message: str, current: Optional[int] = None, total: Optional[int] = None) -> None:
    """安全地调用状态回调（如果提供）。"""
    if callback is not None:
        callback(message, current, total)


def _load_yaml(path: Path) -> Dict[str, object]:
    """加载 YAML 配置文件。"""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _resolve_path(raw_path: str | Path, base_paths: Sequence[Path]) -> Path:
    """路径解析：支持绝对路径，相对路径依次在 base_paths 中查找。
    若都找不到，默认使用第一个 base_path 拼接。
    """
    candidate = Path(raw_path).expanduser()
    if candidate.is_absolute():
        return candidate
    for base_path in base_paths:
        resolved = (base_path / candidate).resolve()
        if resolved.exists():
            return resolved
    return (base_paths[0] / candidate).resolve()


def _select_device(preferred: Optional[str] = None) -> str:
    """智能选择计算设备：CUDA > MPS (macOS) > CPU。
    支持环境变量 STFUSION_DEVICE 覆盖。
    """
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
    """草图预处理 pipeline：Resize → CenterCrop → 灰度转3通道 → ToTensor → ImageNet 归一化。"""
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.Grayscale(num_output_channels=3),  # 草图转 3 通道以适配 ImageNet 预训练模型
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _load_image(path: Path, preprocess) -> torch.Tensor:
    """加载照片并应用 CLIP 预处理。"""
    with Image.open(path) as img:
        return preprocess(img.convert("RGB"))


def _load_sketch(path: Path, transform) -> torch.Tensor:
    """加载草图并应用草图专用变换。"""
    with Image.open(path) as img:
        return transform(img.convert("RGB"))


def _read_text_index(path: Path) -> List[tuple[str, str]]:
    """解析文本索引文件（TSV 格式）：image_path \\t text_description。"""
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
    """从文件名提取尾部数字用作排序键。如 "sketch_3.png" → ("sketch_", 3)。
    无数字时返回 (stem, -1)，排在最后。
    """
    match = re.search(r"(\d+)$", path.stem)
    if match:
        return path.stem[: match.start(1)], int(match.group(1))
    return path.stem, -1


def _sorted_sketches(paths: Iterable[Path]) -> List[Path]:
    """按文件名尾部数字排序草图路径。"""
    return sorted(paths, key=lambda item: _trailing_number_key(item))


def _sample_label(index: int, primary: str, secondary: str, text: str) -> str:
    """生成样本在列表中的显示标签：序号 | 类别 | 文件名 | 文本摘要。"""
    snippet = " ".join(text.split())[:28]
    return "{:04d} | {} | {} | {}".format(index + 1, primary, secondary, snippet)


def _build_sketchy_samples(definition: DatasetDefinition) -> List[QuerySample]:
    """为 Sketchy 模式构建样本列表。
    文本索引格式: transform_dir/category/filename.jpg \\t description
    草图为同一 transform_dir/category 下 filename-*.png。
    """
    samples: List[QuerySample] = []
    for index, (relative_image, raw_text) in enumerate(_read_text_index(definition.text_index_path)):
        rel_path = Path(relative_image)
        if len(rel_path.parts) < 3:
            continue
        transform_dir, category, filename = rel_path.parts[0], rel_path.parts[1], rel_path.parts[-1]
        image_path = (definition.image_root / rel_path).resolve()
        stem = Path(filename).stem
        # 同一类别目录下查找所有同名前缀的草图
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
    """为 Paired 模式（ChairV2/ShoeV2）构建样本列表。
    文本索引格式: filename \\t description
    草图匹配: stem_*.png 前缀匹配。
    """
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
    """根据数据集模式分派样本构建。"""
    if definition.mode == "sketchy":
        return _build_sketchy_samples(definition)
    if definition.mode == "paired":
        return _build_paired_samples(definition)
    raise ValueError("Unsupported dataset mode: {}".format(definition.mode))


def _resolve_required_files(definition: DatasetDefinition) -> List[AssetCheck]:
    """检查数据集所需的所有文件是否存在：config、checkpoint、文本索引、图片/草图根目录、
    以及 sketch backbone 预训练权重和 CLIP 模型权重。
    """
    config = _load_yaml(definition.config_path) if definition.config_path.exists() else {}
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}

    checks = [
        AssetCheck("{} config".format(definition.label), definition.config_path, definition.config_path.exists(), True),
        AssetCheck("{} checkpoint".format(definition.label), definition.checkpoint_path, definition.checkpoint_path.exists(), True),
        AssetCheck("{} text index".format(definition.label), definition.text_index_path, definition.text_index_path.exists(), True),
        AssetCheck("{} image root".format(definition.label), definition.image_root, definition.image_root.exists(), True),
        AssetCheck("{} sketch root".format(definition.label), definition.sketch_root, definition.sketch_root.exists(), True),
    ]

    # 检查 sketch 编码器预训练权重（如 QuickDraw pretrained ResNet-50）
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

    # 检查 CLIP 模型权重（支持本地文件或自动下载）
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
    """生成工作区完整资源报告（用于 audit.py 诊断）。"""
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


# ============================================================
# FusionDatasetSession — 单个数据集的检索会话
# ============================================================
# 管理一个数据集的完整生命周期：
#   1. 加载 YAML 配置 + 构建样本列表
#   2. 构建 SketchyFusionModel（CLIP + Sketch Encoder + Fusion Prompt）
#   3. 预计算图库中所有图像的特征矩阵（带缓存）
#   4. 执行跨模态检索（Sketch + Text → 图像排名）

class FusionDatasetSession:
    """单个数据集的检索会话。惰性初始化：首次 ensure_ready() 时才加载模型和图库。"""

    def __init__(self, definition: DatasetDefinition, cache_dir: Optional[Path] = None) -> None:
        self.definition = definition
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.config = _load_yaml(self.definition.config_path)      # YAML 配置字典
        self.samples = _build_samples(self.definition)              # 所有查询样本
        self.gamma = float(self.config.get("retrieval", {}).get("gamma", 0.6))  # 融合权重：gamma * fusion_query + (1-gamma) * avg_baseline

        # 延迟初始化的成员
        self.device_name: Optional[str] = None
        self.device: Optional[torch.device] = None
        self.preprocess = None           # CLIP 图像预处理
        self.sketch_transform = None     # 草图预处理
        self.model: Optional[SketchyFusionModel] = None
        self.gallery_paths: List[Path] = []            # 图库图像路径列表
        self.gallery_matrix: Optional[torch.Tensor] = None  # 图库特征矩阵 [N, D]
        self._gallery_index: Dict[Path, int] = {}       # 路径 → 矩阵行索引映射
        self._ready = False

    def _resolve_num_template_prompts(self) -> int:
        """推断模板 prompt 数量：优先用配置值，否则用数据集中唯一 virtual_class 数量。"""
        model_cfg = self.config.get("model", {})
        explicit = model_cfg.get("num_template_prompts")
        if explicit is not None:
            return max(1, int(explicit))
        return max(1, len({sample.virtual_class for sample in self.samples}))

    def _cache_path(self) -> Path:
        """图库特征缓存的磁盘路径。"""
        return self.cache_dir / "{}_gallery.pt".format(self.definition.key)

    def _build_model(self, status_callback: StatusCallback = None) -> None:
        """构建融合检索模型：
        1. 加载 CLIP（ViT-B/32 或自定义路径）
        2. 加载 Sketch Encoder（ResNet-50 + QuickDraw 预训练权重）
        3. 构建 FusionPromptModule（可学习的类感知模板 prompt）
        4. 加载训练好的 checkpoint，冻结所有特征提取器
        """
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
        """尝试从磁盘加载图库特征缓存。
        缓存有效性验证：检查 dataset key + config/checkpoint/text_index 的路径和修改时间。
        任何不匹配都会导致缓存失效，触发重新计算。
        """
        cache_path = self._cache_path()
        if not cache_path.exists() or self.device is None:
            return False

        payload = torch.load(str(cache_path), map_location="cpu")
        # 验证缓存与当前数据集配置是否一致
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
        """将图库特征矩阵保存到磁盘缓存，附带数据集元信息用于后续验证。"""
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
        """使用 CLIP 图像编码器预计算图库中所有图像的特征矩阵。
        支持批量编码；遇到 cuDNN 错误时自动降级（减小 batch size 或禁用 cuDNN）。
        结果自动缓存到磁盘。
        """
        if self.model is None or self.preprocess is None or self.device is None:
            raise RuntimeError("Model session is not initialized.")

        # 收集所有唯一图像路径（去重）
        unique_images = sorted({sample.image_path.resolve() for sample in self.samples if sample.image_path.exists()})
        if not unique_images:
            raise ValueError("No gallery images found for dataset '{}'.".format(self.definition.key))

        batch_size = int(self.config.get("feature_cache", {}).get("image_batch_size", 32))
        current_batch_size = max(1, batch_size)
        cudnn_enabled = torch.backends.cudnn.enabled  # 保存原始 cuDNN 状态

        def encode_once(batch_value: int) -> torch.Tensor:
            """按指定 batch size 编码全部图像，返回特征矩阵 [N, D]。"""
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
                    # 仅在 CUDA cuDNN 相关错误时才尝试降级策略
                    cudnn_error = "Unable to find a valid cuDNN algorithm" in message or "cuDNN" in message
                    if self.device.type != "cuda" or not cudnn_error:
                        raise
                    torch.cuda.empty_cache()
                    if current_batch_size > 1:
                        current_batch_size = max(1, current_batch_size // 2)  # 减半 batch size 重试
                        continue
                    if torch.backends.cudnn.enabled:
                        torch.backends.cudnn.enabled = False  # 禁用 cuDNN 重试
                        continue
                    raise
        finally:
            torch.backends.cudnn.enabled = cudnn_enabled  # 恢复原始 cuDNN 状态

    def ensure_ready(self, status_callback: StatusCallback = None) -> None:
        """确保会话就绪：加载模型 → 加载/构建图库特征矩阵。幂等操作。"""
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
        """执行跨模态检索：Sketch + Text → Top-K 图像排名。

        检索流程：
          1. 用 CLIP 编码文本 → text_feat
          2. 用 Sketch Encoder 编码草图 → sketch_feat
          3. 通过 FusionPromptModule 融合两个模态 → fusion_query
          4. 用 gamma 插值：final_query = gamma * fusion_query + (1-gamma) * avg(sketch, text)
          5. final_query 与预计算的图库矩阵做余弦相似度，取 Top-K
          6. 同时计算 ground-truth 图片的排名（用于评估）
        """
        if not self._ready:
            self.ensure_ready()
        if self.model is None or self.sketch_transform is None or self.device is None or self.gallery_matrix is None:
            raise RuntimeError("Model is not ready.")
        if not sketch_path.exists():
            raise FileNotFoundError("Sketch not found: {}".format(sketch_path))

        top_k = max(1, min(int(top_k), len(self.gallery_paths)))

        with torch.no_grad():
            # 1. 文本编码
            tokens = clip.tokenize([text], truncate=True).to(self.device)
            text_feat = self.model.encode_texts(tokens).squeeze(0)

            # 2. 草图编码
            sketch_tensor = _load_sketch(sketch_path, self.sketch_transform).unsqueeze(0).to(self.device)
            sketch_feat = self.model.encode_sketches(sketch_tensor).squeeze(0)

            # 3. 融合查询
            query_fusion = self.model.build_fusion_query(
                text_feat=text_feat.unsqueeze(0),
                sketch_feat=sketch_feat.unsqueeze(0),
            ).squeeze(0)
            # 4. gamma 加权插值：在纯融合信号和简单平均之间平衡
            base_signal = F.normalize((sketch_feat + text_feat) * 0.5, dim=0)
            query = F.normalize(self.gamma * query_fusion + (1.0 - self.gamma) * base_signal, dim=0)

            # 5. 余弦相似度排序
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

        # 6. 查找 ground-truth 图片排名
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


# ============================================================
# FusionWorkspace — 多数据集工作区
# ============================================================
# 管理多个 FusionDatasetSession，按需创建和缓存。

class FusionWorkspace:
    """多数据集工作区管理器。维护 dataset_key → FusionDatasetSession 的映射。"""

    def __init__(self, cache_dir: Optional[Path] = None) -> None:
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self._sessions: Dict[str, FusionDatasetSession] = {}

    def list_datasets(self) -> List[DatasetDefinition]:
        """列出所有可用数据集的元信息。"""
        return list_dataset_definitions()

    def get_session(self, dataset_key: str) -> FusionDatasetSession:
        """获取指定数据集的会话（惰性创建 + 缓存）。"""
        if dataset_key not in DATASETS:
            raise KeyError("Unknown dataset: {}".format(dataset_key))
        if dataset_key not in self._sessions:
            self._sessions[dataset_key] = FusionDatasetSession(DATASETS[dataset_key], cache_dir=self.cache_dir)
        return self._sessions[dataset_key]

    def asset_report(self) -> Dict[str, object]:
        """生成工作区资源状态报告。"""
        return workspace_asset_report()

    def export_dataset_summary(self) -> str:
        """导出所有数据集的摘要信息（JSON lines 格式）。"""
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
