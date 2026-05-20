"""
stfusion_gui.inference — 推理引擎子包
=====================================
包含 CLIP 模型加载和融合检索模型实现。

子模块:
  - clip         — CLIP 模型加载、预处理、tokenization
  - cstbir_fusion_model — 融合检索模型（SketchyFusionModel + 损失函数）
"""

from . import clip
from .cstbir_fusion_model import SketchyFusionModel

__all__ = ["clip", "SketchyFusionModel"]
