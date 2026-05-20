"""
stfusion_gui — STFusionIR 跨模态检索 GUI 包
============================================
提供基于 PyQt5 的桌面检索界面和底层推理引擎。

公开 API:
  - FusionWorkspace       — 多数据集工作区管理器
  - MainWindow            — 主窗口（PyQt5）
  - launch_app            — 启动 GUI 应用
  - workspace_asset_report — 资源文件诊断报告
"""

from .backend import FusionWorkspace, workspace_asset_report
from .ui import MainWindow, launch_app

__all__ = ["FusionWorkspace", "MainWindow", "launch_app", "workspace_asset_report"]
