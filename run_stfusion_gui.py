"""
STFusionIR Retrieval Desktop — 启动入口
========================================
基于 Sketch + Text 融合的跨模态图像检索桌面应用。
通过 PyQt5 提供图形界面，支持 Sketchy / Chair / Shoe 三个数据集。

用法: python run_stfusion_gui.py
依赖: PyQt5, torch, torchvision, Pillow, PyYAML, tqdm, ftfy, regex
"""

from __future__ import annotations

import sys


def _configure_high_dpi() -> None:
    """在高 DPI 显示器上启用 Qt 的 HiDPI 缩放，避免界面模糊。"""
    try:
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QApplication

        if hasattr(Qt, "AA_EnableHighDpiScaling"):
            QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        if hasattr(Qt, "AA_UseHighDpiPixmaps"):
            QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    except Exception:
        pass


def main() -> int:
    """应用主入口：配置高 DPI → 导入 GUI → 启动事件循环。"""
    _configure_high_dpi()
    try:
        from stfusion_gui import launch_app
    except ImportError as exc:
        print("Missing dependency:", exc)
        print("Install PyQt5, torch, torchvision, Pillow, PyYAML, tqdm, ftfy, and regex first.")
        return 1
    return launch_app()


if __name__ == "__main__":
    sys.exit(main())
