from __future__ import annotations

import sys


def _configure_high_dpi() -> None:
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
