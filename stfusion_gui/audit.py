"""
STFusionIR 资源诊断工具
=======================
检查所有必需文件（模型、配置、数据）是否存在，输出 JSON 格式报告。

用法: python -m stfusion_gui.audit
"""

from __future__ import annotations

import json

from .backend import workspace_asset_report


def main() -> int:
    """输出工作区完整资源报告（JSON 格式）。"""
    print(json.dumps(workspace_asset_report(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
