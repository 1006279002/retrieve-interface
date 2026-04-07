from __future__ import annotations

import json

from .backend import workspace_asset_report


def main() -> int:
    print(json.dumps(workspace_asset_report(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
