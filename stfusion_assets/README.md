# STFusion Assets

这个目录是 GUI 运行时所需的最小模型资产集合。

保留内容：

- `configs/`
- `checkpoints/`
- `model/`

删除 `models/` 后，只要本目录和 `datasets/` 仍在，GUI 仍可运行。

如果要把本目录移动到其他位置，可以通过环境变量指定：

```bash
STFUSION_ASSET_ROOT=/absolute/path/to/stfusion_assets python run_stfusion_gui.py
```
