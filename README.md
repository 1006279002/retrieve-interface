# STFusionIR PyQt Frontend

一个用于 `Sketchy`、`Chair`、`Shoe` 三套数据的草图 + 文本联合检索桌面前端。

当前实现支持：

- 从查询样本中选择一个条目
- 在多个草图候选中点选一个草图
- 手动编辑文本描述
- 检索并展示 top-5 图片结果

## 项目结构

```text
.
├── run_stfusion_gui.py
├── stfusion_gui/
├── stfusion_assets/
├── datasets/
└── datasets.zip
```

- `stfusion_gui/`：PyQt 界面和推理逻辑
- `stfusion_assets/`：推理所需的配置、checkpoint 和草图 backbone 权重
- `datasets/`：查询索引、参考图、草图候选和检索图库
- `datasets.zip`：`datasets/` 的压缩归档，当前生成大小约 `4.1G`

## 运行前提

当前 GUI 已经和旧的 `models/` 目录解耦，但运行时仍然需要：

- `stfusion_gui/`
- `stfusion_assets/`
- `run_stfusion_gui.py`
- `datasets/`

其中 `datasets/` 是必需的，因为当前版本会：

- 从 `datasets/sketchy_test.txt`、`datasets/chair_test.txt`、`datasets/shoe_test.txt` 构建查询样本
- 读取本地参考图和草图候选
- 从本地图片库计算或加载 gallery 特征

## 安装

建议环境：

- Python `3.10` 或 `3.11`
- macOS：`mps` 或 `cpu`
- Windows：`cuda` 或 `cpu`

先按你的平台安装 `torch` 和 `torchvision`，再安装其余依赖：

```bash
pip install -r requirements-gui.txt
```

## 准备数据

如果仓库中没有直接保留 `datasets/` 目录，可以把 `datasets.zip` 解压到项目根目录，解压后结构应为：

```text
datasets/
```

也就是说，程序期望看到的是：

```text
./datasets/...
```

而不是：

```text
./somewhere_else/datasets/...
```

## 启动

在项目根目录运行：

```bash
python run_stfusion_gui.py
```

强制指定设备时：

```bash
# macOS / Linux
STFUSION_DEVICE=cpu python run_stfusion_gui.py
```

```powershell
# Windows PowerShell
$env:STFUSION_DEVICE = "cpu"
python run_stfusion_gui.py
```

可选值：`cpu`、`mps`、`cuda`

如果你把模型资产移动到了别的位置，可以通过环境变量覆盖：

```bash
STFUSION_ASSET_ROOT=/absolute/path/to/stfusion_assets python run_stfusion_gui.py
```

## 运行机制

- 运行时不再依赖 `models/FG_CSTBIR/src`
- 运行时不再依赖 `models/FG_CSTBIR/manifests`
- 首次进入某个数据集时，会对 gallery 编码并缓存到 `.stfusion_cache/`
- 后续启动会优先复用缓存，除非 `config`、`checkpoint` 或文本索引发生变化

## 资产审计

可以用下面的命令检查当前工作区里哪些文件是运行必须的：

```bash
python -m stfusion_gui.audit
```

## 大文件说明

当前仓库的运行资产较大：

- `datasets.zip` 当前约 `4.1G`
- `datasets/` 更适合作为单独归档分发，而不是普通源码文件
- `stfusion_assets/` 也包含大体积权重文件

如果你要把项目放到 GitHub，比较稳妥的方式是：

- 仓库中保留代码和说明
- 将 `datasets.zip` 作为单独下载文件分发
- 如有需要，再把 `stfusion_assets/` 也按同样方式单独分发

## 常见问题

### 首次启动会下载 CLIP 权重

如果本机没有缓存 `ViT-B/32`，第一次启动时会尝试下载到：

```text
~/.cache/clip
```

如果你希望完全离线运行，可以在对应配置文件里增加：

```yaml
model:
  clip_model_path: /absolute/path/to/ViT-B-32.pt
```

### 想换成自己的草图输入

当前版本先实现的是“从候选草图中选择一个再检索”。如果后续需要，也可以继续扩展：

- 本地上传草图
- 画板手绘
- 导出检索结果
