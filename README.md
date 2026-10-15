# vsr-evalkit

[![简体中文](https://img.shields.io/badge/Language-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-blue?style=for-the-badge)](README.md) [![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](README_EN.md)

独立评价已经生成的 Video Super-Resolution 结果。模型、数据集、指标与评价协议分开配置；无需安装 DOVE、SeedVR2 等 VSR 模型代码。

```bash
vsr-eval run --model dove --dataset udm10
vsr-eval run --model dove,seedvr2 --dataset udm10,videolq
```

**默认离线，不下载任何评价权重、骨干或统计参数。** 完整环境以 Linux + NVIDIA GPU 为目标；PSNR/SSIM 无需 PyTorch、权重或 GPU。仓库不包含数据和模型生成结果。[English quick start](README_EN.md)

## 安装与第一个可运行示例

推荐 Python 3.10 或 3.11。先安装基础包：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/make_demo.py --output demo-input
vsr-eval doctor --config demo-input/eval.yaml --model demo --dataset synthetic
vsr-eval run --config demo-input/eval.yaml --model demo --dataset synthetic --device cpu
```

Windows 的激活命令为 `.venv\Scripts\Activate.ps1`。示例只生成小型合成 PNG，无需下载数据或权重。

完整指标环境先安装匹配的 PyTorch/torchvision CUDA 构建，再执行：

```bash
python -m pip install -r requirements/deep-linux.txt
python scripts/source_commands.py
```

第二条命令**只打印** DOVER、FAST-VQA 和 RAFT 的固定 commit 源码配置命令。手动安装对应源码并配置本地权重后才能运行完整集合；详细步骤见 [权重与环境](docs/WEIGHTS.md)。上游非商业许可证需单独遵守。

## 配置自己的模型和数据集

将 [examples/eval.yaml](examples/eval.yaml) 复制为仓库根目录的 `eval.local.yaml`，修改一次路径即可。该本机文件受 `.gitignore` 保护。

```yaml
version: 1
models:
  dove:
    pred_root: /data/results/dove/{dataset}
  my_model:
    pred_root: /data/results/my_model/{dataset}
datasets:
  udm10:
    gt_root: /data/UDM10/GT
  videolq: {}
metrics: [psnr, ssim]
protocol: standard-rgb-v1
runtime:
  device: cpu
output_root: runs
```

`metrics: auto` 表示有 GT 时选择全部 11 项，无 GT 时选择 7 项无参考/视频/时序指标。选中的权重必须全部准备好；缺失会在运行前列出，而不会偷偷跳过或下载。无 GT 时显式指定 PSNR 等有参考指标会报错。

常见输入布局：

```text
results/dove/udm10/           GT/
  clip001.mp4                  clip001/000000.png ...
  clip002/000000.png ...        clip002.mkv
```

支持 PNG/JPEG 帧序列、MP4/MKV/MOV/AVI，以及 GT 与预测混合存储。按 clip ID 严格配对；帧目录采用自然排序，双方帧目录还会核对帧编号（忽略零填充和前缀）。名称不同可用 [mapping.csv](examples/mapping.csv)：设置 `models.dove.mappings.udm10: maps/udm10.csv` 或数据集的 `mapping_file`。CSV 的 `pred` 相对该模型结果根目录，`gt` 相对数据集 GT 根目录；也可使用绝对路径。

单个视频文件或纯帧目录也可作为根路径，默认 clip ID 为其文件主名/目录名；单序列名称不同仍需映射。非序号命名但仅按顺序对应时，显式设置 `protocol.frame_pairing: ordinal`。

数据集可配置真实 `fps`，用来补充帧目录时间信息；若与视频时间戳不符则报错。未知 FPS 保留为未知，视频指标的官方采样按帧索引执行，不伪造 8/24/30 FPS。首版不支持 VFR、HDR、高位深及插帧输出的自动对齐。

## 指标与协议

| 类别 | 首版指标 | 实现来源 |
|---|---|---|
| 有参考 | PSNR、SSIM | 独立 float64 实现，与 BasicSR / PyIQA 对照 |
| 有参考感知 | LPIPS-Alex v0.1、DISTS | PyIQA 0.1.15.post2 |
| 无参考图像 | NIQE、MUSIQ-KonIQ、CLIP-IQA | PyIQA 0.1.15.post2 |
| 无参考视频 | DOVER、FAST-VQA、FasterVQA | 各自官方固定源码、采样配置与权重 |
| 时序 | RAFT masked RGB warping MSE | 双向原版 RAFT + 明确有效区域/遮挡掩码 |

默认 `standard-rgb-v1`：RGB、零裁边、全部帧、严格尺寸/帧数配对，逐帧→视频均值→视频等权平均。PSNR 为零误差时返回正无穷；SSIM 11×11、σ=1.5、valid 区域，保留负的结构响应。LPIPS/DISTS 始终使用 RGB。

```bash
# 查看注册项、检查环境和本地文件
vsr-eval list --config eval.local.yaml
vsr-eval weights info --metric lpips
vsr-eval doctor --model dove --dataset udm10

# 只算无需权重的两项
vsr-eval run --model dove --dataset udm10 --metrics psnr,ssim --device cpu

# 显式切换亮度通道或节省光流显存的协议
vsr-eval run --model dove --dataset udm10 --protocol standard-y-v1
vsr-eval run --model dove --dataset udm10 --protocol flow-960-v1
```

原尺寸 RAFT 在 4K 上可能耗尽显存，默认不会自动降低分辨率。`flow-960-v1` 只限制光流估计长边，并正确恢复光流向量；误差仍在原尺寸图像上计算，协议标识不同。DOVER 等自身定义的视图采样照常执行。跨视频可使用任意长宽尺寸，单个序列内部尺寸需固定。

`dove-code-alignment-v1` 提供 DOVE 已公开的对齐方式及指定 PyIQA 版本的默认口径。DOVE 原项目未固定 PyIQA 版本，因此此预设不能承诺重现论文表格，也不复现已发现的错误。完整公式、预设差异和短视频规则见 [PROTOCOLS.md](docs/PROTOCOLS_ZH.md)。

## 输出与失败处理

每次运行创建新的目录，包含 `report.json`、`details.csv`、`summary.csv`、`comparison.md`。记录逐帧/帧对、逐视频和数据集分数，协议、采样索引、随机种子、依赖版本、源码 commit、权重 SHA-256、有效覆盖数量和错误原因。

不同协议、权重、分辨率、长度或样本集合生成不同 `comparison_group`。只在相同数据集、指标和组内比较，不自动给出混合口径排名。任何视频计算失败或帧对仅部分有效，该指标的数据集汇总置空，禁止对成功子集偷偷平均。单帧时序指标为不可计算，不记零分。JSON 中无穷表示为字符串 `"+inf"`。

退出码：`0` 完成（允许明确的整项不可计算），`1` 存在计算失败/不完整，`2` 配置、输入或预检查失败。`--output` 必须是不存在的新目录。

## 调研、正确性与验证状态

- [21 个固定源码的调研与整合决策](docs/SURVEY_ZH.md)
- [已确认的问题、协议差异和证据](docs/AUDIT.md)
- [权重来源及另一台机器的环境配置](docs/WEIGHTS.md)
- [数据集版本、配对及配置说明](docs/DATASETS_ZH.md)
- [已执行与待执行的验证](docs/VALIDATION_ZH.md)
- [第三方许可证与引用](THIRD_PARTY_NOTICES_ZH.md)

```bash
python -m pip install -e '.[test]'
python -m pytest -q
# 可选：实际经典指标源码对照；--fetch 只获取固定 Python 源文件
python scripts/verify_references.py --cache reference-cache --fetch
# 用户在另一台机器备齐权重后执行，仍禁止联网兜底
python scripts/validate_local_weights.py --config eval.local.yaml --output local-validation.json
```

目前无权重测试与经典指标对照已执行。真实预训练权重、CUDA、论文表格数值复现尚未验证。未用模拟分数代替这些验证。仓库以可追溯协议为目标，不将任何单一指标作为视觉质量的充分证明。
