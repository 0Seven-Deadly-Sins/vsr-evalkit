# vsr-evalkit

<a href="README.md"><kbd>简体中文</kbd></a> <a href="README_EN.md"><kbd>English</kbd></a>

一个独立的视频超分辨率（Video Super-Resolution，VSR）结果评价工具包。将模型结果路径、数据集、指标与评价协议分开配置，便于用同一套流程比较多个模型。

这里的“模型”指已生成结果的路径标识。项目不运行 VSR 推理，也无需安装待评模型的推理或训练代码。配置好 `eval.local.yaml` 后，即可按模型和数据集 ID 运行：

```bash
vsr-eval run --model dove --dataset udm10
vsr-eval run --model dove,seedvr2 --dataset udm10,spmcs
```

[快速开始](#快速开始) · [配置自己的结果](#配置自己的结果) · [指标与协议](#指标与协议) · [完整指标环境](#完整指标环境) · [验证状态](#验证状态) · [文档](#文档)

## 支持范围

- **输入形式**：PNG/JPEG 帧序列，以及 MP4、MKV、MOV、AVI 视频。预测与参考真值（GT）可以采用不同存储形式；命名差异可通过显式映射文件处理。
- **视频尺寸**：不同视频可以使用不同分辨率和长宽比，同一序列内部尺寸需固定。各指标仍受其输入尺寸要求和可用内存、显存限制。
- **默认数据范围**：SDR、8-bit 数据。视频文件需为恒定帧率；帧序列的未知 FPS 保持未知。VFR、HDR、高位深和插帧输出的自动时间对齐不在默认支持范围内。

## 快速开始

要求 Python ≥3.10，已验证版本为 3.10 和 3.11。以下安装示例使用 Bash：

```bash
git clone https://github.com/0Seven-Deadly-Sins/vsr-evalkit.git
cd vsr-evalkit
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Windows PowerShell 的虚拟环境激活命令为 `.venv\Scripts\Activate.ps1`。

随后运行无需评价权重的 CPU 演示：

```bash
python scripts/make_demo.py --output demo-input
vsr-eval doctor --config demo-input/eval.yaml --model demo --dataset synthetic
vsr-eval run --config demo-input/eval.yaml --model demo --dataset synthetic --device cpu
```

演示脚本在新的 `demo-input` 目录中生成少量合成 PNG，并配置 PSNR/SSIM；无需下载数据集、模型权重或安装 PyTorch。`doctor` 检查输入、依赖和本地文件，不运行评价模型；实际计算由 `run` 执行。

## 配置自己的结果

将下面的配置保存为仓库根目录下的 `eval.local.yaml`，替换其中的示例路径。该文件已被 Git 忽略。这里选择的 UDM10 和 SPMCS 都需要提供正确配对的 GT，评价指标为无需权重的 PSNR/SSIM。

```yaml
version: 1
models:
  dove:
    pred_root: /data/results/dove/{dataset}
  seedvr2:
    pred_root: /data/results/seedvr2/{dataset}
datasets:
  udm10:
    gt_root: /data/UDM10/GT
  spmcs:
    gt_root: /data/SPMCS/GT
metrics: [psnr, ssim]
protocol: standard-rgb-v1
runtime:
  device: cpu
output_root: runs
```

`{dataset}` 会替换为当前数据集 ID，例如 `/data/results/dove/udm10`。模型和数据集 ID 均可自行命名；它们是配置项，不会触发对应模型推理或数据集下载。

预测与 GT 根目录中的序列默认按 clip ID 配对，例如双方都包含 `clip001/` 帧目录，或分别包含 `clip001.mp4` 和 `clip001/`。帧序列按自然顺序读取，双方均为帧目录时还会核对帧标识；帧数、尺寸和可用时间信息统一检查。目录布局和名称映射见[数据集配置](docs/DATASETS_ZH.md)。

```bash
vsr-eval doctor --model dove --dataset udm10
vsr-eval run --model dove,seedvr2 --dataset udm10,spmcs
```

不指定 `--config` 时，命令读取当前目录下的 `eval.local.yaml`。多个模型与多个数据集按全部组合运行，上面的批量命令会执行四组评价。

VideoLQ 等无 GT 数据集不应配置 `gt_root`，也不能使用上面的 PSNR/SSIM 指标集合。无 GT 用法见[完整指标环境](#完整指标环境)。

## 指标与协议

表中为命令行使用的指标 ID；↑ 表示越高越好，↓ 表示越低越好。

| 类型 | 指标 ID 与方向 | 采用的版本或定义 | 需要 GT |
|---|---|---|---|
| 全参考图像 | `psnr` ↑、`ssim` ↑ | 独立 float64 实现，与 BasicSR / PyIQA 对照 | 是 |
| 全参考感知 | `lpips` ↓、`dists` ↓ | LPIPS-Alex v0.1、DISTS；PyIQA 0.1.15.post2 | 是 |
| 无参考图像 | `niqe` ↓ | 原始 NIQE 统计参数；PyIQA 0.1.15.post2 | 否 |
| 无参考图像 | `musiq` ↑、`clipiqa` ↑ | MUSIQ-KonIQ、普通 CLIP-IQA RN50 | 否 |
| 无参考视频 | `dover` ↑、`fastvqa` ↑、`fastervqa` ↑ | 固定官方源码、采样配置与权重版本 | 否 |
| 时序一致性 | `ewarp` ↓ | RAFT Things 双向光流与遮挡掩码 RGB warping MSE | 否 |

`metrics: auto` 会在有 GT 时选择全部 11 项，无 GT 时选择 7 项无参考、视频质量和时序指标。它不会自动跳过缺失依赖或权重；显式请求有参考指标却没有 GT 时，预检查会报错。

默认 `standard-rgb-v1` 协议：

- 严格检查样本、帧数、尺寸和可用时间信息，不自动缩放、裁齐或截短。
- PSNR/SSIM 使用 RGB、零裁边；SSIM 使用 11×11、σ=1.5 的高斯窗口。`standard-y-v1` 可显式切换这两项的 BT.601 Y 通道计算；LPIPS/DISTS 始终接收 RGB。
- 图像指标逐帧计算后求视频均值；时序指标对相邻帧对求视频均值；最后均按视频等权平均。视频质量模型按固定上游规则采样，每个视频得到一个分数。
- RAFT 默认在原尺寸估计光流。显存不足时明确失败；可显式选择 `flow-960-v1`，只缩小光流估计输入，恢复光流向量后仍在原尺寸图像上计算误差。

完整公式、采样规则、聚合方式和兼容预设见[指标协议](docs/PROTOCOLS_ZH.md)。指标、预处理和权重版本共同决定分数含义，不能仅凭相同指标名称直接比较结果。

## 完整指标环境

PSNR/SSIM 以外的指标需要额外依赖，以及本地权重或统计参数；DOVER、FAST-VQA/FasterVQA 和 RAFT 还需要固定版本的上游指标源码。完整评价环境主要面向 Linux + NVIDIA GPU，安装和配置步骤见[权重与环境指南](docs/WEIGHTS.md)。

```bash
vsr-eval weights info
python scripts/source_commands.py
```

第一条命令列出所需文件及下载来源；第二条只打印固定版本的源码获取命令。工具不会自动下载评价权重、隐式骨干或 NIQE 统计参数。请按指南手动准备文件，并在配置中填写 `sources` 和 `weights`。

完整配置模板为 [examples/eval.yaml](examples/eval.yaml)。环境与文件准备好后，将 `metrics` 设为 `auto`，先运行 `doctor`，再执行评价。模板中已注册无 GT 的 `videolq: {}`；若沿用上面的基础配置，需先在 `datasets` 下添加这一项。准备好对应结果目录后，可运行：

```bash
vsr-eval run --model dove --dataset videolq --metrics auto
```

该命令使用七项无参考、视频和时序指标。所需文件缺失时会在运行前报错；`doctor` 通过也不代表真实模型的加载与数值验证已完成。

## 输出与比较

默认输出到 `runs/` 下的新目录：

| 文件 | 内容 |
|---|---|
| `report.json` | 完整运行记录、逐视频结果和数据集汇总 |
| `details.csv` | 图像指标的逐帧结果、时序指标的帧对结果及视频指标记录 |
| `summary.csv` | 各模型、数据集和指标的汇总 |
| `comparison.md` | 模型比较表 |

结果记录协议标识、依赖版本、源码 commit、权重 SHA-256、采样索引、覆盖数量和失败原因。不同协议、权重或样本集合的结果不会自动混合排名；存在失败或覆盖不完整时，该指标的数据集汇总留空。单帧视频的时序指标记为不可计算，不用零分代替。

## 验证状态

- **源码核查**：记录 VSR 工作及指标上游的固定版本、评价入口与协议差异；已确认问题及证据见[源码审计](docs/AUDIT.md)。
- **已通过**：无权重测试、经典指标与上游函数的数值对照、合成数据 CLI 评价及 Linux CPU CI。环境、用例和已验证提交见[验证记录](docs/VALIDATION_ZH.md)。
- **待完成**：真实预训练权重的数值对照、Linux CUDA 运行验证和论文表格复现。当前不据此声称已复现论文分数。

运行本地测试：

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

部分时序与后端测试需要 PyTorch；未安装时会跳过，完整验证环境见验证记录。仓库不包含数据集、VSR 生成结果或评价权重。

## 文档

| 内容 | 入口 |
|---|---|
| VSR 工作与指标上游调研 | [源码调研](docs/SURVEY_ZH.md) |
| 公式、预处理、采样与聚合 | [指标协议](docs/PROTOCOLS_ZH.md) |
| 已确认的问题与实现差异 | [源码审计](docs/AUDIT.md) |
| 数据集示例、目录布局与名称映射 | [数据集配置](docs/DATASETS_ZH.md) |
| 依赖安装、权重来源与离线配置 | [权重与环境](docs/WEIGHTS.md) |
| 已执行检查及真实权重验证方法 | [验证记录](docs/VALIDATION_ZH.md) |
| 第三方许可与引用 | [第三方声明](THIRD_PARTY_NOTICES_ZH.md) |

本项目代码采用 [Apache-2.0](LICENSE) 许可证。可选指标的源码和权重遵循各自的许可条款，部分上游组件仅允许非商业使用，详见第三方声明。
