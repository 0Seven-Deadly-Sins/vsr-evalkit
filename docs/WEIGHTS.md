# 手动配置环境与权重 / Offline setup

本工具不下载权重。`vsr-eval weights info` 输出完整机器可读清单（位于 `src/vsr_evalkit/resources/weights.json`），包括隐式骨干。模型构造前检查文件存在且非空；运行时计算 SHA-256，并核对已知完整哈希或官方文件名中的哈希前缀。未知隐式下载请求会明确失败，不使用随机权重替代。

## Linux GPU 环境

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Example only: adjust CUDA wheel index to your driver. This installs runtimes, not pretrained weights.
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements/deep-linux.txt
```

`pyiqa==0.1.15.post2`、`timm==0.9.16` 被固定；不可随意换成 latest 后继续混用旧结果。decord/skvideo 是上游导入依赖，本工具统一以 PyAV 解码。只算 PSNR/SSIM 不需安装以上深度环境。Linux/CUDA 仍需用户机器上的真实权重验证，不能将 CPU 测试当作 GPU 测试。

基础包固定 `av==14.2.0`，该版本提供目标 Python/Linux/Windows wheel；14.3/14.4 的源码包会触发额外 FFmpeg 编译，因此不进入默认环境。

## 指标源码

```bash
python scripts/source_commands.py
```

此命令打印三组 `git clone --no-checkout` 和 `git checkout --detach <full commit>`，不执行任何下载。手动执行打印的**源码**命令后，配置 `sources.dover`、`sources.fastvqa` 和 `sources.raft`。固定版本、许可证和候选文件见 [SURVEY](SURVEY.md)；完整 commit 在 `src/vsr_evalkit/resources/sources.json`。不需安装这些项目的 VSR/训练模型。

后端会核对源码 HEAD 并拒绝有 tracked 修改的源码。权重放在独立 `weights/`，不改上游的 YAML。模型及其采样配置来自该固定 checkout，权重路径由本项目配置覆盖。DOVER 和 FAST-VQA 的老代码可有 CUDA/依赖差异；如果需要分开环境，可在各环境安装本项目并分别选择指标，结果保留完整来源，不能将不同环境的比较组强行合并。

## 必须手动准备的文件

| 配置键 | 文件/版本 | 官方/上游发布源 |
|---|---|---|
| alexnet | alexnet-owt-7be5be79.pth | [torchvision](https://download.pytorch.org/models/alexnet-owt-7be5be79.pth) |
| lpips_alex | LPIPS_v0.1_alex-df73285e.pth | [PyIQA weights](https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/LPIPS_v0.1_alex-df73285e.pth) |
| vgg16 | vgg16-397923af.pth | [torchvision](https://download.pytorch.org/models/vgg16-397923af.pth) |
| dists | DISTS_weights-f5e65c96.pth | [PyIQA weights](https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/DISTS_weights-f5e65c96.pth) |
| niqe_params | niqe_modelparameters.mat，original 统计参数 | [PyIQA weights](https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/niqe_modelparameters.mat) |
| musiq_koniq | musiq_koniq_ckpt-e95806b9.pth | [PyIQA weights](https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/musiq_koniq_ckpt-e95806b9.pth) |
| clip_rn50 | RN50.pt | [OpenAI CLIP](https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt) |
| dover | DOVER.pth，完整 DOVER，非 Mobile | [官方 release v0.1.0](https://github.com/VQAssessment/DOVER/releases/tag/v0.1.0) |
| fastvqa | FAST_VQA_B_1_4.pth | [官方 release v2.0.0](https://github.com/VQAssessment/FAST-VQA-and-FasterVQA/releases/tag/v2.0.0) |
| fastervqa | FAST_VQA_3D_1_1.pth | [官方 release v2.0.0](https://github.com/VQAssessment/FAST-VQA-and-FasterVQA/releases/tag/v2.0.0) |
| raft_things | models/raft-things.pth，full Things | [官方 RAFT 下载说明](https://github.com/princeton-vl/RAFT#demos) |

LPIPS 需要 **alexnet + lpips_alex**；DISTS 需要 **vgg16 + dists**。只有头部参数不够。NIQE 的 `.mat` 也必须手动提供。普通 CLIP-IQA 不需要 CLIP-IQA+ 的 learned-prompt 文件。FasterVQA 与 FAST-VQA 不可互换；不要换成 `_Scr`、Mobile、MS 或 MT 变体。

PyIQA 同时在 [GitHub v0.1-weights](https://github.com/chaofengc/IQA-PyTorch/releases/tag/v0.1-weights) 发布若干同名文件。备用源只能使用相同文件及匹配哈希，不能随意换训练版本。RAFT 的官方 `download_models.sh` 下载模型包，用户手动操作后只需配置其中 `raft-things.pth`；本工具不会执行该脚本。

例如，以下是**用户自行下载时**可使用的单个文件命令，本次交付未执行：

```bash
mkdir -p weights
curl --fail --location 'https://github.com/VQAssessment/DOVER/releases/download/v0.1.0/DOVER.pth' --output weights/DOVER.pth
```

各配置路径不要求保持官方文件名；解析器按配置键映射隐式请求到本地文件。不要将不可信的 pickle 权重交给 `torch.load`；这里只指定官方来源文件。

## 检查及真实权重验证

```bash
cp examples/eval.yaml eval.local.yaml
# Edit actual local paths first.
vsr-eval weights info
vsr-eval doctor --model dove --dataset udm10
python scripts/validate_local_weights.py --config eval.local.yaml --output local-validation.json
vsr-eval run --model dove --dataset udm10
```

`doctor` 检查输入、选中项的本地权重、PyTorch设备、PyIQA版本及上游源码commit，不运行模型。实际构造和推理仍由 `run`/验证脚本检查。评价时 Python socket 连接被禁止，torchvision/PyIQA/CLIP 的下载 helper 被重定向到本地清单，且架构的延迟导入先于别名绑定。不会联网查缺失文件。

验证脚本检查感知距离恒等性、输出有限性、视频模型同 seed 重复稳定性和时序路径可运行性。通过它仍不等于复现论文表格；后者需要相同数据版本、退化方式、预处理、保存编码和采样协议。
