# 第三方来源、许可证与引用

[![简体中文](https://img.shields.io/badge/Language-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-blue?style=for-the-badge)](THIRD_PARTY_NOTICES_ZH.md) [![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](THIRD_PARTY_NOTICES.md)

本工具原创代码按 Apache-2.0 许可证发布（见 [LICENSE](LICENSE)）。仓库未再分发预训练权重、评价数据集、复原模型代码或未获许可的上游评价文件。可选依赖、手动安装的源码 checkout 及其模型文件仍须遵守各自条款；本工具许可证不替代这些条款。

| 来源 | 本项目用途 | 核查到的上游条款 |
|---|---|---|
| BasicSR / XPixelGroup | 经典指标公式及数值对照 | Apache-2.0 |
| DOVE / Zheng Chen 等 | 包装脚本与协议审计 | Apache-2.0 |
| QuantVSR / Bowen Chai 等 | RAFT 调用接口审计 | Apache-2.0 |
| SparkVSR / taco-group | 统一指标包装审计 | Apache-2.0 |
| PyIQA / Chaofeng Chen 等 | 可选图像指标软件包 | 固定 PyPI 0.1.15.post2 附带 CC BY-NC-SA 4.0 的 LICENSE 和 LICENSE-S-Lab；核查时 GitHub main 已改用 PolyForm Noncommercial。应查看实际安装分发包的条款，不要用其他分支的许可证替代。 |
| DOVER / VQAssessment | 可选固定源码 checkout、官方采样和模型 | S-Lab License 1.0，非商业用途 |
| FAST-VQA / FasterVQA / VQAssessment | 可选固定源码 checkout、官方采样和模型 | S-Lab License 1.0，非商业用途 |
| RAFT / Princeton-VL | 可选固定源码 checkout、原版完整 Things 模型 | BSD-3-Clause |
| LPIPS / richzhang | 通过 PyIQA 使用原始指标 | 请核对原始 PerceptualSimilarity 仓库及随模型文件提供的条款 |
| DISTS / dingkeyan93 | 通过 PyIQA 使用原始指标 | 请核对原始 DISTS 仓库及随模型文件提供的条款 |
| OpenAI CLIP | 通过 PyIQA 使用 RN50 骨干 | 代码为 MIT；模型文件另需遵守原始模型条款 |
| torchvision | AlexNet/VGG16 运行时与权重接口 | 代码为 BSD-3-Clause；模型/数据条款另行适用 |

其他调研 VSR 仓库各自拥有许可证。GitHub 标为 `NOASSERTION` 或 `not_declared` 不代表可复制代码。Upscale-A-Video 明确采用 NTU S-Lab License；STAR 区分 I2VGen-XL 与 CogVideoX 的模型条款。本工具没有复制这些项目的评价代码。

源码 URL 与固定版本见 [SURVEY.md](docs/SURVEY.md) 和 `src/vsr_evalkit/resources/sources.json`；软件包和权重选择见 [WEIGHTS.md](docs/WEIGHTS.md)。可选源码对照脚本会将上游 Python 文件取到用户指定的缓存目录供本地执行，不会把这些文件打包进仓库。

发布评价结果时，请引用所比较的复原模型、数据集和实际使用的指标实现。相关原始工作包括：Wang 等（SSIM，2004）、Zhang 等（LPIPS，CVPR 2018）、Ding 等（DISTS，TPAMI）、Mittal 等（NIQE）、Ke 等（MUSIQ，ICCV 2021）、Wang 等（CLIP-IQA）、Wu 等（DOVER，ICCV 2023；FAST-VQA，ECCV 2022；FasterVQA，TPAMI）以及 Teed 与 Deng（RAFT，ECCV 2020）。请从所链接的官方仓库获取准确 BibTeX，避免猜测不同版本的会议和年份。
