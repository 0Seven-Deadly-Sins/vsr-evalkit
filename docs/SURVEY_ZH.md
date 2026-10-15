# 上游源码调研与评价入口清单

[![简体中文](https://img.shields.io/badge/Language-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-blue?style=for-the-badge)](SURVEY_ZH.md) [![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](SURVEY.md)

核查日期：2026-10-15。表中源码链接均固定到具体 commit。尺寸分类描述上游推理代码或 README 的证据，不代表本机实测能力。本项目评价已生成的结果，不接入模型推理；没有下载模型权重或数据集。

## 固定源码、尺寸支持与评价脚本

| 工作/库 | 固定源码 | 尺寸支持证据 | 评价入口候选 / 指标代码 | GitHub 标注许可证 |
|---|---|---|---|---|
| DOVE | [zhengchen1999/DOVE@0cd4240](https://github.com/zhengchen1999/DOVE/tree/0cd4240442cb5d122839c279977142cb6d648987) | **分块支持**：README 描述空间分块与时间分块；尺寸支持依赖这些设置。见[说明/代码](https://github.com/zhengchen1999/DOVE/blob/0cd4240442cb5d122839c279977142cb6d648987/README.md)。 | [`eval_metrics.py`](https://github.com/zhengchen1999/DOVE/blob/0cd4240442cb5d122839c279977142cb6d648987/eval_metrics.py)；`finetune/scripts/eval_{dover,ewarp,metrics,vbench}.py` | Apache-2.0 |
| SeedVR / SeedVR2 | [ByteDance-Seed/SeedVR@e4de8c2](https://github.com/ByteDance-Seed/SeedVR/tree/e4de8c24441a67e1b7df56abea10645059bb1185) | **原生自适应**：README 明确称支持任意分辨率恢复；SeedVR2 增加自适应 attention window。见[说明/代码](https://github.com/ByteDance-Seed/SeedVR/blob/e4de8c24441a67e1b7df56abea10645059bb1185/readme.md)。 | 未找到独立 VSR 评价入口；不从论文指标反推预处理。 | Apache-2.0 |
| FlashVSR | [OpenImagingLab/FlashVSR@683f755](https://github.com/OpenImagingLab/FlashVSR/tree/683f75554d68f170fc503ca1b5e2cddd604cb5e0) | **原生自适应**：README 以局部注意力说明高分辨率泛化，主要面向 4×。见[说明/代码](https://github.com/OpenImagingLab/FlashVSR/blob/683f75554d68f170fc503ca1b5e2cddd604cb5e0/README.md)。 | [`ImageQualityMetric`](https://github.com/OpenImagingLab/FlashVSR/tree/683f75554d68f170fc503ca1b5e2cddd604cb5e0/diffsynth/extensions/ImageQualityMetric)：通用图像奖励代码，不等同于已验证的 VSR benchmark。 | Apache-2.0 |
| STAR | [NJU-PCALab/STAR@69b8bc5](https://github.com/NJU-PCALab/STAR/tree/69b8bc53e4bb04267265fa9a288ee30882c5b2ed) | **未核实**：本次源码核查无法确认任意空间尺寸行为，也未集成推理。见[说明/代码](https://github.com/NJU-PCALab/STAR/blob/69b8bc53e4bb04267265fa9a288ee30882c5b2ed/README.md)。 | 未找到独立入口；不推测论文指标口径。 | not_declared |
| Upscale-A-Video | [sczhou/Upscale-A-Video@10ca1d7](https://github.com/sczhou/Upscale-A-Video/tree/10ca1d751cad25b8ac0b9b0cab415a82d9df1ac6) | **分块/分段处理**：推理采用空间/时间处理；没有证据表明评价协议与分辨率无关。见[说明/代码](https://github.com/sczhou/Upscale-A-Video/blob/10ca1d751cad25b8ac0b9b0cab415a82d9df1ac6/README.md)。 | 未发现独立评价入口；不猜论文表格预处理。 | NOASSERTION |
| MGLD-VSR | [IanYeung/MGLD-VSR@74a3425](https://github.com/IanYeung/MGLD-VSR/tree/74a34253619b863cf196de614a177972cad61bca) | **VAE 分块**：README 明确提到 arbitrary-size 测试和 VAE chopping。见[说明/代码](https://github.com/IanYeung/MGLD-VSR/blob/74a34253619b863cf196de614a177972cad61bca/README.md)。 | [`psnr_ssim.py`](https://github.com/IanYeung/MGLD-VSR/blob/74a34253619b863cf196de614a177972cad61bca/basicsr/metrics/psnr_ssim.py)、[`niqe.py`](https://github.com/IanYeung/MGLD-VSR/blob/74a34253619b863cf196de614a177972cad61bca/basicsr/metrics/niqe.py)。 | NOASSERTION |
| DLoRAL | [yjsunnn/DLoRAL@e8a5574](https://github.com/yjsunnn/DLoRAL/tree/e8a5574124dd18d7d6ea71d6974bab6705f6e1f4) | **分块候选**：推理暴露 VAE tile size 与 `process_size`；确切原生尺寸能力需实测。见[说明/代码](https://github.com/yjsunnn/DLoRAL/blob/e8a5574124dd18d7d6ea71d6974bab6705f6e1f4/README.md)。 | [`ram/utils/metrics.py`](https://github.com/yjsunnn/DLoRAL/blob/e8a5574124dd18d7d6ea71d6974bab6705f6e1f4/ram/utils/metrics.py)、`src/cross_frame_retrieval/cfr_main.py`。 | MIT |
| RealViformer | [Yuehan717/RealViformer@bd5f88d](https://github.com/Yuehan717/RealViformer/tree/bd5f88d05ba62136727a61cb162da53f22560465) | **未核实**：测试代码公开，但未运行验证任意空间尺寸。见[说明/代码](https://github.com/Yuehan717/RealViformer/blob/bd5f88d05ba62136727a61cb162da53f22560465/README.md)。 | 未确认完整独立评价入口；框架、退化和划分约定待单独核实。 | MIT |
| RealBasicVSR | [ckkelvinchan/RealBasicVSR@b908e08](https://github.com/ckkelvinchan/RealBasicVSR/tree/b908e08323c84e96b08f0d58621026ab27f56bcb) | **未核实**：测试代码公开；独立评价交由框架依赖处理。见[说明/代码](https://github.com/ckkelvinchan/RealBasicVSR/blob/b908e08323c84e96b08f0d58621026ab27f56bcb/README.md)。 | 未确认完整独立评价入口；不从推理代码推导表格协议。 | Apache-2.0 |
| SparkVSR | [taco-group/SparkVSR@a082284](https://github.com/taco-group/SparkVSR/tree/a082284b80005bb5615c0f5f5f5ed66650b1b1e7) | **分块/分段**：推理提供空间 tile 和时间 chunk；并提供统一评价脚本。见[说明/代码](https://github.com/taco-group/SparkVSR/blob/a082284b80005bb5615c0f5f5f5ed66650b1b1e7/README.md)。 | [`eval_all_metrics.py`](https://github.com/taco-group/SparkVSR/blob/a082284b80005bb5615c0f5f5f5ed66650b1b1e7/finetune/scripts/eval_all_metrics.py)：PSNR/SSIM、感知/无参考图像指标、DOVER、warping 等。 | Apache-2.0 |
| QuantVSR | [bowenchai/QuantVSR@b3c69a2](https://github.com/bowenchai/QuantVSR/tree/b3c69a2e2bb3e3f1536515c7616bdfccbd99033f) | **分块**：推理源于 MGLD；benchmark 分别包含图像、DOVER 与 RAFT 评价。见[说明/代码](https://github.com/bowenchai/QuantVSR/blob/b3c69a2e2bb3e3f1536515c7616bdfccbd99033f/README.md)。 | [`eval_metrics.py`](https://github.com/bowenchai/QuantVSR/blob/b3c69a2e2bb3e3f1536515c7616bdfccbd99033f/benchmarks/eval_metrics.py)、[`eval_dover.py`](https://github.com/bowenchai/QuantVSR/blob/b3c69a2e2bb3e3f1536515c7616bdfccbd99033f/benchmarks/eval_dover.py)、[`eval_ewarp.py`](https://github.com/bowenchai/QuantVSR/blob/b3c69a2e2bb3e3f1536515c7616bdfccbd99033f/benchmarks/eval_ewarp.py)。 | Apache-2.0 |
| STCDiT | [JyChen9811/STCDiT@7c4be6e](https://github.com/JyChen9811/STCDiT/tree/7c4be6e2774b1bdf51658d1e495a0a3a3ace3772) | **未核实**：文件名候选包含通用图像奖励，而非已验证的 VSR benchmark 入口。见[说明/代码](https://github.com/JyChen9811/STCDiT/blob/7c4be6e2774b1bdf51658d1e495a0a3a3ace3772/README.md)。 | `diffsynth/extensions/ImageQualityMetric` 通用奖励模块。 | not_declared |
| DUO-VSR | [cszy98/DUO-VSR@d0042b0](https://github.com/cszy98/DUO-VSR/tree/d0042b00105b015f3c13b3086777e1f769e8ef12) | **未核实**：文件名候选中未发现独立指标入口。 | 未找到独立评价入口；不推测论文指标。 | not_declared |
| Stream-DiffVSR | [jamichss/Stream-DiffVSR@f18b9e4](https://github.com/jamichss/Stream-DiffVSR/tree/f18b9e481436afd1ac286faf88709bc27834f37b) | **未核实**：流式推理不等于支持任意空间尺寸；未找到独立指标入口。见[说明/代码](https://github.com/jamichss/Stream-DiffVSR/blob/f18b9e481436afd1ac286faf88709bc27834f37b/README.md)。 | 未确认独立入口。 | Apache-2.0 |
| DiffVSR | [xh9998/DiffVSR@3b2b998](https://github.com/xh9998/DiffVSR/tree/3b2b99881930e0b39358430ad93bdcd273b43047) | **分块**：README 提供用于显存控制的 `tile_size`、`tile_overlap`。见[说明/代码](https://github.com/xh9998/DiffVSR/blob/3b2b99881930e0b39358430ad93bdcd273b43047/README.md)。 | 未找到独立评价入口；不推测论文表格设置。 | not_declared |
| VEnhancer | [Vchitect/VEnhancer@80ffaa3](https://github.com/Vchitect/VEnhancer/tree/80ffaa33988c583b129b730ce9d559b114de2d8c) | **调整目标尺寸**：推理可调整目标分辨率并插帧；严格帧配对需要仅包含 SR 的输出。见[说明/代码](https://github.com/Vchitect/VEnhancer/blob/80ffaa33988c583b129b730ce9d559b114de2d8c/README.md)。 | 未找到独立评价入口；插帧不能静默对齐到 SR GT。 | not_declared |
| BasicSR | [XPixelGroup/BasicSR@8d56e3a](https://github.com/XPixelGroup/BasicSR/tree/8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a) | **指标库**：经典 PSNR/SSIM 参考，使用 valid-window SSIM 和显式亮度转换。 | [`psnr_ssim.py`](https://github.com/XPixelGroup/BasicSR/blob/8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a/basicsr/metrics/psnr_ssim.py)、[`niqe.py`](https://github.com/XPixelGroup/BasicSR/blob/8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a/basicsr/metrics/niqe.py)。 | Apache-2.0 |
| PyIQA | [chaofengc/IQA-PyTorch@3186bdc](https://github.com/chaofengc/IQA-PyTorch/tree/3186bdc75cb35871690fe8ebdceb70a7e19f3301) | **指标库**：各图像指标有各自预处理和隐式文件依赖；本工具另外固定 PyPI 0.1.15.post2。 | [`pyiqa/metrics`](https://github.com/chaofengc/IQA-PyTorch/tree/3186bdc75cb35871690fe8ebdceb70a7e19f3301/pyiqa/metrics)。 | NOASSERTION |
| DOVER | [VQAssessment/DOVER@f1ddc96](https://github.com/VQAssessment/DOVER/tree/f1ddc96215bc7fbcf8f315c65d47905f339c3419) | **官方视图采样**：technical 片段与 aesthetic resize 属于 DOVER 自身定义，不是全局评价器缩放。 | [`evaluate_a_set_of_videos.py`](https://github.com/VQAssessment/DOVER/blob/f1ddc96215bc7fbcf8f315c65d47905f339c3419/evaluate_a_set_of_videos.py)、[`evaluate_one_video.py`](https://github.com/VQAssessment/DOVER/blob/f1ddc96215bc7fbcf8f315c65d47905f339c3419/evaluate_one_video.py)。 | NOASSERTION |
| FAST-VQA / FasterVQA | [VQAssessment@8db452e](https://github.com/VQAssessment/FAST-VQA-and-FasterVQA/tree/8db452e2caa5d5d4da507bcf577c19b8114f2ebd) | **官方视图采样**：二者使用不同 checkpoint/config 和校准常数。 | [`fastvqa/models/evaluator.py`](https://github.com/VQAssessment/FAST-VQA-and-FasterVQA/blob/8db452e2caa5d5d4da507bcf577c19b8114f2ebd/fastvqa/models/evaluator.py) 及官方配置/采样器。 | NOASSERTION |
| RAFT | [princeton-vl/RAFT@2888e15](https://github.com/princeton-vl/RAFT/tree/2888e15a51fa41140771d3f498ed8023cff098d1) | **光流后端**：原版 RAFT 期待 [0,255] 输入并在内部归一化；需 pad/unpad 到 8 的倍数。 | [`evaluate.py`](https://github.com/princeton-vl/RAFT/blob/2888e15a51fa41140771d3f498ed8023cff098d1/evaluate.py)。 | BSD-3-Clause |

尺寸类别包括：原生自适应、依赖分块、目标尺寸调整、官方模型自身视图采样、指标库，以及未核实。支持任意输入尺寸不等于支持任意放大倍数、无需 GT 对齐，或可在有限显存下原尺寸运行 RAFT。

## 评价协议覆盖审计

这是对公开源码和评价入口的核查。下表只填写检查到的代码事实，不根据论文表格惯例补猜缺失字段。即使原论文没有独立评价脚本，本工具也可按结果路径评价模型输出，但不声称复现论文表格。

| 工作/入口 | 源码中可确认的指标 | 已查明的预处理/聚合 | 权重来源/状态 |
|---|---|---|---|
| DOVE 根目录评价器 | PSNR、SSIM、LPIPS、DISTS、CLIP-IQA；还接受其他 PyIQA 名称 | 帧共同前缀和空间共同裁切；全局 Y 开关影响所有有参考指标；逐帧平均后视频宏平均，中间舍入 | PyIQA 及 torchvision/CLIP 隐式文件；视频包装器另依赖外部代码。见 [AUDIT](AUDIT.md)。 |
| SparkVSR 统一评价器 | PSNR、SSIM、LPIPS、DISTS、CLIP-IQA、NIQE、MUSIQ；DOVER、warping、VBench，以及标为 fastvqa 的 FasterVQA 路径 | 共同帧前缀/中心裁剪及模型特定包装；未验证论文表格数值一致性 | PyIQA 和独立安装的 DOVER/RAFT/VQA/VBench；见上方固定源码候选与审计。 |
| QuantVSR benchmark 包装 | 图像指标、DOVER、RAFT warping | RAFT 调用方先缩小一半，并把 [0,1] 输入传给内部还会归一化的 RAFT；原始误差乘 1000。其他包装设置未作为默认口径。 | 原版 RAFT Things 和所引用指标包；接口量纲问题已审计。 |
| MGLD-VSR 内嵌 BasicSR | 可见经典 PSNR/SSIM、NIQE、FID | 可查看库公式；未运行验证当前 YAML、样本总体或论文聚合方法。 | 依赖 BasicSR 和相应统计/模型文件；代码存在本身不构成已验证论文预设。 |
| SeedVR / SeedVR2、STAR、Upscale-A-Video、DLoRAL | 本次未独立确认 VSR 评价入口 | 不从模型推理参数推测论文协议。 | 可按通用结果目录评价；不声称复现原论文表格。 |
| RealViformer、RealBasicVSR | 推理代码公开；未在此确认完整独立评价器 | 框架、退化方式及划分约定待核实。 | 可评价成对/非成对的生成结果。 |
| FlashVSR、STCDiT | 源码候选中有通用 ImageQualityMetric 奖励模块 | 未将奖励模块视作已发布的 VSR benchmark 评价器。 | 通用结果评价，不表示复现其报告指标。 |
| DUO-VSR、Stream-DiffVSR、DiffVSR、VEnhancer | 未确认完整独立 benchmark 入口 | 尺寸/流式/推理设置不能确定指标色彩、边界、采样或聚合方式。 | 通用输出评价；VEnhancer 插帧不能静默与 SR GT 对齐。 |
| BasicSR / PyIQA | 经典和图像质量参考实现 | 已检查颜色空间、epsilon、舍入、SSIM-CS 和学习指标预处理差异。 | 固定软件包版本和完整工件清单；工具口径见 [PROTOCOLS](PROTOCOLS.md)、[WEIGHTS](WEIGHTS.md)。 |
| DOVER / FAST-VQA / FasterVQA / RAFT | 检查官方模型及预处理接口 | 无权重运行了真实 VQA 采样/视图函数；RAFT 调用范围与光流数学通过模拟流验证。 | 固定 commit 源码 + 用户手动提供官方模型文件；预训练数值验证待完成。 |

## 整合选择

- 审计 DOVE、SparkVSR、QuantVSR 的包装和结果配对方式，不整段复制这些仓库的包装代码。
- BasicSR 用作经典公式的独立数值参照。PyIQA 提供 5 项非经典图像指标；原始指标仓库作为引用来源。
- DOVER、FAST-VQA/FasterVQA 和 RAFT 使用手动安装、commit 固定的源码 checkout；模型推理在网络拦截保护下运行。
- 对没有已核实独立评价入口的工作，通过通用结果路径评价其生成结果；这不代表复现论文表格。
- 模型接受任意尺寸不代表接受任意倍率、与 GT 自动对齐，或能在有限显存下运行原尺寸 RAFT。

## 未列入默认集合的指标

| 候选指标 | 不作为默认项的原因 | 官方来源 |
|---|---|---|
| VBench | 面向生成视频多个维度；还需若干模型和单独规定的采样协议。 | https://github.com/Vchitect/VBench |
| FID / FVD | 分布指标要求匹配采样总体、足量视频、固定特征提取器和数据集级聚合。 | https://github.com/bioinf-jku/TTUR / https://github.com/google-research/google-research/tree/master/frechet_video_distance |
| VMAF | 需要明确显示设备/模型、缩放和像素格式选择；不能等同于复原保真度。 | https://github.com/Netflix/vmaf |

完整机器可读元数据、源码候选路径和 commit 日期见 `src/vsr_evalkit/resources/sources.json`。`NOASSERTION`/`not_declared` 不代表允许复制代码。另见 [第三方说明](../THIRD_PARTY_NOTICES.md)。
