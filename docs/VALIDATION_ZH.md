# 验证记录

[![简体中文](https://img.shields.io/badge/Language-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-blue?style=for-the-badge)](VALIDATION_ZH.md) [![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](VALIDATION.md)

日期：2026-10-15。没有下载任何评价权重、骨干权重、NIQE 统计参数或数据集。安装了 PyTorch/torchvision 以及仅含代码的 PyIQA 作为软件依赖。合成图像/视频样本在临时目录生成，不作为模型输出分发。

清单中 10 个直接权重/统计参数链接均仅通过 HEAD 元数据请求返回 HTTP 200。RAFT 链接指向官方说明。没有读取响应正文或保存模型文件；链接可访问不等于已验证权重数值。

## 本地已执行

Windows CPU，Python 3.10.10，numpy 1.26.4，scipy 1.14.1，Pillow 11.1.0，PyYAML 6.0.2，PyAV 14.2.0，OpenCV-headless 4.11.0.86，torch 2.5.1+cpu，torchvision 0.20.1+cpu。环境没有 CUDA 设备。

- `python -m pytest -q`：交付验证时 **49 项通过**；后续 CI 与仓库历史记录了额外修复和检查（如有）。
- 无权重 CLI 合成评价：成功生成 JSON/CSV/Markdown 输出，覆盖横屏、竖屏配对视频和长度不同的视频。
- 经典公式及 OpenCV 有效窗口对照，1080p/4K 恒等输入、奇数尺寸、色条、裁边边界和非法输入。
- 25、16 和 30000/1001 FPS 下，FFV1/MKV 无损视频与 PNG 帧序列一致；容器时间戳量化、实际不规则 PTS 拒绝、空/损坏/高位深输入、序列中尺寸变化、自然排序、显式映射、帧编号、FPS 和对齐检查。
- 视频宏平均、禁止提前舍入、JSON 无穷值、覆盖不完整时抑制汇总、协议 ID 区分及拒绝覆盖已有输出目录。
- 已知平移下的光流方向、有效区域、遮挡、RGB 分母、零有效像素处理、光流向量尺寸恢复，以及 RAFT 原始 `[0,255]` 输入和奇数尺寸 pad/unpad。
- 网络 socket 禁止、本地文件映射、已导入的 torchvision 下载别名、隐式骨干缺失、感知指标 RGB 输入、DOVER 按名称融合及 FAST/Faster 校准约定。

## 与上游经典指标函数的实际数值对照

`scripts/verify_references.py` 执行了 BasicSR commit `8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a` 中的真实 Python 函数，以及已安装 PyIQA **0.1.15.post2** 的经典指标函数，全程没有导入学习型模型。比较覆盖横屏、竖屏、奇数尺寸下 6 组 RGB/Y 输入，并另外检查显式变体：

| 对照项 | 最大绝对误差 |
|---|---:|
| BasicSR PSNR，同 RGB/Y 约定 | 1.2313109110e-7 dB |
| BasicSR SSIM，同 RGB/Y 约定 | 3.6035385786e-10 |
| PyIQA PSNR，显式关闭 epsilon | 0 |
| PyIQA SSIM，显式匹配 CS 截断变体 | 4.2489201046e-10 |

BasicSR 的亮度转换保留 float32 中间值；本工具使用 float64。PyIQA 的高斯核初始为 float32。验证脚本显式设置容差。PyIQA 默认 PSNR epsilon 与 SSIM 的 CS 截断属于语义差异，不计作数值误差。

反相纹理样本：标准/不截断 SSIM 为 −0.9964064299，截断 CS 变体为 0。恒等输入 PSNR：标准为 `+∞`，添加 epsilon=1e-8 后为 80 dB。这些样本说明仅写指标名称不足以形成可复现协议。

复现方法：先安装仅含代码的 PyIQA、torch 和 torchvision，然后运行：

```bash
python scripts/verify_references.py --cache reference-cache --fetch --output reference-results.json
```

`--fetch` 只获取固定版本的 **Python 源文件**，不会获取模型参数。去掉该参数后可复用本地源码缓存。

## 上游 VQA 预处理接口

`scripts/verify_vqa_contracts.py` 使用合成 5 帧序列执行了固定官方采样器和视图函数；本工具适配器连接模拟分数头：

| 模型 | 生成的模型输入尺寸 (N,C,T,H,W) |
|---|---|
| DOVER technical | (3,3,32,224,224) |
| DOVER aesthetic | (1,3,32,224,224) |
| FAST-VQA | (4,3,32,224,224) |
| FasterVQA | (1,3,32,224,224) |

相同随机种子得到相同采样索引；校准后的模拟分数符合预期为 0.5。这验证了**真实上游预处理**，但模型分数头是模拟实现，因此没有验证预训练模型数值一致性或感知质量。

```bash
python scripts/verify_vqa_contracts.py --cache reference-cache/vqa --fetch --output vqa-contracts.json
```

## 干净安装与 Linux CI

`no-weight-tests` GitHub Actions 工作流在 Ubuntu CPU 上测试 Python 3.10 和 3.11，不安装任何预训练文件，并执行测试套件及生成的 CLI 示例。[已验证运行 38714936098](https://github.com/0Seven-Deadly-Sins/vsr-evalkit/actions/runs/38714936098) 在实现提交 `bcf2918f25807732dd91ca70c13192d522eea551` 上两个任务均通过：每个版本 49 项测试通过，CLI 列表命令成功，并完成合成评价。此后仅文档变更不影响已测实现。

另在全新本地 Git checkout 中构建 wheel，并安装到一个不含 **PyTorch** 的独立 Python 3.10 环境。`pip check`、资源加载（11 个权重/工件定义、21 个源码固定版本）、`doctor` 和完整的合成 PSNR/SSIM CLI 评价均通过。这确认基础包及其资源不依赖开发目录或可选深度学习包。

## 待验证（当前不声称已通过）

- 使用所选预训练文件计算 LPIPS/DISTS/NIQE/MUSIQ/CLIP-IQA 的真实数值。
- DOVER/FAST-VQA/FasterVQA 完整加载，以及同输入/同种子下与官方脚本的数值对照。
- 真实 RAFT Things 光流准确性和真实权重下的时序分数。
- Linux CUDA 执行、峰值显存、运行速度或 4K 学习型指标吞吐测试。
- 任何 VSR 模型推理、数据集下载、人工质量研究或论文表格数值复现。

准备好本地文件后运行 `scripts/validate_local_weights.py`，再评价实际数据。该脚本只检查恒等性、有限性和重复运行稳定性。复现论文还需匹配数据集版本、退化过程、帧间隔、颜色/解码及保存设置；单次冒烟检查不能证明完整复现。
