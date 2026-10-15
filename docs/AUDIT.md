# 源码审计与差异记录

审计日期：2026-10-15。源码、固定 commit 与许可信息见 [SURVEY](SURVEY.md)。以下结论来自代码路径和无权重合成测试；没有推断这些问题对论文表格或排名的实际影响。

## DOVE：全参考指标共用 Y 开关

[固定评价脚本](https://github.com/zhengchen1999/DOVE/blob/0cd4240442cb5d122839c279977142cb6d648987/eval_metrics.py) 将 PSNR、SSIM、LPIPS、DISTS 都放在 `fr_metrics`，并在共同分支应用 `test_y_channel`。开启该参数时，四项均收到单通道输入。即便感知模型通过广播继续运行，这也不再是其标准 RGB 协议。

本项目：PSNR/SSIM 单独选择颜色空间；LPIPS/DISTS 始终收到 RGB。测试 `test_perceptual_input_stays_rgb_and_zero_to_one` 覆盖输入约定。默认未开启 Y 开关时，不应笼统声称 DOVE 所有指标都受此问题影响。

## DOVE：尺寸、帧数与失败样本

同一脚本取两序列最小帧数及最小 H/W，默认从左上角裁齐；缺 GT/异常样本会被跳过，视频分数先保留四位再平均。它们分别属于协议选择、覆盖透明度和精度损失，不能全部叫公式错误。

本项目：严格对齐为默认；只有显式预设允许 common crop/truncation，并记录变换。缺失样本和失败不静默消失，保留原精度至最终展示。不同长度视频使用视频等权平均，不误用逐帧全局平均。

## DOVE DOVER 包装：结果错位及未公开接口依赖

[固定包装脚本](https://github.com/zhengchen1999/DOVE/blob/0cd4240442cb5d122839c279977142cb6d648987/finetune/scripts/eval_dover.py) 使用 `dover_results[i-1]` 回填从 0 枚举的预测名称。若结果是通常顺序的列表，则三个分数 [0.1,0.2,0.3] 会回填为 [0.3,0.1,0.2]。独立于此，它导入的 `evaluate_set` 在本次固定官方 DOVER 的 `evaluate_a_set_of_videos.py` 中没有定义，因此依赖未公开的改版接口。不能假定任何官方 checkout 都直接兼容。

本项目：直接调用固定官方模型与采样函数，并按 clip ID 存结果；分支按名称融合。原包装还把帧序列转换为固定 8 FPS 视频，本项目不进行这种转换。

## DOVER：分支顺序依赖输入字典

官方模型 `DOVER.forward` 的分数列表跟随输入字典遍历顺序。单视频与批量入口构造字典的顺序可以不同，因此不能把一个入口的数组下标公式机械移植到另一个入口。这里不是断言两个官方入口都计算错误。

本项目：先用输入视图名称绑定输出，再使用 technical/aesthetic 校准常数。测试交换字典顺序后融合值仍相同，并记录两分支原始值。

## QuantVSR：RAFT 输入重复归一化

[固定 warping 代码](https://github.com/bowenchai/QuantVSR/blob/b3c69a2e2bb3e3f1536515c7616bdfccbd99033f/benchmarks/RAFT/ewarp.py) 先将图像除以 255；[其 RAFT.forward](https://github.com/bowenchai/QuantVSR/blob/b3c69a2e2bb3e3f1536515c7616bdfccbd99033f/benchmarks/RAFT/core/raft.py) 又执行除以 255 的内部归一化。调用方 [0,1] 与被调用方期待 [0,255] 不匹配，这是明确的接口量纲问题；没有真实权重就不能量化光流误差幅度。

同一脚本在计算误差前把图像缩小一半；掩码分母按像素而分子累加 RGB 通道，也与本项目的 RGB 均方定义不同。这些是需要独立标识的其他差异，不能只修输入后将数值叫成“论文完全复现”。

本项目用模拟模型断言输入仍为 [0,255]，并验证奇数尺寸 pad/unpad；光流估计降尺寸只发生在显式预设中，恢复向量后在原图评分。

## SparkVSR：FAST/Faster 命名与 warping 变体

[官方评价入口](https://github.com/taco-group/SparkVSR/blob/main/finetune/scripts/eval_all_metrics.py) 的 `fastvqa` 路径选取 `f3dvqa-b.yml` 和 3D checkpoint，即 FasterVQA 变体。该实现的 warping 使用未加遮挡掩码的 MSE，不能与遮挡归一化版本仅凭同名合并。固定 commit 链接见 SURVEY。

本项目将 FAST-VQA 与 FasterVQA 注册为两项，固定各自配置、文件和归一化常数。warping 文档明确掩码、分母及单位；不按论文表格数量级猜测指标定义。

## PyIQA 与 BasicSR：合理的同名差异

固定 PyIQA 包为 **0.1.15.post2**；其 GitHub main 源码快照仅供比较，不代替包版本。

- PyIQA 默认 PSNR 使用 RGB，默认 SSIM 使用 YIQ 亮度通道；默认列表中的同一开关不代表有效口径相同。
- PyIQA PSNR 在 MSE 上加 1e-8；恒等输入返回 80 dB。本项目标准无 epsilon，返回 +∞。两者不能用恒等样本要求完全相等。
- PyIQA SSIM 对 CS 响应作 ReLU，BasicSR 保留负值。反相棋盘样本中，本项目标准/BasicSR约 −0.996406，而截断版本为 0；差异不是浮点噪声。
- YIQ full-range 与 BT.601 studio-range 是不同公式。BasicSR 的 Y 工具主要用于其 OpenCV BGR 流程，直接喂 RGB 而不核对约定会交换红蓝影响。
- PyIQA 的 AlexNet/VGG16 构造会请求 torchvision 骨干；仅配置 LPIPS/DISTS 头文件不能保证离线。

本项目提供显式预设、独立骨干清单、下载拦截和实际源码对照脚本。参见 [VALIDATION](VALIDATION.md) 的误差界与未完成验证。

## 核查状态

| 层次 | 状态 |
|---|---|
| 来源、输入范围、默认参数、配对、聚合与接口 | 已静态核查，关键分支有无权重测试 |
| PSNR/SSIM 与真实 BasicSR/PyIQA 函数的数值对照 | 已执行，按明确同口径和容差比较 |
| 感知模型真实预训练权重数值、DOVER/FAST/Faster 官方端到端 parity | 待用户本地权重 |
| RAFT 真实光流质量、CUDA 峰值显存、所有 VSR 论文表格复现 | 未执行 |

不把缺少公开代码当作该工作没有指标，不把推断当作实测，也不把模拟后端测试当作真实模型验证。
