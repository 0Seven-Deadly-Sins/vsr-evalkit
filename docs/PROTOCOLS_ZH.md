# 评价协议

[![简体中文](https://img.shields.io/badge/Language-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-blue?style=for-the-badge)](PROTOCOLS_ZH.md) [![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](PROTOCOLS.md)

## 经典图像指标

输入解码为 8-bit RGB 并归一化到 [0,1]。默认计算保留 float64。PSNR = −10 log10(mean((P−G)²))；零误差时返回 +∞。默认不加 epsilon；预设可以显式为归一化 MSE 添加 `psnr_eps`。

SSIM 使用可分离的 11×11 高斯窗口（σ=1.5）、常数 C1=0.01²、C2=0.03²、总体矩和 valid 窗口区域。先计算每个通道的 SSIM 图，再对所有有效像素及通道求平均。标准口径保留负的对比度-结构响应。`ssim_clip_negative_cs: true` 会按 PyIQA 的 CS 项 ReLU 改变定义。`ssim_round_values: true` 会在计算 SSIM 前将输入通道取整到最近的 8-bit 值。

`bt601-y` 是 MATLAB/BasicSR 使用的 studio-range 亮度转换：

```text
Y = (16 + 65.481 R + 128.553 G + 24.966 B) / 255
```

这里的 R/G/B 已在 [0,1]。`yiq-y` 是 full-range Y = 0.299R + 0.587G + 0.114B。二者公式不同。`crop_border` 只作用于 PSNR/SSIM，裁边后至少保留 11×11 像素。感知指标和无参考指标不会使用经典指标的 Y 转换或边界裁剪。显式启用共同尺寸对齐时，它作用于所有有参考图像对；无参考/视频/时序指标仍使用原始预测序列。

## 预设

| 预设 | 经典指标通道 | 对齐方式 | 其他差异 |
|---|---|---|---|
| standard-rgb-v1 | RGB / RGB | 严格 | 不加 epsilon，不截断 CS |
| standard-y-v1 | BT.601 Y / BT.601 Y | 严格 | 显式 studio-range 转换 |
| flow-960-v1 | RGB / RGB | 严格 | 光流估计长边限制为 960；误差仍在原尺寸上计算 |
| dove-code-alignment-v1 | RGB PSNR / 取整后的 YIQ SSIM | 左上角共同尺寸、共同帧前缀、序号配对 | PSNR eps=1e-8，SSIM 截断 CS；不声称复现论文结果 |

配置示例：`protocol: {preset: standard-rgb-v1, crop_border: 4, seed: 7}`。`ssim_color_space` 与 PSNR 的 `color_space` 相互独立。任一生效选项改变后都会生成不同协议 ID。由于 DOVE 没有固定其 PyIQA 版本，DOVE 预设只依据已公开代码并选用本工具固定的 PyIQA 版本。浮点实现差异见 [VALIDATION.md](VALIDATION.md)。

## 图像后端定义

- LPIPS：AlexNet IMAGENET1K_V1，校准版本 0.1；RGB [0,1]，由 PyIQA 内部归一化。
- DISTS：VGG16 IMAGENET1K_V1 及学习得到的 alpha/beta 参数，采用 PyIQA 原生预处理。
- NIQE：原始 `niqe_modelparameters.mat` 参数；PyIQA YIQ 亮度选项，不裁边；最小尺寸 192×192。退化统计仍可能产生未定义输出，此时记录为失败。
- MUSIQ：KonIQ-10k 版本，不是 AVA/SPAQ。CLIP-IQA：普通 RN50，不是 CLIP-IQA+。
- 所有模型均使用 eval/inference 模式。指标自身固定的内部预处理属于指标定义的一部分，不会被全局缩略图或分块平均替代。

## 视频质量模型

配置来自固定 commit 的上游源码。DOVER 使用 `dover.yml`、`val-l1080p`、其 `UnifiedFrameSampler` 和 `get_single_view`。FAST-VQA 使用 `options/fast/fast-b.yml`；FasterVQA 使用 `options/fast/f3dvqa-b.yml`。两者都使用 `val-kv1k`、上游采样器和空间片段。短输入依照上游的取模/重复规则；实际采样帧索引会保存，因此重复帧可见。

归一化为 `(RGB8 − [123.675,116.28,103.53]) / [58.395,57.12,57.375]`。不重新编码视频，也不伪造 FPS。每个视图只解码并保留采样帧。原尺寸 CPU 采样占用受上游采样帧数量限制，但 4K 输入仍可能占用较多内存；GPU 视图使用上游指定尺寸。

随机数种子由 clip ID 与 `protocol.seed` 的稳定哈希生成，与所选模型及遍历顺序无关。种子和采样索引会记录。相同种子不保证 CUDA 按位完全一致。

DOVER 列表输出会先绑定到输入视图字典的名称/顺序。设原始 technical 分数为 T，aesthetic 分数为 A：

```text
t = (T - 0.1107) / 0.07355
a = (A + 0.08285) / 0.03774
overall = sigmoid(0.6104*t + 0.3896*a)
```

technical/aesthetic 分支概率和原始分数均会保留。FAST-VQA 校准为 `sigmoid((raw + 0.110198185)/0.04178565)`；FasterVQA 校准为 `sigmoid((raw − 0.14759505)/0.03613452)`。输出分数保持在 [0,1]，不乘以 100。

## RAFT warping 定义

本项目计算的是**使用 RAFT Things 的遮挡掩码 RGB MSE**，并不声称所有论文中的 `E*warp` 都采用相同定义。

1. 使用完整 RAFT Things、20 次迭代和 float32 分别估计 F: I(t)→I(t+1) 与 B: I(t+1)→I(t)，关闭混合精度/alternate correlation。原版 RAFT 接收 RGB [0,255]；其内部仅归一化一次。
2. 使用复制边界对称填充到 8 的倍数；评分前从两个光流场移除填充。光流输入各维至少为 128 像素。
3. 使用双线性 `grid_sample`、`align_corners=True`，在坐标 x+F(x) 采样 I(t+1)。越出原图的坐标无效。
4. 在 x+F(x) 处采样 B。排除前后向不一致像素：`||F + warp(B,F)||² > 0.01*(||F||² + ||warp(B,F)||²) + 0.5`。
5. 对剩余 M 个像素，帧对误差为 `sum_x,c M(x)*(I_t(x,c)-warped(x,c))² / (3*sum_x M(x))`，图像值在 [0,1]。不增加运动梯度排除。求和使用 float64，并记录有效像素数。
6. 对所有可计算的相邻帧对等权平均。单帧视频或零有效像素的帧对记为未定义，不产生零分。帧对覆盖不完整会明确显示，并阻止数据集汇总。

可选 960 预设只将光流输入双线性缩放到长边 960（`align_corners=False`），再将光流恢复到原始 H/W；水平位移乘以原图宽/光流宽，垂直位移乘以原图高/光流高。评分仍在全分辨率图像上完成。记录所有尺寸和填充信息。原尺寸运行若 OOM 则报告失败，不自动降分辨率。

同时返回原始值和 `scaled_1000`（原始值×1000）。×1000 只是展示单位，不是另一项指标。上游包装若在评分前缩小图像、使用其他掩码、将 RGB 通道求和但不除以三，或错误归一化 RAFT 输入，定义均与此处不同。

## 配对、聚合与比较

clip ID 必须完全匹配，除非显式映射 CSV 指定评价样本。GT 与预测可使用不同扩展名，但同一根目录内 clip ID 重复时会报错。帧目录按自然顺序排列；identity 模式检查每帧文件名末尾的数字字段，若无数字则比较完整文件名。混合视频与帧序列按解码顺序配对。恒定帧率的经过时间 PTS 与可用 FPS 必须一致；记录容器起始偏移，但不将其解释为场景偏移。

默认对齐会拒绝帧数或尺寸不等。CFR 验证将经过时间 PTS 与标称（或推断）等间隔时间网格比较，容差为一个容器 time-base tick，避免将按毫秒取整的 16/29.97 FPS MKV 时间戳误判成 VFR。跨视频时间线比较采用两个容器时间量化单位之和，不使用随时长增长的相对容差。真实不规则/非单调时间线仍会失败。显式共同尺寸预设会记录每次裁剪/截断。不会通过时域插值或图像缩放对齐 GT。灰度/调色板图像转为 RGB；拒绝 alpha 与高位深图像。标准视频解码使用 FFmpeg rgb24，并记录源格式/范围/色彩空间。已知 HDR 传递函数标签和超过 8-bit 的格式会被拒绝；无标签 HDR 无法仅凭像素可靠识别，须由使用者确认。

逐帧均值形成逐视频分数，数据集分数为视频等权平均，不提前舍入。生成数据集汇总要求每个必需 clip 都有完整指标结果。存在缺失、错误或部分帧对时不生成总分。明确不可计算的指标以 null 和原因表示。JSON 中无穷值写成 `"+inf"`。比较组包括协议、指标权重哈希、源码 commit、依赖版本、clip ID、参考路径、尺寸、FPS、帧数和预期覆盖。参考路径会保守地区分不同映射；比较组用于本地运行防误比，不是跨机器的数据内容哈希。仅比较数据集、指标和比较组均相同的结果；不会生成混用协议的排名。
