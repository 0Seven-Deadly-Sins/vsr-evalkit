# 数据集配置说明

[![简体中文](https://img.shields.io/badge/Language-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-blue?style=for-the-badge)](DATASETS_ZH.md) [![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](DATASETS.md)

`examples/eval.yaml` 中的 8 个 ID 是路径配置示例，不包含数据集加载器，也不表示所有同名数据发布版都相同。本工具不会下载数据。其他数据集可通过目录路径和可选映射 CSV 注册。

| 示例 ID | GT 配置 | 数据来源/协议注意事项 |
|---|---|---|
| udm10 | 显式设置 GT 根目录 | DOVE、STAR 等项目发布的合成退化版本可能不同。需确保 LQ→预测结果来源与对应 GT 一致。 |
| spmcs | 显式设置 GT 根目录 | 核实测试子集、帧范围和退化方式；完整数据集不一定等于论文采用的子集。 |
| youhq40 | 显式设置 GT 根目录 | Upscale-A-Video 使用的测试子集也被 DOVE 等工作采用；保留准确的视频和帧对应关系。 |
| reds4 | 显式设置 GT 根目录 | REDS4 是特定子集，不等同于全部 REDS 验证视频或 REDS30。 |
| vid4 | 显式设置 GT 根目录 | 传统重建论文常使用 Y 通道和边界裁剪；应显式选择并记录协议。 |
| realvsr | 仅在有配对且对齐的参考帧时设置 GT | 真实拍摄帧可能需要配准。本工具不会静默配准场景；非配对使用时移除 `gt_root`。 |
| mvsr4x | 仅在存在正确配对的参考帧时设置 GT | 核实数据集版本、参考帧对齐和放大倍率；非配对使用时移除 `gt_root`。 |
| videolq | 默认无 GT | `auto` 选择无参考图像、视频质量和时序指标；不会伪造有参考分数。 |

可参考发布方的一手说明：[DOVE 数据集表](https://github.com/zhengchen1999/DOVE#-datasets)、[Upscale-A-Video](https://github.com/sczhou/Upscale-A-Video)、[QuantVSR 数据集](https://github.com/bowenchai/QuantVSR#-datasets)、[REDS 官方页面](https://seungjunnah.github.io/Datasets/reds.html)、[RealBasicVSR](https://github.com/ckkelvinchan/RealBasicVSR)。固定源码快照见 [SURVEY.md](SURVEY.md)；数据托管地址及使用条款由发布方管理。

每次实验都应记录数据集发布版和来源、所选 clip ID、GT/LQ 对应关系、合成退化流程、帧范围/步长、已知 FPS、模型输出设置和编码格式。映射 CSV 明确界定评价样本；比较模型时应使用同一映射并确保覆盖完整。评价器不会根据数据集名称猜测未公开的论文协议。
