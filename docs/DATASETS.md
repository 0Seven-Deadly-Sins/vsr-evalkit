# Dataset configuration notes

[![简体中文](https://img.shields.io/badge/Language-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-blue?style=for-the-badge)](DATASETS_ZH.md) [![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](DATASETS.md)

The eight IDs in `examples/eval.yaml` are configuration examples, not embedded dataset loaders or a claim that all distributions with those names are identical. No datasets are downloaded. Any additional dataset can be registered by paths and an optional mapping CSV.

| Example ID | GT configuration | Source/protocol caveat |
|---|---|---|
| udm10 | Explicit GT root | DOVE/STAR and other projects distribute different synthetic degradation variants. Match the LQ→prediction origin to its GT. |
| spmcs | Explicit GT root | Verify the chosen test subset, frame range and degradation; do not assume the complete original collection equals a paper's subset. |
| youhq40 | Explicit GT root | The Upscale-A-Video test subset is reused by DOVE and other works; preserve exact clip/frame identity. |
| reds4 | Explicit GT root | REDS4 is a particular subset, not all REDS validation videos or REDS30. |
| vid4 | Explicit GT root | Classical reconstruction papers often use Y channels and border cropping; select a documented protocol explicitly. |
| realvsr | GT only if paired/aligned references are available | Real captures can require registration. The toolkit does not silently register scenes. Remove gt_root for unpaired use. |
| mvsr4x | GT only if correct paired references are available | Verify dataset version, reference alignment and scale. Remove gt_root for unpaired use. |
| videolq | No GT by default | Auto chooses NR/video/temporal metrics; FR values are not fabricated. |

Useful primary release documentation: [DOVE dataset table](https://github.com/zhengchen1999/DOVE#-datasets), [Upscale-A-Video](https://github.com/sczhou/Upscale-A-Video), [QuantVSR datasets](https://github.com/bowenchai/QuantVSR#-datasets), [official REDS page](https://seungjunnah.github.io/Datasets/reds.html), [RealBasicVSR](https://github.com/ckkelvinchan/RealBasicVSR). Fixed code snapshots are listed in SURVEY.md; dataset hosting URLs and terms remain controlled by their publishers.

For each experiment retain: dataset release/source, selected clip IDs, GT/LQ correspondence, degradation recipe if synthetic, frame range/stride, FPS if known, exact model output settings and codec/pixel format. A clip mapping CSV explicitly defines the evaluation population; compare models using the same mapping and complete coverage. The evaluator will not infer a dataset's hidden paper protocol from its name.
