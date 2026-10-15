# Third-party sources, licenses and citations

Original toolkit code is distributed under Apache-2.0 (LICENSE). No pretrained weights, evaluation datasets, restoration model code, or unlicensed upstream evaluator files are redistributed. Optional dependencies, manually installed source checkouts and their model artifacts retain their own terms; the toolkit's license does not replace those terms.

| Source | Use here | Upstream terms observed |
|---|---|---|
| BasicSR / XPixelGroup | Classical formula and differential reference | Apache-2.0 |
| DOVE / Zheng Chen et al. | Wrapper/protocol audit | Apache-2.0 |
| QuantVSR / Bowen Chai et al. | RAFT caller contract audit | Apache-2.0 |
| SparkVSR / taco-group | Unified wrapper audit | Apache-2.0 |
| PyIQA / Chaofeng Chen et al. | Optional installed image metric package | Pinned PyPI 0.1.15.post2 ships LICENSE (CC BY-NC-SA 4.0) and LICENSE-S-Lab; inspected GitHub main now uses PolyForm Noncommercial. Consult the exact installed distribution, not a different branch. |
| DOVER / VQAssessment | Optional fixed source checkout, official sampling/model | S-Lab License 1.0, noncommercial |
| FAST-VQA / FasterVQA / VQAssessment | Optional fixed source checkout, official sampling/model | S-Lab License 1.0, noncommercial |
| RAFT / Princeton-VL | Optional fixed source checkout, original full Things model | BSD-3-Clause |
| LPIPS / richzhang | Original metric, used through PyIQA | Consult original PerceptualSimilarity repository and supplied model terms |
| DISTS / dingkeyan93 | Original metric, used through PyIQA | Consult original DISTS repository and supplied model terms |
| OpenAI CLIP | RN50 backbone, used through PyIQA | MIT code; consult original model terms |
| torchvision | AlexNet/VGG16 runtime and weight interfaces | BSD-3-Clause code; original model/data terms remain separate |

Other surveyed VSR repositories have individual licenses; entries reported `NOASSERTION` or `not_declared` are not treated as permission to copy. In particular, Upscale-A-Video identifies the NTU S-Lab License, and STAR distinguishes I2VGen-XL and CogVideoX model terms. No evaluator code was copied from those projects.

Source URLs and pinned revisions: [SURVEY.md](docs/SURVEY.md) and `src/vsr_evalkit/resources/sources.json`. Package- and checkpoint-specific choices: [WEIGHTS.md](docs/WEIGHTS.md). The optional reference verification script retrieves upstream Python files into a user-selected cache for local execution; it does not package those files into this repository.

When publishing evaluation results, cite the restoration models under comparison, datasets and the metric implementations actually used. Appropriate original works include Wang et al. (SSIM, 2004), Zhang et al. (LPIPS, CVPR 2018), Ding et al. (DISTS, TPAMI), Mittal et al. (NIQE), Ke et al. (MUSIQ, ICCV 2021), Wang et al. (CLIP-IQA), Wu et al. (DOVER, ICCV 2023; FAST-VQA, ECCV 2022; FasterVQA, TPAMI), and Teed & Deng (RAFT, ECCV 2020). Obtain exact BibTeX from the linked official repositories rather than guessing venue/year variants.
