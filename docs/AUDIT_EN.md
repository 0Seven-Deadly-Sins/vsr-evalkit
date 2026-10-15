# Source audit and metric differences

[![简体中文](https://img.shields.io/badge/Language-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-blue?style=for-the-badge)](AUDIT.md) [![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](AUDIT_EN.md)

Audit date: 2026-10-15. Source revisions and license observations are listed in the [survey](SURVEY.md). Findings below come from code paths and no-weight synthetic checks; they do not estimate effects on paper tables or rankings.

## DOVE: one Y-channel switch is shared by all full-reference metrics

The [pinned evaluator](https://github.com/zhengchen1999/DOVE/blob/0cd4240442cb5d122839c279977142cb6d648987/eval_metrics.py) places PSNR, SSIM, LPIPS and DISTS in `fr_metrics`, then applies `test_y_channel` in a shared branch. When enabled, all four metrics receive a single channel. Even if a perceptual model continues to run through broadcasting, that is no longer its standard RGB protocol.

This toolkit selects the color space for PSNR/SSIM independently; LPIPS/DISTS always receive RGB. `test_perceptual_input_stays_rgb_and_zero_to_one` covers that contract. With the Y switch disabled, it would be inaccurate to claim that every DOVE metric is affected.

## DOVE: size, frame count and failed samples

The same script uses the shorter frame count and smaller height/width, cropping from the top-left by default. Missing GT and exceptional samples can be skipped, and per-video scores are rounded to four decimal places before averaging. These are protocol choices, coverage transparency and precision loss, respectively; they are not all formula errors.

This toolkit defaults to strict alignment. Only an explicit preset permits common cropping/truncation, and it records the transformation. Missing samples and failures remain visible, and full precision is retained until display. Videos receive equal weight, including when their frame counts differ; the implementation does not substitute a global frame average.

## DOVE's DOVER wrapper: misassociated results and an unpublished API dependency

The [pinned wrapper](https://github.com/zhengchen1999/DOVE/blob/0cd4240442cb5d122839c279977142cb6d648987/finetune/scripts/eval_dover.py) assigns `dover_results[i-1]` while enumerating prediction names from zero. If results are a conventional ordered list, scores `[0.1, 0.2, 0.3]` are assigned as `[0.3, 0.1, 0.2]`. Separately, it imports `evaluate_set`, which is not defined in the pinned official `evaluate_a_set_of_videos.py`; it therefore depends on an unpublished modified interface. An arbitrary official checkout cannot be assumed to work with this wrapper.

This toolkit calls the fixed official model and sampling functions directly, associates results by clip ID and fuses branches by name. The original wrapper also converts frame sequences into fixed-8-FPS video; this toolkit does not perform that conversion.

## DOVER: branch order depends on the input dictionary

The official `DOVER.forward` score list follows the iteration order of the input dictionary. Single-video and batch entry points may construct dictionaries in different orders, so index-based formulas from one entry point cannot be transplanted mechanically into another. This finding does not claim that both official entry points are incorrect.

This toolkit binds outputs to input view names before fusion, then applies the technical/aesthetic calibration constants. A test that swaps dictionary order confirms the fused result remains the same, and raw branch scores are retained.

## QuantVSR: RAFT input is normalized twice

The [pinned warping code](https://github.com/bowenchai/QuantVSR/blob/b3c69a2e2bb3e3f1536515c7616bdfccbd99033f/benchmarks/RAFT/ewarp.py) divides images by 255; its [RAFT.forward](https://github.com/bowenchai/QuantVSR/blob/b3c69a2e2bb3e3f1536515c7616bdfccbd99033f/benchmarks/RAFT/core/raft.py) then performs its own division by 255. The caller supplies `[0,1]` while the callee expects `[0,255]`, a clear interface/range mismatch. Without real weights, the effect on flow error magnitude cannot be quantified.

The same script also halves image dimensions before scoring. Its mask denominator counts pixels while the numerator sums RGB channels, which differs from this toolkit's RGB mean-squared-error definition. These are separate deviations; fixing the input range alone would not make the result a complete reproduction of the paper protocol.

With a mock model, this toolkit asserts that RAFT still receives `[0,255]` inputs and checks odd-size padding/unpadding. Flow estimation is reduced only by an explicit preset; the restored vectors are scored against the original-resolution images.

## SparkVSR: FAST/Faster naming and a warping variant

The [official evaluation entry point](https://github.com/taco-group/SparkVSR/blob/main/finetune/scripts/eval_all_metrics.py) routes `fastvqa` to `f3dvqa-b.yml` and a 3D checkpoint, which is a FasterVQA variant. Its warping implementation uses unmasked MSE. It must not be merged with an occlusion-masked variant solely because the metric name is similar. The pinned revision is linked in the survey.

This toolkit registers FAST-VQA and FasterVQA separately and pins each configuration, checkpoint and calibration constant. Its warping definition states the mask, denominator and units; the metric definition is not guessed from the magnitude of a paper table.

## PyIQA and BasicSR: expected differences under shared metric names

The pinned PyIQA package is **0.1.15.post2**. Its GitHub `main` snapshot is only a comparison point and does not replace the package version.

- PyIQA's default PSNR uses RGB while its default SSIM uses YIQ luminance; a shared option list does not imply identical color handling.
- PyIQA adds `1e-8` to PSNR's MSE, so identical inputs return 80 dB. This toolkit's standard uses no epsilon and returns `+∞`. The two are not expected to match on an identity fixture.
- PyIQA applies ReLU to the SSIM contrast-structure response; BasicSR keeps negative values. On an inverted checkerboard, this toolkit's standard/BasicSR value is about −0.996406, whereas the clipped variant is 0. This is a semantic difference, not floating-point noise.
- Full-range YIQ and studio-range BT.601 are different formulas. BasicSR's Y utility is primarily used in its OpenCV BGR pipeline; passing RGB without checking the convention swaps red/blue contributions.
- PyIQA's AlexNet/VGG16 builders request torchvision backbones. Configuring only LPIPS/DISTS head files is insufficient for offline operation.

This toolkit exposes those variants explicitly, lists backbone artifacts separately, blocks downloads and provides a source-level differential checker. See [validation](VALIDATION.md) for error bounds and remaining checks.

## Audit status

| Scope | Status |
|---|---|
| Sources, input ranges, default options, pairing, aggregation and interfaces | Statically inspected; key branches have no-weight tests |
| Numerical comparison with real BasicSR/PyIQA PSNR/SSIM functions | Executed with explicit matching conventions and tolerances |
| Pretrained numerical values for perceptual models and end-to-end DOVER/FAST/Faster parity | Pending user-supplied local weights |
| RAFT quality with real optical flow, CUDA peak memory and reproduction of all VSR paper tables | Not executed |

Absence of public code is not treated as proof that a work has no metrics. Inference is not reported as measurement, and mock-backend tests are not presented as real model validation.
