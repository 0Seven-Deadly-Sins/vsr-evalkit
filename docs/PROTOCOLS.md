# Evaluation protocols

## Classical image metrics

Inputs are decoded 8-bit RGB and normalized to [0,1]. Default calculations retain float64. PSNR = −10 log10(mean((P−G)²)); zero error returns +∞. The default adds no epsilon. A preset may explicitly add `psnr_eps` to the normalized MSE.

SSIM uses a separable 11×11 Gaussian (σ=1.5), constants C1=0.01², C2=0.03², population moments and the valid window region. Compute each channel's SSIM map, then average all valid pixels/channels. The standard keeps negative contrast-structure responses. `ssim_clip_negative_cs: true` changes that definition, following PyIQA's ReLU on the CS term. `ssim_round_values: true` rounds the evaluated channels to the nearest 8-bit value before SSIM.

`bt601-y` is studio-range MATLAB/BasicSR luminance:

```text
Y = (16 + 65.481 R + 128.553 G + 24.966 B) / 255
```

Here R/G/B are already in [0,1]. `yiq-y` is full-range Y = 0.299R + 0.587G + 0.114B. They are distinct. `crop_border` is applied only to PSNR/SSIM and must leave at least 11×11 pixels for SSIM. Perceptual and NR metrics never receive the classical Y conversion or border crop. Common-shape alignment, when explicitly enabled, applies to all FR pairs; NR/video/temporal metrics still use the original prediction sequence.

## Presets

| Preset | Classical channels | Alignment | Other differences |
|---|---|---|---|
| standard-rgb-v1 | RGB / RGB | strict | No epsilon or CS clipping |
| standard-y-v1 | BT.601 Y / BT.601 Y | strict | Explicit studio-range conversion |
| flow-960-v1 | RGB / RGB | strict | Limit flow estimation long edge to 960; full-size error |
| dove-code-alignment-v1 | RGB PSNR / rounded YIQ SSIM | top-left common size, common frame prefix, ordinal pairing | PSNR eps=1e-8, SSIM CS clipped; no paper reproduction claim |

Configuration may use `protocol: {preset: standard-rgb-v1, crop_border: 4, seed: 7}`. `ssim_color_space` is independent of `color_space` (PSNR). Changing any effective option changes the protocol ID. The DOVE preset documents current disclosed code with a chosen PyIQA version, because DOVE did not pin its package. Floating-point implementation differences are measured in VALIDATION.md.

## Image backend definitions

- LPIPS: AlexNet IMAGENET1K_V1, calibration version 0.1; RGB [0,1], PyIQA's internal normalization.
- DISTS: VGG16 IMAGENET1K_V1 plus learned alpha/beta parameters, native PyIQA preprocessing.
- NIQE: original `niqe_modelparameters.mat`, PyIQA's YIQ luminance option, no border crop; minimum dimensions 192×192. Degenerate statistics can still yield undefined output, reported as failure.
- MUSIQ: KonIQ-10k variant, not AVA/SPAQ. CLIP-IQA: plain RN50, not CLIP-IQA+.
- Every model uses eval/inference mode. Metric-specific fixed internal preprocessing is part of the definition and is not replaced by a global thumbnail or tiled score average.

## Video quality models

The exact configs come from commit-pinned upstream source. DOVER uses `dover.yml`, `val-l1080p`, its UnifiedFrameSampler and get_single_view. FAST-VQA uses `options/fast/fast-b.yml`; FasterVQA uses `options/fast/f3dvqa-b.yml`, each using `val-kv1k`, the upstream samplers and spatial fragments. Short inputs use upstream modulo/repetition rules; actual sampled indices are stored, so repetition is visible.

Normalization is (RGB8 − [123.675,116.28,103.53]) / [58.395,57.12,57.375]. No video re-encoding or fake FPS is introduced. Only sampled frames are decoded/retained for each view. Memory for original-sized CPU samples is bounded by the upstream sample count, and can still be substantial for 4K. GPU views have the upstream spatial dimensions.

RNGs use a stable hash of clip ID plus `protocol.seed`, independent of selected model and traversal order. Seeds and indices are recorded. Identical seeds do not guarantee bitwise CUDA reproducibility.

DOVER list outputs are first bound to the names/order of the input view dictionary. With raw technical T and aesthetic A:

```text
t = (T - 0.1107) / 0.07355
a = (A + 0.08285) / 0.03774
overall = sigmoid(0.6104*t + 0.3896*a)
```

Technical/aesthetic component probabilities and raw branch scores are also retained. FAST-VQA calibration is sigmoid((raw + 0.110198185)/0.04178565); FasterVQA calibration is sigmoid((raw − 0.14759505)/0.03613452). Scores stay in [0,1], without multiplying by 100.

## RAFT warping definition

This project's metric is **masked RGB MSE with RAFT Things**, not a promise that every paper's `E*warp` is identical.

1. Estimate F: I(t)→I(t+1) and B: I(t+1)→I(t), using full RAFT Things, 20 iterations and float32, with mixed precision/alternate correlation disabled. Original RAFT receives RGB [0,255]; its own normalization divides by 255 exactly once.
2. Pad symmetrically with replicated borders to multiples of eight, then remove padding from both fields before scoring. Require at least 128 pixels per flow-input dimension.
3. Warp I(t+1) at coordinate x+F(x), using bilinear `grid_sample`, align_corners=True. Coordinates outside the original image are invalid.
4. Sample B at x+F(x). Exclude forward/backward inconsistent pixels where
   `||F + warp(B,F)||² > 0.01*(||F||² + ||warp(B,F)||²) + 0.5`.
5. On remaining M pixels, pair error is `sum_x,c M(x)*(I_t(x,c)-warped(x,c))² / (3*sum_x M(x))`, with image values in [0,1]. No motion-gradient exclusion is added. Sum uses float64. Record valid pixels.
6. Average all defined adjacent pair errors equally. A single-frame video or a pair with zero valid pixels is undefined, never a zero score. Partial pair coverage is exposed and prevents a dataset aggregate.

The optional 960 preset resizes only flow inputs with bilinear align_corners=False, restores fields to the original H/W, and multiplies horizontal displacement by W/original_flow_W and vertical displacement by H/original_flow_H. Scoring still uses full-resolution images. All dimensions/padding are recorded. Native OOM is reported; there is no automatic fallback.

The raw value and `scaled_1000` = raw×1000 are both returned. x1000 is a presentation unit, not another metric. The definition differs from upstream wrappers that resize images before scoring, use different masks, sum RGB channels without division by three, or normalize RAFT inputs incorrectly.

## Pairing, aggregation and comparison

Clip IDs must match completely unless an explicit mapping CSV enumerates the intended population. Equal IDs with different extensions are allowed across prediction/GT, but duplicate IDs inside one root are rejected. Frame directories use natural order; identity mode checks the last numeric field of each frame stem, or the full stem if nonnumeric. Mixed video/frames use decoded order. Constant-rate elapsed PTS and available FPS must agree; container start offsets are recorded but not interpreted as a scene shift.

Default alignment rejects unequal counts/shapes. CFR validation compares elapsed PTS to the nominal (or inferred) regular grid with one container time-base tick of tolerance, so millisecond-rounded 16/29.97 FPS MKV timestamps are not misclassified as VFR. Cross-video timeline comparison allows the two containers' combined timestamp quantum, without a duration-dependent relative tolerance. Actual irregular/nonmonotonic timelines still fail. Explicit common-size presets record every truncation/crop. No temporal interpolation or image resizing aligns GT. Grayscale/palette images are converted to RGB; alpha and high-bit-depth images are rejected. Standard video decode is FFmpeg rgb24 with source format/range/colorspace recorded. HDR detection rejects known HDR transfer tags and >8-bit formats; externally verify untagged HDR inputs, which cannot be reliably identified from pixels alone.

Per-frame means form per-video scores; dataset scores are video-macro means with no early rounding. Every required clip must have a complete metric score for a dataset aggregate. Missing/error/partial clips produce no aggregate. Explicitly undefined metrics remain null with a reason. JSON infinity is `"+inf"`. Comparison groups include protocol, metric artifact hashes, source commits, dependency versions, clip IDs, reference paths, sizes, FPS, frame counts and expected coverage. Reference paths conservatively separate distinct mappings; comparison groups are local-run guards, not cross-machine dataset content hashes. Compare only equal dataset/metric/group entries; no mixed-protocol ranking is generated.
