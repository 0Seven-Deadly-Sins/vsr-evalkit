# Validation record

Date: 2026-10-15. No evaluation checkpoints, backbone weights, NIQE statistics or datasets were downloaded. PyTorch/torchvision and code-only PyIQA were installed as software dependencies. Synthetic image/video fixtures are created in temporary directories and are not distributed as model outputs.

All ten direct checkpoint/statistic URLs in the manifest responded HTTP 200 to HEAD-only metadata checks. RAFT is linked to its official download instructions. No response bodies or model files were retrieved; availability checks are not checkpoint numerical verification.

## Executed locally

Windows CPU, Python 3.10.10, numpy 1.26.4, scipy 1.14.1, Pillow 11.1.0, PyYAML 6.0.2, PyAV 14.2.0, OpenCV-headless 4.11.0.86, torch 2.5.1+cpu, torchvision 0.20.1+cpu. No CUDA device was available.

- `python -m pytest -q`: **44 passed** at the delivery validation gate; later CI and repository history record additional fixes/checks, if any.
- Actual no-weight CLI synthetic run: completed; JSON/CSV/Markdown outputs generated, paired landscape and portrait clips, differing video lengths.
- Classical formula and independent OpenCV valid-window comparisons, 1080p/4K identity, odd dimensions, color bars, crop boundaries and invalid inputs.
- Lossless FFV1/MKV versus PNG frame equality, empty/corrupt/high-bit-depth inputs, sequence dimension changes, natural ordering, explicit mappings, frame IDs, FPS and alignment rejection.
- Macro aggregation, no early rounding, valid JSON infinity, incomplete-coverage suppression, distinct protocol IDs and refusal to overwrite outputs.
- Known translation flow direction, in-bounds regions, occlusion, RGB denominator, zero-valid-pixel handling, flow-vector restoration and original RAFT 0..255 caller range with odd-size pad/unpad.
- Socket denial, local artifact lookup, imported torchvision download aliases, missing implicit backbones, RGB perceptual inputs, DOVER name-based fusion and FAST/Faster calibration contracts.

## Actual upstream classical-function differential check

`scripts/verify_references.py` executed the actual BasicSR Python functions from commit `8d56e3a045f9fb3e1d8872f92ee4a4f07f886b0a`, and actual classical functions from installed PyIQA **0.1.15.post2**, without importing learned models. Six RGB/Y cases across landscape/portrait/odd sizes, with additional explicit variant checks:

| Comparison | Maximum absolute error |
|---|---:|
| BasicSR PSNR, matching RGB/Y convention | 1.2313109110e-7 dB |
| BasicSR SSIM, matching RGB/Y convention | 3.6035385786e-10 |
| PyIQA PSNR with epsilon explicitly set to 0 | 0 |
| PyIQA SSIM with matching CS-clipped variant | 4.2489201046e-10 |

BasicSR's luminance conversion retains float32 intermediates; this toolkit retains float64. PyIQA's Gaussian kernel is initially float32. Tolerances are explicit in the verification script. Default PyIQA PSNR epsilon and SSIM CS clipping are semantic differences and are not counted as numerical error.

Contrast-inverted texture: standard/unclipped SSIM −0.9964064299, clipped CS variant 0. Identity PSNR: standard +∞, epsilon=1e-8 variant 80 dB. These fixtures demonstrate why a metric name alone is not a reproducible protocol.

To reproduce, install code-only PyIQA and torch/torchvision, then:

```bash
python scripts/verify_references.py --cache reference-cache --fetch --output reference-results.json
```

`--fetch` retrieves pinned **Python source files only**, never parameters. Remove it to reuse a local source cache.

## Actual upstream VQA preprocessing contracts

`scripts/verify_vqa_contracts.py` executed the fixed official sampler and view functions with a synthetic five-frame sequence, using this toolkit's adapter and mock score heads:

| Model | Generated model input shapes (N,C,T,H,W) |
|---|---|
| DOVER technical | (3,3,32,224,224) |
| DOVER aesthetic | (1,3,32,224,224) |
| FAST-VQA | (4,3,32,224,224) |
| FasterVQA | (1,3,32,224,224) |

Repeated seed produced identical sampled indices; calibrated mock scores yielded 0.5 as expected. This exercises **real upstream preprocessing**, but the model heads are mocks. It does not validate pretrained model numerical parity or perceptual quality.

```bash
python scripts/verify_vqa_contracts.py --cache reference-cache/vqa --fetch --output vqa-contracts.json
```

## Clean installation and Linux CI

The `no-weight-tests` GitHub Actions workflow tests Python 3.10 and 3.11 on Ubuntu CPU, installs no pretrained artifacts, and executes the test suite plus a generated CLI example. See the repository Actions page for actual run results; a configured workflow is not itself evidence of a passing run. The delivery process also checks a fresh local checkout and base installation without PyTorch.

## Pending (not claimed)

- Real LPIPS/DISTS/NIQE/MUSIQ/CLIP-IQA numerical values under the selected pretrained artifacts.
- DOVER/FAST-VQA/FasterVQA complete model loading and direct official-script numerical parity under identical inputs/seeds.
- Actual RAFT Things optical flow accuracy and real-checkpoint temporal values.
- Linux CUDA execution, peak VRAM, runtime or 4K learned-metric throughput measurements.
- Any VSR model inference, dataset download, human quality study, or reproduction of paper result tables.

After supplying local artifacts, run `scripts/validate_local_weights.py`, and then evaluate the actual prepared data. This script checks identity/finiteness/repeatability. Paper reproduction additionally needs matching dataset version, degradation, frame interval, color/decode and save settings; no single smoke test certifies it.
