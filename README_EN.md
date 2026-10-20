<div align="center">

<img src="assets/hero.svg" alt="VSR EvalKit — independent, protocol-explicit video quality evaluation" width="100%">

[![简体中文](https://img.shields.io/badge/语言-简体中文-475569?style=for-the-badge)](README.md) [![English](https://img.shields.io/badge/Language-English-2563EB?style=for-the-badge)](README_EN.md)

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white) ![11 metrics](https://img.shields.io/badge/metrics-11-0F9D8A?style=flat-square) ![No auto-download](https://img.shields.io/badge/weights-no%20auto--download-6B5BD2?style=flat-square) [![License](https://img.shields.io/badge/license-Apache--2.0-64748B?style=flat-square)](LICENSE)

**Evaluate pre-generated video super-resolution results independently.** Configure paths once, then run by model and dataset ID—without coupling evaluation to inference code.

[Quick start](#quick-start) · [Metrics and protocols](#metrics-and-protocols) · [Documentation](#documentation) · [Validation status](#validation-status)

</div>

```bash
vsr-eval run --config eval.local.yaml --model dove --dataset udm10
vsr-eval run --config eval.local.yaml --model dove,seedvr2 --dataset udm10,videolq
```

`vsr-evalkit` evaluates videos you have already generated; it does not run VSR inference. It accepts PNG/JPEG frame sequences and MP4, MKV, MOV or AVI. Predictions and ground truth can use different storage formats, with explicit mappings for naming differences.

## Why this toolkit

| Separate model and dataset paths | Explicit evaluation protocols | Auditable results |
|---|---|---|
| Register a new output directory without copying metric scripts into each model repository. | No silent crop, resize or truncation. Color space, sampling, aggregation and flow settings are identified by the protocol. | Per-frame/pair, per-video and dataset outputs record weight hashes, dependency versions, sample indices and failure reasons. |

Videos may have different resolutions and aspect ratios; dimensions must stay fixed within each sequence. The default scope is SDR, 8-bit video.

## Metrics and protocols

| Evaluation type | Metrics | Ground truth required |
|---|---|---:|
| Full-reference image quality | PSNR, SSIM, LPIPS-Alex v0.1, DISTS | Yes |
| No-reference image quality | NIQE, MUSIQ-KonIQ, CLIP-IQA | No |
| No-reference video quality | DOVER, FAST-VQA, FasterVQA | No |
| Temporal consistency | RAFT bidirectional flow with masked warping MSE | No |

With GT, `metrics: auto` selects all 11 metrics; without GT, it selects the seven no-reference, video and temporal metrics. The default `standard-rgb-v1` protocol uses RGB, no border crop, strict frame/size checks, per-frame averaging within each video, then equal weighting across videos. See the [metric protocols](docs/PROTOCOLS.md) for complete definitions.

## Quick start

Python 3.10 or 3.11. This CPU demo creates a small synthetic PNG dataset and needs no evaluation weights or external data:

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\Activate.ps1
python -m pip install -e .
python scripts/make_demo.py --output demo-input
vsr-eval doctor --config demo-input/eval.yaml --model demo --dataset synthetic
vsr-eval run --config demo-input/eval.yaml --model demo --dataset synthetic --device cpu
```

To evaluate your own results, start from the complete [`examples/eval.yaml`](examples/eval.yaml). Here is a minimal configuration for running PSNR/SSIM on CPU without model weights:

```yaml
version: 1
models:
  dove:
    pred_root: /data/results/dove/{dataset}
  seedvr2:
    pred_root: /data/results/seedvr2/{dataset}
datasets:
  udm10:
    gt_root: /data/UDM10/GT
  videolq: {}
metrics: [psnr, ssim]
protocol: standard-rgb-v1
runtime:
  device: cpu
```

```bash
vsr-eval doctor --config eval.local.yaml --model dove --dataset udm10
vsr-eval run --config eval.local.yaml --model dove,seedvr2 --dataset udm10,videolq
```

`psnr` and `ssim` run on CPU without PyTorch or model weights. To run all 11 metrics, set `metrics: auto` and manually prepare the required weights, backbones and NIQE statistics using the [weights guide](docs/WEIGHTS_EN.md); the toolkit never auto-downloads them. The complete evaluation environment primarily targets Linux with an NVIDIA GPU.

## Reproducible outputs

Each run writes `report.json`, `details.csv`, `summary.csv` and `comparison.md`. Protocols, coverage, sampled frames, weight SHA-256 hashes and dependency versions are recorded. Results from different protocols or sample populations are not mixed; mismatched frame counts, missing weights and evaluation failures are reported instead of silently truncating videos or averaging only successful samples.

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

**Validation boundary:** no-weight tests and classical metric comparisons have run. Numerical validation with real pretrained weights, CUDA validation and paper-table reproduction remain pending. The repository bundles no datasets, model outputs or evaluation weights and does not claim to reproduce published scores.

## Documentation

[21-work source survey](docs/SURVEY.md) · [Metric protocols](docs/PROTOCOLS.md) · [Source audit](docs/AUDIT_EN.md) · [Dataset setup](docs/DATASETS.md) · [Validation record](docs/VALIDATION.md) · [Weights and environment](docs/WEIGHTS_EN.md) · [Third-party notices](THIRD_PARTY_NOTICES.md)

中文入口：[README.md](README.md)。
