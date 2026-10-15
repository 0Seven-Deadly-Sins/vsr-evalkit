# vsr-evalkit

[![简体中文](https://img.shields.io/badge/Language-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-blue?style=for-the-badge)](README.md) [![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](README_EN.md)

Evaluate **pre-generated** video super-resolution (VSR) outputs independently of restoration model code. Configure model output paths, datasets, metrics and protocols once; select model and dataset IDs whenever you run an evaluation.

```bash
vsr-eval run --model dove --dataset udm10
vsr-eval run --model dove,seedvr2 --dataset udm10,videolq
```

## Install and run a CPU demo

Python 3.10 or 3.11 is recommended. The demo creates small synthetic PNG sequences and needs no downloaded data or pretrained artifacts.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/make_demo.py --output demo-input
vsr-eval doctor --config demo-input/eval.yaml --model demo --dataset synthetic
vsr-eval run --config demo-input/eval.yaml --model demo --dataset synthetic --device cpu
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## Configure your paths

Copy [`examples/eval.yaml`](examples/eval.yaml) to `eval.local.yaml` and edit the paths once. This local file is ignored by Git.

```yaml
version: 1
models:
  dove:
    pred_root: /data/results/dove/{dataset}
  my_model:
    pred_root: /data/results/my_model/{dataset}
datasets:
  udm10:
    gt_root: /data/UDM10/GT
  videolq: {}
metrics: [psnr, ssim]
protocol: standard-rgb-v1
runtime:
  device: cpu
output_root: runs
```

The toolkit supports PNG/JPEG frame sequences and MP4/MKV/MOV/AVI, with GT and predictions stored in different formats. It pairs clip IDs strictly, naturally sorts frame directories and checks frame identities, counts, sizes and available timing. Use an explicit [`mapping.csv`](examples/mapping.csv) when names differ. It does not silently truncate, resize or align mismatched sequences.

You can provide known dataset FPS for frame directories; a mismatch with video timestamps is an error. Unknown FPS stays unknown, and video metrics sample by frame index without inventing 8/24/30 FPS. V0.1 supports SDR 8-bit constant-frame-rate inputs; VFR, HDR, higher bit depth and automatic alignment of interpolated outputs are outside the default protocol. Different videos may have different dimensions and aspect ratios; a single sequence must keep a fixed frame size.

For datasets without GT, `metrics: auto` selects the seven no-reference, video-quality and temporal metrics. If a requested full-reference metric has no GT, preflight reports an error.

## Metrics and protocols

| Category | Metrics |
|---|---|
| Full-reference | PSNR, SSIM, LPIPS-Alex v0.1, DISTS |
| No-reference image | NIQE, MUSIQ-KonIQ, CLIP-IQA |
| No-reference video | DOVER, FAST-VQA, FasterVQA |
| Temporal consistency | RAFT masked RGB warping MSE, with raw and ×1000 display values |

The default `standard-rgb-v1` protocol uses RGB, no border crop, all frames, strict size/count matching and video-macro aggregation: frame scores are averaged per video, then videos receive equal weight. SSIM uses an 11×11 Gaussian window with σ=1.5. LPIPS and DISTS always receive RGB.

PSNR and SSIM have independent weight-free implementations. Learned image metrics use pinned PyIQA; DOVER, FAST-VQA/FasterVQA and RAFT use fixed upstream code and preprocessing. DOVER/VQA sampling indices and seeds are recorded. Explicit alternatives include BT.601 luminance and `flow-960-v1`; every effective protocol has a distinct ID. See [protocol definitions](docs/PROTOCOLS.md) and the [audit](docs/AUDIT_EN.md).

RAFT at native 4K resolution may run out of memory. The default fails explicitly; it does not silently lower resolution. `flow-960-v1` limits only the flow-estimation long edge, rescales the flow vectors correctly and scores error on the original frames. The DOVE code-alignment preset captures disclosed code choices with the pinned PyIQA version; because DOVE did not pin PyIQA, it does not promise to reproduce a paper table or repeat known bugs.

Useful commands:

```bash
vsr-eval list --config eval.local.yaml
vsr-eval weights info --metric lpips
vsr-eval doctor --model dove --dataset udm10
vsr-eval run --model dove --dataset udm10 --metrics psnr,ssim --device cpu
vsr-eval run --model dove --dataset udm10 --protocol standard-y-v1
vsr-eval run --model dove --dataset udm10 --protocol flow-960-v1
```

## Full Linux/CUDA environment

Full learned-metric evaluation targets Linux with an NVIDIA GPU. Install a PyTorch/torchvision pair that matches your CUDA setup, then install the deep dependencies and manually configure the required upstream source checkouts and model artifacts:

```bash
python -m pip install -r requirements/deep-linux.txt
python scripts/source_commands.py
vsr-eval weights info
```

`source_commands.py` only prints commands; it does not download source. No pretrained weights, NIQE statistics, datasets or VSR results are bundled or automatically downloaded. Missing files fail preflight. Follow the [manual weights and environment guide](docs/WEIGHTS_EN.md) and review [third-party terms](THIRD_PARTY_NOTICES.md) before using optional upstream components.

## Results and comparison

Each run writes `report.json`, `details.csv`, `summary.csv` and `comparison.md`, including per-frame/pair, per-video and dataset values, coverage, failures, protocol fingerprint, source commits, checkpoint hashes and VQA sampling indices. A failed or partially covered metric has no dataset aggregate. Compare only matching dataset, metric and `comparison_group` entries; the toolkit does not rank mixed protocols or populations.

Exit code `0` means the run completed (including an explicitly not-applicable metric), `1` indicates evaluation failure/incomplete coverage and `2` indicates a configuration, input or preflight error. `--output` must name a new directory.

## Documentation

[Source survey](docs/SURVEY.md) · [Protocols](docs/PROTOCOLS.md) · [Audit](docs/AUDIT_EN.md) · [Dataset setup](docs/DATASETS.md) · [Validation record](docs/VALIDATION.md) · [Weights](docs/WEIGHTS_EN.md) · [Third-party notices](THIRD_PARTY_NOTICES.md)

**Validation status:** no-weight tests, source-contract checks and actual classical-function differential checks were run. Real learned-weight numerical parity, Linux CUDA behavior and reproduction of paper tables remain pending. No weights were downloaded. See the [validation record](docs/VALIDATION.md) and run `scripts/validate_local_weights.py` after configuring your own artifacts. Passing that smoke test does not establish paper-table reproduction.
