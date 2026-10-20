# vsr-evalkit

<a href="README.md"><kbd>简体中文</kbd></a> <a href="README_EN.md"><kbd>English</kbd></a>

An independent evaluation toolkit for pre-generated video super-resolution (VSR) results. Configure model output paths, datasets, metrics and evaluation protocols separately to compare multiple models through the same workflow.

A “model” here is an ID for a set of generated results. The toolkit does not run VSR inference or require the restoration model's inference or training code. After configuring `eval.local.yaml`, select model and dataset IDs:

```bash
vsr-eval run --model dove --dataset udm10
vsr-eval run --model dove,seedvr2 --dataset udm10,spmcs
```

[Quick start](#quick-start) · [Configure your results](#configure-your-results) · [Metrics and protocols](#metrics-and-protocols) · [Full metric environment](#full-metric-environment) · [Validation status](#validation-status) · [Documentation](#documentation)

## Supported inputs

- **Storage formats:** PNG/JPEG frame sequences and MP4, MKV, MOV or AVI videos. Predictions and ground truth (GT) can use different storage formats; explicit mapping files handle naming differences.
- **Video dimensions:** different videos can have different resolutions and aspect ratios; dimensions must stay fixed within each sequence. Individual metrics retain their input-size and memory requirements.
- **Default data scope:** SDR, 8-bit data. Video files must have a constant frame rate; unknown frame-sequence FPS stays unknown. VFR, HDR, higher bit depth and automatic temporal alignment of interpolated outputs are outside the default scope.

## Quick start

Requires Python ≥3.10; the verified versions are 3.10 and 3.11. The installation example below uses Bash:

```bash
git clone https://github.com/0Seven-Deadly-Sins/vsr-evalkit.git
cd vsr-evalkit
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

Then run a CPU demo that requires no evaluation weights:

```bash
python scripts/make_demo.py --output demo-input
vsr-eval doctor --config demo-input/eval.yaml --model demo --dataset synthetic
vsr-eval run --config demo-input/eval.yaml --model demo --dataset synthetic --device cpu
```

The demo script creates a small synthetic PNG dataset and a PSNR/SSIM configuration in a new `demo-input` directory. It needs no dataset downloads, pretrained weights or PyTorch. `doctor` checks inputs, dependencies and local artifacts without running evaluation models; `run` performs the actual computation.

## Configure your results

Save the following configuration as `eval.local.yaml` at the repository root and replace the example paths. This file is ignored by Git. Both UDM10 and SPMCS in this example require correctly paired GT; the selected PSNR/SSIM metrics need no weights.

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
  spmcs:
    gt_root: /data/SPMCS/GT
metrics: [psnr, ssim]
protocol: standard-rgb-v1
runtime:
  device: cpu
output_root: runs
```

`{dataset}` is replaced by the selected dataset ID, for example `/data/results/dove/udm10`. Model and dataset IDs are user-defined configuration entries; they do not launch model inference or download data.

Sequences under the prediction and GT roots are paired by clip ID. For example, both roots can contain a `clip001/` frame directory, or one can contain `clip001.mp4` and the other `clip001/`. Frame sequences use natural sorting, with additional frame-identity checks when both inputs are frame directories. Frame counts, dimensions and available timing are checked for all input formats. See [dataset setup](docs/DATASETS.md) for directory layouts and explicit mappings.

```bash
vsr-eval doctor --model dove --dataset udm10
vsr-eval run --model dove,seedvr2 --dataset udm10,spmcs
```

Without `--config`, commands read `eval.local.yaml` from the current directory. Multiple model and dataset IDs run all combinations; the batch command above evaluates four combinations.

Datasets without GT, such as VideoLQ, should omit `gt_root` and cannot use the PSNR/SSIM selection above. See [full metric environment](#full-metric-environment) for evaluation without GT.

## Metrics and protocols

The table lists command-line metric IDs. ↑ means higher is better; ↓ means lower is better.

| Type | Metric IDs and direction | Version or definition | GT required |
|---|---|---|---|
| Full-reference image | `psnr` ↑, `ssim` ↑ | Independent float64 implementation, compared against BasicSR / PyIQA | Yes |
| Full-reference perceptual | `lpips` ↓, `dists` ↓ | LPIPS-Alex v0.1, DISTS; PyIQA 0.1.15.post2 | Yes |
| No-reference image | `niqe` ↓ | Original NIQE statistics; PyIQA 0.1.15.post2 | No |
| No-reference image | `musiq` ↑, `clipiqa` ↑ | MUSIQ-KonIQ, plain CLIP-IQA RN50 | No |
| No-reference video | `dover` ↑, `fastvqa` ↑, `fastervqa` ↑ | Pinned official source, sampling configuration and weight variants | No |
| Temporal consistency | `ewarp` ↓ | RAFT Things bidirectional flow and occlusion-masked RGB warping MSE | No |

`metrics: auto` selects all 11 metrics when GT is available, or seven no-reference, video-quality and temporal metrics otherwise. It does not skip missing dependencies or weights. Explicitly requesting a full-reference metric without GT fails preflight.

The default `standard-rgb-v1` protocol:

- Strictly checks samples, frame counts, dimensions and available timing, without automatic resizing, cropping or truncation.
- Uses RGB and no border crop for PSNR/SSIM, with an 11×11 Gaussian window and σ=1.5 for SSIM. `standard-y-v1` explicitly switches these two metrics to BT.601 Y; LPIPS/DISTS always receive RGB.
- Averages image metrics over frames within each video, and temporal metrics over adjacent frame pairs, then weights videos equally. Video-quality models use pinned upstream sampling to produce one score per video.
- Estimates RAFT flow at native resolution. Out-of-memory errors fail explicitly. The optional `flow-960-v1` preset reduces only the flow-estimation input, restores the flow vectors, and still scores the original-resolution frames.

See [metric protocols](docs/PROTOCOLS.md) for formulas, sampling, aggregation and compatibility presets. Metric definitions, preprocessing and weight variants all affect scores; a shared metric name alone does not establish comparability.

## Full metric environment

Metrics beyond PSNR/SSIM require additional dependencies and local weights or statistics. DOVER, FAST-VQA/FasterVQA and RAFT also need pinned upstream metric source checkouts. The complete environment primarily targets Linux with an NVIDIA GPU; follow the [weights and environment guide](docs/WEIGHTS_EN.md).

```bash
vsr-eval weights info
python scripts/source_commands.py
```

The first command lists required files and their sources; the second only prints commands for obtaining pinned source checkouts. The toolkit never auto-downloads evaluation weights, implicit backbones or NIQE statistics. Prepare the files manually and configure `sources` and `weights` as described in the guide.

The complete template is [examples/eval.yaml](examples/eval.yaml). Once dependencies and artifacts are ready, set `metrics` to `auto`, run `doctor`, then evaluate. The template includes the unpaired entry `videolq: {}`; if you are extending the basic configuration above, add this entry under `datasets` first. With the corresponding output directory ready, run:

```bash
vsr-eval run --model dove --dataset videolq --metrics auto
```

This command selects the seven no-reference, video and temporal metrics. Missing artifacts fail preflight; passing `doctor` does not establish successful model loading or numerical correctness.

## Outputs and comparison

By default, each run creates a new directory under `runs/`:

| File | Contents |
|---|---|
| `report.json` | Complete run record, per-video results and dataset aggregates |
| `details.csv` | Per-frame image scores, temporal frame-pair scores and video-metric records |
| `summary.csv` | Summaries by model, dataset and metric |
| `comparison.md` | Model comparison tables |

Results record protocol IDs, dependency versions, source commits, weight SHA-256 hashes, sample indices, coverage and failure reasons. Scores from different protocols, weights or sample populations are not automatically ranked together. Failed or partially covered metrics have no dataset aggregate. Temporal metrics for single-frame videos are marked not applicable rather than assigned zero.

## Validation status

- **Source inspection:** pinned versions, evaluation entry points and protocol differences are recorded for VSR projects and metric upstreams. Confirmed issues and evidence are in the [source audit](docs/AUDIT_EN.md).
- **Passed:** no-weight tests, numerical comparisons with upstream classical functions, synthetic CLI evaluation and Linux CPU CI. Environments, cases and verified commits are documented in the [validation record](docs/VALIDATION.md).
- **Pending:** numerical comparisons using real pretrained weights, Linux CUDA validation and paper-table reproduction. The repository does not claim to reproduce published scores.

Run local tests with:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

Some temporal and backend tests require PyTorch and are skipped when it is unavailable; see the validation record for the full environment. The repository includes no datasets, VSR outputs or evaluation weights.

## Documentation

| Subject | Guide |
|---|---|
| VSR projects and metric upstreams | [Source survey](docs/SURVEY.md) |
| Formulas, preprocessing, sampling and aggregation | [Metric protocols](docs/PROTOCOLS.md) |
| Confirmed issues and implementation differences | [Source audit](docs/AUDIT_EN.md) |
| Dataset examples, directory layouts and naming maps | [Dataset setup](docs/DATASETS.md) |
| Dependencies, weight sources and offline configuration | [Weights and environment](docs/WEIGHTS_EN.md) |
| Completed checks and validation with real weights | [Validation record](docs/VALIDATION.md) |
| Third-party licenses and citations | [Third-party notices](THIRD_PARTY_NOTICES.md) |

This project's code is licensed under [Apache-2.0](LICENSE). Optional metric code and weights retain their own terms, including noncommercial restrictions on some upstream components; see the third-party notices.
