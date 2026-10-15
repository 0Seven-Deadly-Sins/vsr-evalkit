# Offline environment and weights

[![简体中文](https://img.shields.io/badge/Language-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-blue?style=for-the-badge)](WEIGHTS.md) [![English](https://img.shields.io/badge/Language-English-blue?style=for-the-badge)](WEIGHTS_EN.md)

This toolkit does not download weights. `vsr-eval weights info` prints the complete machine-readable manifest at `src/vsr_evalkit/resources/weights.json`, including implicit backbones. Each selected file is checked for existence and nonzero size before model construction. Its SHA-256 is calculated at runtime and checked against a known full hash or an official filename hash prefix. Unknown implicit download requests fail explicitly; random initialization is never substituted for missing artifacts.

## Linux GPU environment

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Example only: adjust the CUDA wheel index for your driver. This installs runtimes, not pretrained weights.
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements/deep-linux.txt
```

`pyiqa==0.1.15.post2` and `timm==0.9.16` are pinned. Do not casually replace them with latest releases and mix the results. `decord`/`skvideo` are import dependencies of upstream code; this toolkit consistently decodes with PyAV. PSNR/SSIM alone do not require the deep environment. Linux/CUDA still needs validation with real artifacts on your machine; CPU tests do not establish GPU behavior.

The base package pins `av==14.2.0`, which supplies wheels for the target Python/Linux/Windows environments. Versions 14.3/14.4 may fall back to source distributions requiring additional FFmpeg compilation, so they are not used in the default setup.

## Metric source checkouts

```bash
python scripts/source_commands.py
```

This command prints three sets of `git clone --no-checkout` and `git checkout --detach <full commit>` commands; it does not download anything. After manually executing the printed **source-only** commands, configure `sources.dover`, `sources.fastvqa` and `sources.raft`. Pinned revisions, license observations and candidate files are in the [survey](SURVEY.md); full commit IDs are in `src/vsr_evalkit/resources/sources.json`. You do not need to install these projects' VSR or training models.

The backend verifies source HEADs and rejects tracked modifications. Keep weights in a separate `weights/` directory and do not edit upstream YAML files. Model code and sampling configuration come from the pinned checkout; this toolkit overrides checkpoint paths through its own config. Older DOVER/FAST-VQA code can have CUDA/dependency conflicts. If you need separate environments, install this toolkit in each and select metrics separately; results retain source information and must not be combined across comparison groups.

## Files to obtain manually

| Config key | File/version | Official/upstream release source |
|---|---|---|
| alexnet | `alexnet-owt-7be5be79.pth` | [torchvision](https://download.pytorch.org/models/alexnet-owt-7be5be79.pth) |
| lpips_alex | `LPIPS_v0.1_alex-df73285e.pth` | [PyIQA weights](https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/LPIPS_v0.1_alex-df73285e.pth) |
| vgg16 | `vgg16-397923af.pth` | [torchvision](https://download.pytorch.org/models/vgg16-397923af.pth) |
| dists | `DISTS_weights-f5e65c96.pth` | [PyIQA weights](https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/DISTS_weights-f5e65c96.pth) |
| niqe_params | `niqe_modelparameters.mat`, original statistics | [PyIQA weights](https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/niqe_modelparameters.mat) |
| musiq_koniq | `musiq_koniq_ckpt-e95806b9.pth` | [PyIQA weights](https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/musiq_koniq_ckpt-e95806b9.pth) |
| clip_rn50 | `RN50.pt` | [OpenAI CLIP](https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt) |
| dover | `DOVER.pth`, full DOVER, not Mobile | [Official release v0.1.0](https://github.com/VQAssessment/DOVER/releases/tag/v0.1.0) |
| fastvqa | `FAST_VQA_B_1_4.pth` | [Official release v2.0.0](https://github.com/VQAssessment/FAST-VQA-and-FasterVQA/releases/tag/v2.0.0) |
| fastervqa | `FAST_VQA_3D_1_1.pth` | [Official release v2.0.0](https://github.com/VQAssessment/FAST-VQA-and-FasterVQA/releases/tag/v2.0.0) |
| raft_things | `models/raft-things.pth`, full Things | [Official RAFT download instructions](https://github.com/princeton-vl/RAFT#demos) |

LPIPS requires **alexnet + lpips_alex**; DISTS requires **vgg16 + dists**. The head parameters alone are insufficient. The NIQE `.mat` file is also required. Plain CLIP-IQA does not require CLIP-IQA+'s learned prompt file. FasterVQA and FAST-VQA are not interchangeable; do not substitute `_Scr`, Mobile, MS or MT variants.

PyIQA also publishes some same-named files on [GitHub v0.1-weights](https://github.com/chaofengc/IQA-PyTorch/releases/tag/v0.1-weights). Use an alternate source only when the file and matching hash are identical; do not substitute a different training version. RAFT's official `download_models.sh` retrieves a model archive. After you run it manually, configure only its `raft-things.pth` here; this toolkit never executes the script.

For example, a user may choose to download one file with the following command; it was **not run for this delivery**:

```bash
mkdir -p weights
curl --fail --location 'https://github.com/VQAssessment/DOVER/releases/download/v0.1.0/DOVER.pth' --output weights/DOVER.pth
```

Configured paths do not need to use the upstream basename; the resolver maps implicit requests to local files by config key. Do not pass untrusted pickle weights to `torch.load`; only the official source files are documented here.

## Preflight and real-weight validation

```bash
cp examples/eval.yaml eval.local.yaml
# Edit the actual local paths first.
vsr-eval weights info
vsr-eval doctor --model dove --dataset udm10
python scripts/validate_local_weights.py --config eval.local.yaml --output local-validation.json
vsr-eval run --model dove --dataset udm10
```

`doctor` checks inputs, selected local artifacts, the PyTorch device, PyIQA version and upstream source commits; it does not run the models. Actual model construction/inference is checked by `run` or the validation script. During evaluation, Python socket connections are blocked. torchvision/PyIQA/CLIP download helpers are redirected to the local manifest, and lazy architecture imports occur before aliases are bound. Missing artifacts are never looked up online.

The validation script checks perceptual identity, finite outputs, repeatability of video models with the same seed and whether the temporal path runs. Passing it does not reproduce a paper table; that additionally requires matching dataset version, degradation, preprocessing, saved encoding and sampling protocol.
