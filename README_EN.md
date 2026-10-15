# vsr-evalkit

Evaluate **pre-generated** video super-resolution outputs independently of restoration model code. Configure model output paths, datasets and protocols once; select model/dataset IDs thereafter.

```bash
python -m pip install -e .
python scripts/make_demo.py --output demo-input
vsr-eval run --config demo-input/eval.yaml --model demo --dataset synthetic --device cpu
```

Copy `examples/eval.yaml` to `eval.local.yaml`, set your paths, then:

```bash
vsr-eval run --model dove --dataset udm10
vsr-eval run --model dove,seedvr2 --dataset udm10,videolq
vsr-eval doctor --model dove --dataset udm10
vsr-eval weights info
```

Default execution is offline. No weights, backbone checkpoints, NIQE statistics or datasets are bundled or automatically downloaded. Missing artifacts fail preflight. PSNR/SSIM need no weights or PyTorch. Full evaluation targets Linux/CUDA; install compatible torch/torchvision builds first, then `pip install -r requirements/deep-linux.txt` and follow [WEIGHTS](docs/WEIGHTS.md).

Metrics: PSNR, SSIM, LPIPS-Alex v0.1, DISTS, NIQE, MUSIQ-KonIQ, CLIP-IQA, DOVER, FAST-VQA, FasterVQA, and explicitly defined RAFT masked warping MSE. DOVER and FAST-VQA use their fixed official sampling, not an arbitrary global resize.

Supports PNG/JPEG frame sequences and MP4/MKV/MOV/AVI. Strict pairing is the default: no silent truncation, resizing or shape correction. Standard PSNR/SSIM use RGB, zero border crop and video-macro aggregation. Choose luminance, disclosed DOVE alignment, or reduced-resolution flow estimation explicitly; each gets a distinct protocol ID. SDR8 and constant-rate video only in v0.1.

Results include per-frame/pair, per-video and dataset values, JSON/CSV/Markdown reports, coverage, failures, protocol fingerprints, source commits, checkpoint hashes and sampling indices. Incomplete metrics have no dataset score. Only compare equal dataset/metric/comparison groups.

[Source survey](docs/SURVEY.md) · [Protocol formulas](docs/PROTOCOLS.md) · [Audit](docs/AUDIT.md) · [Validation](docs/VALIDATION.md) · [Licenses](THIRD_PARTY_NOTICES.md)

**Validation limits:** no-weight tests and actual classical-function differential checks were run. Real checkpoint numerical parity, CUDA execution and paper-table reproduction remain pending. `scripts/validate_local_weights.py` provides identity, finiteness and repeatability checks after you manually configure artifacts. Passing it does not establish paper-table reproduction.
