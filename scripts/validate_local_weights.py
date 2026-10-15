"""Run identity/finiteness/repeatability checks using USER-SUPPLIED weights only."""

import argparse
import json
from pathlib import Path

import numpy as np

from vsr_evalkit.backends import create_backend, preflight_backends
from vsr_evalkit.config import Config
from vsr_evalkit.media import MediaInfo
from vsr_evalkit.offline import check_artifacts, fingerprint, network_disabled
from vsr_evalkit.registry import METRICS


class TinySequence:
    def __init__(self):
        self.info = MediaInfo(4, 192, 256, 25.0, [0, 0.04, 0.08, 0.12], "synthetic", {})

    def read_indices(self, indices):
        rng = np.random.default_rng(71)
        image = rng.integers(0, 256, (192, 256, 3), dtype=np.uint8)
        return {int(i): image.copy() for i in indices}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="eval.local.yaml")
    parser.add_argument(
        "--metrics",
        default="lpips,dists,niqe,musiq,clipiqa,dover,fastvqa,fastervqa,ewarp",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite an existing validation report")
    config = Config(args.config)
    runtime = config.runtime()
    paths = config.artifact_paths()
    protocol = config.protocol()
    names = args.metrics.split(",")
    if any(n not in METRICS or n in ["psnr", "ssim"] for n in names):
        raise SystemExit("Choose learned/video/flow metrics")
    artifacts = check_artifacts(names, paths)
    preflight_backends(names, runtime)
    rng = np.random.default_rng(71)
    image = rng.integers(0, 256, (192, 256, 3), dtype=np.uint8)
    results = {}
    for name in names:
        try:
            with network_disabled():
                backend = create_backend(name, paths, runtime, protocol)
                if METRICS[name].kind == "video":
                    first, detail = backend.video(TinySequence(), 17)
                    second, _ = backend.video(TinySequence(), 17)
                    assert abs(first - second) < 1e-4, (
                        "Repeated seed is not stable within tolerance"
                    )
                elif name == "ewarp":
                    first, detail = backend.pair(image, image)
                    assert np.isfinite(first)
                else:
                    first = backend.image(
                        image, image if METRICS[name].reference else None
                    )
                    detail = {}
                    if name in ["lpips", "dists"]:
                        assert abs(first) < 1e-4, (
                            "Identity perceptual distance is not near zero"
                        )
                assert np.isfinite(first)
                results[name] = {"status": "passed", "value": first, "details": detail}
                del backend
        except Exception as exc:
            results[name] = {"status": "failed", "error": str(exc)}
    report = {
        "checks": results,
        "artifacts": {
            k: {**v, "sha256": fingerprint(paths[k])} for k, v in artifacts.items()
        },
        "meaning": "Identity/finiteness/repeatability only; not paper-table reproduction or a perceptual validity study.",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    if any(v["status"] != "passed" for v in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
