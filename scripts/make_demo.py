"""Generate tiny synthetic paired clips and a CPU config; no external data or weights."""

import argparse
from pathlib import Path

import numpy as np
import yaml
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="demo-input")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    if root.exists():
        raise SystemExit(f"Refusing to overwrite {root}")
    rng = np.random.default_rng(7)
    for name, count, shape in [
        ("landscape", 3, (39, 61, 3)),
        ("portrait", 5, (61, 39, 3)),
    ]:
        for category in ["pred", "gt"]:
            (root / category / name).mkdir(parents=True)
        for index in range(count):
            truth = rng.integers(20, 230, shape, dtype=np.uint8)
            pred = (truth.astype(np.int16) + 3).clip(0, 255).astype(np.uint8)
            for category, array in [("gt", truth), ("pred", pred)]:
                Image.fromarray(array).save(root / category / name / f"{index:04}.png")
    data = {
        "version": 1,
        "models": {"demo": {"pred_root": "pred"}},
        "datasets": {"synthetic": {"gt_root": "gt"}},
        "metrics": ["psnr", "ssim"],
        "protocol": "standard-rgb-v1",
        "runtime": {"device": "cpu"},
        "output_root": "runs",
    }
    (root / "eval.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
    )
    print(root / "eval.yaml")


if __name__ == "__main__":
    main()
