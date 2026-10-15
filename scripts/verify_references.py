"""Differential checks against actual BasicSR/PyIQA classical functions, no weights.

--fetch retrieves ONLY four pinned Python source files into the chosen cache.
PyIQA is inspected from its installed distribution without importing its package.
"""

import argparse
import ast
import importlib.util
import importlib.metadata
import json
from pathlib import Path
import urllib.request

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from vsr_evalkit.backends import source_pins
from vsr_evalkit.metrics.classical import psnr, ssim


def functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    selected = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            node.decorator_list = []
            selected.append(node)
    if {n.name for n in selected} != set(names):
        raise RuntimeError(f"Missing expected functions in {path}")
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"),
        namespace,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    commit = source_pins()["basicsr"]["commit"]
    paths = {
        "psnr_ssim.py": "basicsr/metrics/psnr_ssim.py",
        "metric_util.py": "basicsr/metrics/metric_util.py",
        "color_util.py": "basicsr/utils/color_util.py",
    }
    if args.fetch:
        for name, path in paths.items():
            url = (
                f"https://raw.githubusercontent.com/XPixelGroup/BasicSR/{commit}/{path}"
            )
            (args.cache / name).write_bytes(
                urllib.request.urlopen(url, timeout=30).read()
            )
    namespace = {"np": np, "cv2": cv2, "torch": torch}
    functions(
        args.cache / "color_util.py",
        ["_convert_input_type_range", "_convert_output_type_range", "bgr2ycbcr"],
        namespace,
    )
    functions(
        args.cache / "metric_util.py", ["reorder_image", "to_y_channel"], namespace
    )
    functions(
        args.cache / "psnr_ssim.py",
        ["calculate_psnr", "calculate_ssim", "_ssim"],
        namespace,
    )
    spec = importlib.util.find_spec("pyiqa")
    if spec is None:
        raise RuntimeError(
            "Install pyiqa==0.1.15.post2 (code only) for this optional reference check"
        )
    package = Path(spec.origin).parent
    if importlib.metadata.version("pyiqa") != "0.1.15.post2":
        raise RuntimeError("Reference check requires exactly pyiqa==0.1.15.post2")
    py = {
        "np": np,
        "torch": torch,
        "F": F,
        "to_2tuple": lambda value: (value, value) if isinstance(value, int) else value,
    }
    functions(package / "matlab_utils/functions.py", ["fspecial", "filter2"], py)
    functions(package / "archs/ssim_arch.py", ["ssim"], py)
    functions(package / "archs/psnr_arch.py", ["psnr"], py)
    rng = np.random.default_rng(37)
    errors = {
        "basicsr_psnr": 0.0,
        "basicsr_ssim": 0.0,
        "pyiqa_psnr_eps0": 0.0,
        "pyiqa_ssim_matching_variant": 0.0,
    }
    cases = 0
    for shape in [(31, 47, 3), (81, 35, 3), (35, 81, 3)]:
        x = rng.integers(0, 256, shape, dtype=np.uint8)
        y = np.clip(x.astype(np.int16) + rng.integers(-25, 26, shape), 0, 255).astype(
            np.uint8
        )
        for color in ["rgb", "bt601-y"]:
            own_p, own_s = psnr(x, y, color), ssim(x, y, color)
            # BasicSR's luminance utility expects BGR (its normal OpenCV input convention).
            bp = namespace["calculate_psnr"](
                x[:, :, ::-1], y[:, :, ::-1], 0, test_y_channel=color != "rgb"
            )
            bs = namespace["calculate_ssim"](
                x[:, :, ::-1], y[:, :, ::-1], 0, test_y_channel=color != "rgb"
            )
            errors["basicsr_psnr"] = max(errors["basicsr_psnr"], abs(own_p - bp))
            errors["basicsr_ssim"] = max(errors["basicsr_ssim"], abs(own_s - bs))
            cases += 1
        a, b = [
            torch.from_numpy(v.copy()).permute(2, 0, 1)[None].double() / 255
            for v in (x, y)
        ]
        pp = float(py["psnr"](a, b, eps=0).item())
        ps = float(py["ssim"](a, b).item())
        errors["pyiqa_psnr_eps0"] = max(errors["pyiqa_psnr_eps0"], abs(psnr(x, y) - pp))
        errors["pyiqa_ssim_matching_variant"] = max(
            errors["pyiqa_ssim_matching_variant"],
            abs(ssim(x, y, clip_negative_cs=True) - ps),
        )
    # BasicSR's luminance utility uses float32; our arithmetic retains float64.
    assert errors["basicsr_psnr"] < 1e-6 and errors["basicsr_ssim"] < 1e-8, errors
    assert (
        errors["pyiqa_psnr_eps0"] < 1e-9
        and errors["pyiqa_ssim_matching_variant"] < 1e-6
    ), errors
    # Contrast-inverted texture exposes an intentional semantic difference, not roundoff.
    x = np.tile(np.array([0, 255], np.uint8), (32, 16))[..., None].repeat(3, axis=2)
    differences = {
        "ssim_unclipped": ssim(x, 255 - x),
        "ssim_clipped": ssim(x, 255 - x, clip_negative_cs=True),
        "identity_psnr_standard": "+inf",
        "identity_psnr_epsilon_1e8": psnr(x, x, eps=1e-8),
    }
    result = {
        "status": "passed",
        "cases": cases,
        "reference": "BasicSR " + commit,
        "pyiqa": "0.1.15.post2 code functions",
        "max_absolute_errors": errors,
        "intentional_differences": differences,
        "weights_downloaded": False,
    }
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
