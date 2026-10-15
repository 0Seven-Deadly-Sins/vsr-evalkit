"""Exercise fixed official samplers/views with mock heads, without any pretrained weights."""

import argparse
import ast
from functools import lru_cache
import json
from pathlib import Path
import random
from types import SimpleNamespace
import urllib.request

import numpy as np
import torch
import torchvision
import yaml

from vsr_evalkit.backends import VQABackend, source_pins
from vsr_evalkit.media import MediaInfo


class Sequence:
    info = MediaInfo(5, 240, 320, None, None, "synthetic", {})

    def read_indices(self, indices):
        return {int(i): np.full((240, 320, 3), 30 + int(i), np.uint8) for i in indices}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    pins = source_pins()
    checks = {}
    for name in ["dover", "fastvqa", "fastervqa"]:
        key = "dover" if name == "dover" else "fastvqa"
        source = (
            "dover/datasets/dover_datasets.py"
            if name == "dover"
            else "fastvqa/datasets/fusion_datasets.py"
        )
        option = (
            "dover.yml"
            if name == "dover"
            else "options/fast/"
            + ("fast-b.yml" if name == "fastvqa" else "f3dvqa-b.yml")
        )
        code = args.cache / (key + ".py")
        config = args.cache / (name + ".yml")
        if args.fetch:
            base = f"{pins[key]['url'].replace('github.com', 'raw.githubusercontent.com')}/{pins[key]['commit']}/"
            code.write_bytes(urllib.request.urlopen(base + source, timeout=30).read())
            config.write_bytes(urllib.request.urlopen(base + option, timeout=30).read())
        namespace = {
            "np": np,
            "torch": torch,
            "torchvision": torchvision,
            "random": random,
            "lru_cache": lru_cache,
        }
        wanted = (
            {
                "get_spatial_fragments",
                "get_resize_function",
                "get_resized_video",
                "get_single_view",
                "UnifiedFrameSampler",
            }
            if name == "dover"
            else {"get_spatial_fragments", "SampleFrames", "FragmentSampleFrames"}
        )
        nodes = [
            node
            for node in ast.parse(code.read_text(encoding="utf-8")).body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted
        ]
        assert {node.name for node in nodes} == wanted
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), str(code), "exec"),
            namespace,
        )
        backend = VQABackend.__new__(VQABackend)
        backend.torch = torch
        backend.device = "cpu"
        backend.name = name
        backend.datasets = SimpleNamespace(**namespace)
        backend.opt = yaml.safe_load(config.read_text())
        shapes = {}

        def model(views, **kwargs):
            shapes.update({k: list(v.shape) for k, v in views.items()})
            assert all(
                v.shape[-2:] == (224, 224) and torch.isfinite(v).all()
                for v in views.values()
            )
            if name == "dover":
                return [
                    torch.tensor(
                        -0.08285 if key == "aesthetic" else 0.1107, dtype=torch.float64
                    )
                    for key in views
                ]
            return torch.tensor(
                -0.110198185 if name == "fastvqa" else 0.14759505, dtype=torch.float64
            )

        backend.model = model
        torch.set_num_threads(4)
        first, detail = backend.video(Sequence(), 123)
        second, again = backend.video(Sequence(), 123)
        assert abs(first - 0.5) < 1e-12 and first == second
        assert detail["sampled_indices"] == again["sampled_indices"]
        checks[name] = {
            "status": "passed",
            "source_commit": pins[key]["commit"],
            "views": shapes,
            "indices": detail["sampled_indices"],
            "scope": "Actual official sampling/view functions + mock score head; not pretrained model numerical parity",
        }
    text = json.dumps(checks, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: {"status": v["status"], "views": v["views"]}
                for k, v in checks.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
