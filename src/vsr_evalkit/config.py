import copy
import hashlib
import json
import os
from pathlib import Path
from importlib.resources import files

import yaml

from .errors import EvaluationError


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:16]


def merge(left, right):
    result = copy.deepcopy(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class Config:
    def __init__(self, path="eval.local.yaml"):
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise EvaluationError(
                f"Config not found: {self.path}. Copy examples/eval.yaml to eval.local.yaml first."
            )
        self.data = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        if not isinstance(self.data, dict) or self.data.get("version") != 1:
            raise EvaluationError("Config must be a mapping with version: 1")
        for key in ("models", "datasets"):
            if not isinstance(self.data.get(key), dict) or not self.data[key]:
                raise EvaluationError(f"Config requires nonempty {key}")
        self.presets = yaml.safe_load(
            files("vsr_evalkit.resources")
            .joinpath("protocols.yaml")
            .read_text(encoding="utf-8")
        )

    def resolve(self, value, **variables):
        if value is None:
            return None
        expanded = os.path.expandvars(
            os.path.expanduser(str(value).format(**variables))
        )
        path = Path(expanded)
        return (
            path.resolve()
            if path.is_absolute()
            else (self.path.parent / path).resolve()
        )

    def protocol(self, override=None):
        raw = (
            self.data.get("protocol", "standard-rgb-v1")
            if override is None
            else override
        )
        options = {} if isinstance(raw, str) else dict(raw)
        name = raw if isinstance(raw, str) else options.pop("preset", "standard-rgb-v1")
        if name not in self.presets:
            raise EvaluationError(f"Unknown protocol preset: {name}")
        result = merge(self.presets[name], options)
        result["preset"] = name
        if result["alignment"] not in ("strict", "top-left-common", "center-common"):
            raise EvaluationError("Unknown alignment policy")
        if result["temporal_alignment"] not in ("strict", "truncate-common"):
            raise EvaluationError("Unknown temporal alignment policy")
        if result["frame_pairing"] not in ("identity", "ordinal"):
            raise EvaluationError("frame_pairing must be identity or ordinal")
        if result["color_space"] not in ("rgb", "bt601-y", "yiq-y") or result[
            "ssim_color_space"
        ] not in ("rgb", "bt601-y", "yiq-y"):
            raise EvaluationError("color_space must be rgb or bt601-y")
        if (
            result["aggregation"] != "video-macro"
            or result["frame_sampling"] != "all"
            or result["vqa_sampling"] != "upstream-fixed"
        ):
            raise EvaluationError(
                "Only video-macro aggregation, all image frames and upstream-fixed VQA sampling are implemented"
            )
        if (
            result["psnr_eps"] < 0
            or result["raft_iterations"] < 1
            or result["occlusion_relative"] < 0
            or result["occlusion_absolute"] < 0
        ):
            raise EvaluationError(
                "Invalid metric epsilon, iterations or mask threshold"
            )
        if not isinstance(result["crop_border"], int) or result["crop_border"] < 0:
            raise EvaluationError("crop_border must be a nonnegative integer")
        if (
            result["flow_max_long_edge"] is not None
            and result["flow_max_long_edge"] < 128
        ):
            raise EvaluationError("flow_max_long_edge must be null or >= 128")
        result["id"] = f"{name}-{digest(result)}"
        return result

    def artifact_paths(self):
        return {
            key: self.resolve(value)
            for key, value in self.data.get("weights", {}).items()
        }

    def runtime(self):
        settings = merge(
            {"device": "cuda", "seed": 0, "threads": 4}, self.data.get("runtime", {})
        )
        settings["sources"] = {
            k: str(self.resolve(v)) for k, v in self.data.get("sources", {}).items()
        }
        return settings
