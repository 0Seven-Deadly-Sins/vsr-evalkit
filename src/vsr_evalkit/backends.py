import importlib
import json
import random
import subprocess
import sys
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from .errors import EvaluationError
from .offline import LocalResolver, network_disabled


def source_pins():
    return json.loads(
        files("vsr_evalkit.resources")
        .joinpath("sources.json")
        .read_text(encoding="utf-8")
    )


def source_root(runtime, key):
    root = runtime.get("sources", {}).get(key)
    if not root:
        raise EvaluationError(
            f"Set sources.{key} to the pinned upstream checkout; see docs/WEIGHTS.md"
        )
    root = Path(root)
    expected = source_pins()[key]["commit"]
    try:
        found = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise EvaluationError(f"Not a Git checkout: {root}") from exc
    if found != expected:
        raise EvaluationError(
            f"sources.{key} commit mismatch: expected {expected}, got {found}"
        )
    result = subprocess.run(
        ["git", "-C", str(root), "diff", "--quiet", "HEAD", "--"], check=False
    )
    if result.returncode:
        raise EvaluationError(f"Upstream tracked files are modified: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def torch_runtime(runtime):
    try:
        import torch
    except ImportError as exc:
        raise EvaluationError(
            "Install the deep extra / appropriate PyTorch build first"
        ) from exc
    device = runtime["device"]
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise EvaluationError(
            "CUDA requested but unavailable. Use --device cpu for CPU evaluation"
        )
    torch.set_num_threads(int(runtime.get("threads", 4)))
    return torch, device


def preflight_backends(names, runtime):
    deep = set(names) - {"psnr", "ssim"}
    if not deep:
        return
    torch_runtime(runtime)
    if deep & {"lpips", "dists", "niqe", "musiq", "clipiqa"}:
        try:
            found = version("pyiqa")
        except importlib.metadata.PackageNotFoundError as exc:
            raise EvaluationError(
                "Install the deep extra (pyiqa==0.1.15.post2)"
            ) from exc
        if found != "0.1.15.post2":
            raise EvaluationError(f"Expected pyiqa==0.1.15.post2, found {found}")
    for key, needed in [
        ("raft", {"ewarp"}),
        ("dover", {"dover"}),
        ("fastvqa", {"fastvqa", "fastervqa"}),
    ]:
        if deep & needed:
            source_root(runtime, key)


class PyIQABackend:
    def __init__(self, name, paths, runtime):
        self.torch, self.device = torch_runtime(runtime)
        self.name = name
        self.resolver = LocalResolver(paths)
        with network_disabled():
            try:
                pyiqa = importlib.import_module("pyiqa")
                # PyIQA lazily imports architectures. Import the selected architecture
                # before scanning aliases, so CLIP/torchvision download helpers are resolved locally too.
                module = {
                    "lpips": "lpips_arch",
                    "dists": "dists_arch",
                    "niqe": "niqe_arch",
                    "musiq": "musiq_arch",
                    "clipiqa": "clipiqa_arch",
                }[name]
                importlib.import_module("pyiqa.archs." + module)
            except ImportError as exc:
                raise EvaluationError(
                    "Install vsr-evalkit[deep] to use PyIQA metrics"
                ) from exc
        if version("pyiqa") != "0.1.15.post2":
            raise EvaluationError("This adapter is pinned to pyiqa==0.1.15.post2")
        options = {"device": self.device}
        artifact = {
            "lpips": "lpips_alex",
            "dists": "dists",
            "niqe": "niqe_params",
            "musiq": "musiq_koniq",
        }.get(name)
        if artifact:
            options["pretrained_model_path"] = str(paths[artifact])
        if name == "lpips":
            options.update(net="alex", version="0.1")
        if name == "musiq":
            options.update(pretrained="koniq10k")
        if name == "niqe":
            options.update(test_y_channel=True, color_space="yiq", crop_border=0)
        with self.resolver.activate():
            self.model = pyiqa.create_metric(name, **options).eval()

    def image(self, pred, reference=None):
        if self.name == "niqe" and min(pred.shape[:2]) < 192:
            raise EvaluationError(
                "NIQE requires >=192 pixels on both dimensions in this protocol"
            )

        def tensor(frame):
            return (
                self.torch.from_numpy(np.ascontiguousarray(frame))
                .permute(2, 0, 1)[None]
                .float()
                .to(self.device)
                / 255
            )

        with self.resolver.activate(), self.torch.inference_mode():
            args = [tensor(pred)]
            if reference is not None:
                args.append(tensor(reference))
            return float(self.model(*args).mean().item())


def fuse_dover(scores):
    # Bind names before fusion; model list order follows the input dictionary order.
    if not all(np.isfinite(scores[key]) for key in ("aesthetic", "technical")):
        raise EvaluationError("DOVER produced non-finite raw branch scores")
    a = (scores["aesthetic"] + 0.08285) / 0.03774
    t = (scores["technical"] - 0.1107) / 0.07355
    sigmoid = lambda x: float(1 / (1 + np.exp(-np.clip(x, -700, 700))))
    return {
        "aesthetic": sigmoid(a),
        "technical": sigmoid(t),
        "overall": sigmoid(0.6104 * t + 0.3896 * a),
    }


class VQABackend:
    def __init__(self, name, paths, runtime):
        self.torch, self.device = torch_runtime(runtime)
        self.name = name
        self.paths = paths
        key = "dover" if name == "dover" else "fastvqa"
        self.root = source_root(runtime, key)
        with network_disabled():
            if name == "dover":
                self.datasets = importlib.import_module("dover.datasets")
                model_class = importlib.import_module("dover.models").DOVER
                opt_path = self.root / "dover.yml"
            else:
                self.datasets = importlib.import_module("fastvqa.datasets")
                model_class = importlib.import_module(
                    "fastvqa.models"
                ).DiViDeAddEvaluator
                opt_path = (
                    self.root
                    / "options/fast"
                    / ("fast-b.yml" if name == "fastvqa" else "f3dvqa-b.yml")
                )
            self.opt = yaml.safe_load(opt_path.read_text(encoding="utf-8"))
            self.model = model_class(**self.opt["model"]["args"]).to(self.device).eval()
            state = self.torch.load(paths[name], map_location="cpu", weights_only=False)
            self.model.load_state_dict(
                state if name == "dover" else state["state_dict"], strict=True
            )

    def video(self, sequence, seed):
        torch = self.torch
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        section = "val-l1080p" if self.name == "dover" else "val-kv1k"
        data_opt = self.opt["data"][section]["args"]
        sample_types = data_opt["sample_types"]
        views, indices = {}, {}
        for key, option in sample_types.items():
            if self.name == "dover":
                if "t_frag" not in option:
                    sampler = self.datasets.UnifiedFrameSampler(
                        option["clip_len"],
                        option["num_clips"],
                        option["frame_interval"],
                    )
                else:
                    sampler = self.datasets.UnifiedFrameSampler(
                        option["clip_len"] // option["t_frag"],
                        option["t_frag"],
                        option["frame_interval"],
                        option["num_clips"],
                    )
            elif data_opt.get("t_frag", 1) > 1:
                sampler = self.datasets.FragmentSampleFrames(
                    fsize_t=option["clip_len"] // option.get("t_frag", 1),
                    fragments_t=option.get("t_frag", 1),
                    num_clips=option.get("num_clips", 1),
                )
            else:
                sampler = self.datasets.SampleFrames(
                    clip_len=option["clip_len"], num_clips=option["num_clips"]
                )
            indices[key] = [int(i) for i in sampler(sequence.info.frames)]
            # One view at a time: retain sampled frames only, release original-resolution stacks after sampling.
            selected = sequence.read_indices(indices[key])
            clip = torch.stack(
                [torch.from_numpy(selected[i]) for i in indices[key]], 0
            ).permute(3, 0, 1, 2)
            if self.name == "dover":
                sampled = self.datasets.get_single_view(clip, key, **option)
            else:
                sampled = self.datasets.get_spatial_fragments(clip, **option)
            del clip, selected
            mean = torch.tensor([123.675, 116.28, 103.53])
            std = torch.tensor([58.395, 57.12, 57.375])
            sampled = ((sampled.permute(1, 2, 3, 0) - mean) / std).permute(3, 0, 1, 2)
            clips = option.get("num_clips", 1)
            views[key] = (
                sampled.reshape(sampled.shape[0], clips, -1, *sampled.shape[2:])
                .transpose(0, 1)
                .to(self.device)
            )
        with network_disabled(), torch.inference_mode():
            result = (
                self.model(views, reduce_scores=False)
                if self.name == "dover"
                else self.model(views)
            )
        if self.name == "dover":
            if len(result) != len(views):
                raise EvaluationError("DOVER branch count mismatch")
            raw = {key: float(item.mean().item()) for key, item in zip(views, result)}
            fused = fuse_dover(raw)
            return fused.pop("overall"), {
                "branches": fused,
                "raw_branches": raw,
                "sampled_indices": indices,
                "seed": seed,
            }
        raw = float(result.mean().item())
        if not np.isfinite(raw):
            raise EvaluationError(f"{self.name} produced a non-finite raw score")
        mean, std = {
            "fastvqa": (-0.110198185, 0.04178565),
            "fastervqa": (0.14759505, 0.03613452),
        }[self.name]
        score = float(1 / (1 + np.exp(-np.clip((raw - mean) / std, -700, 700))))
        return score, {"raw": raw, "sampled_indices": indices, "seed": seed}


class RAFTBackend:
    def __init__(self, paths, runtime, protocol):
        self.torch, self.device = torch_runtime(runtime)
        self.protocol = protocol
        root = source_root(runtime, "raft")
        core = str(root / "core")
        if core not in sys.path:
            sys.path.insert(0, core)

        class Arguments(SimpleNamespace):
            def __contains__(self, key):
                return key in self.__dict__

        with network_disabled():
            model_class = importlib.import_module("raft").RAFT
            self.model = model_class(
                Arguments(
                    small=False, mixed_precision=False, alternate_corr=False, dropout=0
                )
            )
            state = self.torch.load(
                paths["raft_things"], map_location="cpu", weights_only=False
            )
            self.model.load_state_dict(
                {k.removeprefix("module."): v for k, v in state.items()}, strict=True
            )
            self.model = self.model.to(self.device).eval()

    def pair(self, previous, current):
        import torch.nn.functional as F
        from .metrics.temporal import masked_warp_mse, resize_flow

        torch = self.torch

        def tensor(image):
            return (
                torch.from_numpy(np.ascontiguousarray(image))
                .permute(2, 0, 1)[None]
                .float()
                .to(self.device)
            )

        first, second = tensor(previous), tensor(current)
        h, w = first.shape[-2:]
        fh, fw = h, w
        limit = self.protocol["flow_max_long_edge"]
        if limit and max(h, w) > limit:
            factor = limit / max(h, w)
            fh, fw = max(1, round(h * factor)), max(1, round(w * factor))
        if min(fh, fw) < 128:
            raise EvaluationError(
                "RAFT requires >=128 pixels on both flow-estimation dimensions"
            )
        a, b = first, second
        if (fh, fw) != (h, w):
            a, b = [
                F.interpolate(x, size=(fh, fw), mode="bilinear", align_corners=False)
                for x in (a, b)
            ]
        ph, pw = (-fh) % 8, (-fw) % 8
        pad = (pw // 2, pw - pw // 2, ph // 2, ph - ph // 2)
        a, b = [F.pad(x, pad, mode="replicate") for x in (a, b)]
        with network_disabled(), torch.inference_mode():
            # Original RAFT internally divides by 255. Inputs here stay in [0,255].
            forward = self.model(
                a, b, iters=self.protocol["raft_iterations"], test_mode=True
            )[1]
            backward = self.model(
                b, a, iters=self.protocol["raft_iterations"], test_mode=True
            )[1]
            forward, backward = [
                f[..., pad[2] : pad[2] + fh, pad[0] : pad[0] + fw]
                for f in (forward, backward)
            ]
            if (fh, fw) != (h, w):
                forward, backward = [resize_flow(f, h, w) for f in (forward, backward)]
            value, detail = masked_warp_mse(
                first / 255,
                second / 255,
                forward,
                backward,
                self.protocol["occlusion_relative"],
                self.protocol["occlusion_absolute"],
            )
        return value, {**detail, "flow_shape": [fh, fw], "padding": pad}


def create_backend(name, paths, runtime, protocol):
    if name in ("psnr", "ssim"):
        return None
    if name == "ewarp":
        return RAFTBackend(paths, runtime, protocol)
    if name in ("dover", "fastvqa", "fastervqa"):
        return VQABackend(name, paths, runtime)
    return PyIQABackend(name, paths, runtime)
