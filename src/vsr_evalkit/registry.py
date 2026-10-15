from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSpec:
    name: str
    kind: str
    reference: bool
    lower_better: bool
    artifacts: tuple = ()
    backend: str = "classical"


METRICS = {
    spec.name: spec
    for spec in [
        MetricSpec("psnr", "image", True, False),
        MetricSpec("ssim", "image", True, False),
        MetricSpec("lpips", "image", True, True, ("alexnet", "lpips_alex"), "pyiqa"),
        MetricSpec("dists", "image", True, True, ("vgg16", "dists"), "pyiqa"),
        MetricSpec("niqe", "image", False, True, ("niqe_params",), "pyiqa"),
        MetricSpec("musiq", "image", False, False, ("musiq_koniq",), "pyiqa"),
        MetricSpec("clipiqa", "image", False, False, ("clip_rn50",), "pyiqa"),
        MetricSpec("dover", "video", False, False, ("dover",), "dover"),
        MetricSpec("fastvqa", "video", False, False, ("fastvqa",), "fastvqa"),
        MetricSpec("fastervqa", "video", False, False, ("fastervqa",), "fastvqa"),
        MetricSpec("ewarp", "pair", False, True, ("raft_things",), "raft"),
    ]
}


def select_metrics(names, has_reference):
    if names == "auto" or names is None:
        return [n for n, s in METRICS.items() if has_reference or not s.reference]
    if isinstance(names, str):
        names = names.split(",")
    selected = list(dict.fromkeys(n.strip() for n in names))
    unknown = set(selected) - METRICS.keys()
    if unknown:
        raise ValueError(f"Unknown metrics: {sorted(unknown)}")
    if not selected:
        raise ValueError("At least one metric must be selected")
    if not has_reference and any(METRICS[n].reference for n in selected):
        raise ValueError(
            "Explicit full-reference metrics require dataset.gt_root or mapping-file GT"
        )
    return selected
