import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import platform
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__
from .backends import create_backend, preflight_backends, source_pins
from .config import Config, digest
from .errors import EvaluationError, NotApplicable
from .media import Sequence, alignment, crop_common, pair_paths
from .metrics.classical import psnr, ssim
from .offline import catalog, check_artifacts, fingerprint, network_disabled
from .registry import METRICS, select_metrics


def build_jobs(config, models, datasets, metrics=None, protocol=None):
    jobs = []
    for model in models:
        if model not in config.data["models"]:
            raise EvaluationError(f"Unregistered model: {model}")
        for dataset in datasets:
            if dataset not in config.data["datasets"]:
                raise EvaluationError(f"Unregistered dataset: {dataset}")
            m, d = config.data["models"][model], config.data["datasets"][dataset]
            variables = {"model": model, "dataset": dataset}
            pred = config.resolve(m["pred_root"], **variables)
            gt = config.resolve(d.get("gt_root"), **variables)
            mapping_value = m.get("mappings", {}).get(dataset, d.get("mapping_file"))
            mapping = config.resolve(mapping_value, **variables)
            pairs = pair_paths(pred, gt, mapping)
            has_ref = all(row[2] is not None for row in pairs)
            selected = select_metrics(
                metrics if metrics is not None else config.data.get("metrics", "auto"),
                has_ref,
            )
            clips = []
            policy = config.protocol(protocol)
            for clip_id, pred_path, ref_path in pairs:
                p = Sequence(pred_path, d.get("fps"))
                r = Sequence(ref_path, d.get("fps")) if ref_path else None
                count, size, changes = alignment(p, r, policy)
                clips.append(
                    {
                        "id": clip_id,
                        "pred": p,
                        "reference": r,
                        "aligned_frames": count,
                        "size": size,
                        "transformations": changes,
                    }
                )
            jobs.append(
                {
                    "model": model,
                    "dataset": dataset,
                    "metrics": selected,
                    "protocol": policy,
                    "clips": clips,
                }
            )
    return jobs


def _json_safe(value):
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return "+inf" if value > 0 else "-inf" if value < 0 else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def save_json(path, value):
    Path(path).write_text(
        json.dumps(_json_safe(value), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def environment():
    versions = {}
    for name in (
        "numpy",
        "scipy",
        "Pillow",
        "PyYAML",
        "av",
        "torch",
        "torchvision",
        "pyiqa",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
    }


def _clip_metric(clip, name, backend, protocol, writer, job, seed):
    p, r = clip["pred"], clip["reference"]
    kind = METRICS[name].kind
    if kind == "video":
        value, detail = backend.video(p, seed)
        if not math.isfinite(value):
            raise EvaluationError(f"Non-finite {name} output")
        writer.writerow(
            [
                job["model"],
                job["dataset"],
                clip["id"],
                name,
                "video",
                value,
                "ok",
                json.dumps(detail),
            ]
        )
        return {
            "value": value,
            "status": "ok",
            "count": 1,
            "expected": 1,
            "details": detail,
        }
    if kind == "pair" and p.info.frames < 2:
        raise NotApplicable("At least two frames are required")
    values, excluded = [], []
    reference_frames = iter(r) if r and METRICS[name].reference else None
    previous = None
    expected = (
        p.info.frames - 1
        if kind == "pair"
        else clip["aligned_frames"]
        if METRICS[name].reference
        else p.info.frames
    )
    for index, frame in enumerate(p):
        if kind == "pair":
            if previous is None:
                previous = frame
                continue
            try:
                value, detail = backend.pair(previous, frame)
            except NotApplicable as exc:
                excluded.append({"pair": index - 1, "reason": str(exc)})
                writer.writerow(
                    [
                        job["model"],
                        job["dataset"],
                        clip["id"],
                        name,
                        index - 1,
                        "",
                        "not_applicable",
                        str(exc),
                    ]
                )
                previous = frame
                continue
            previous = frame
            step = index - 1
        else:
            if METRICS[name].reference and index >= clip["aligned_frames"]:
                break
            gt = next(reference_frames) if reference_frames is not None else None
            if gt is not None:
                frame, gt = [
                    crop_common(a, clip["size"], protocol["alignment"])
                    for a in (frame, gt)
                ]
            if name in ("psnr", "ssim"):
                color = (
                    protocol["color_space"]
                    if name == "psnr"
                    else protocol["ssim_color_space"]
                )
                if name == "psnr":
                    value = psnr(
                        frame, gt, color, protocol["crop_border"], protocol["psnr_eps"]
                    )
                else:
                    value = ssim(
                        frame,
                        gt,
                        color,
                        protocol["crop_border"],
                        protocol["ssim_clip_negative_cs"],
                        protocol["ssim_round_values"],
                    )
            else:
                # RGB inputs; classical Y/crop options are never applied to perceptual/NR metrics.
                value = backend.image(frame, gt)
            detail, step = {}, index
        if not math.isfinite(value) and not (name == "psnr" and value == float("inf")):
            raise EvaluationError(f"Non-finite {name} output at frame/pair {step}")
        values.append(value)
        writer.writerow(
            [
                job["model"],
                job["dataset"],
                clip["id"],
                name,
                step,
                value,
                "ok",
                json.dumps(detail),
            ]
        )
    if not values:
        raise NotApplicable("No valid frame pairs; no zero score is substituted")
    status = "ok" if len(values) == expected else "partial"
    value = float("inf") if float("inf") in values else math.fsum(values) / len(values)
    result = {
        "value": value,
        "status": status,
        "count": len(values),
        "expected": expected,
    }
    if excluded:
        result["excluded"] = excluded
    if name == "ewarp":
        result["scaled_1000"] = value * 1000
    return result


def evaluate(
    config,
    models,
    datasets,
    *,
    metrics=None,
    protocol=None,
    device=None,
    output=None,
    backend_factory=create_backend,
    progress=None,
):
    config = config if isinstance(config, Config) else Config(config)
    if isinstance(models, str):
        models = models.split(",")
    if isinstance(datasets, str):
        datasets = datasets.split(",")
    jobs = build_jobs(config, models, datasets, metrics, protocol)
    paths = config.artifact_paths()
    names = list(dict.fromkeys(n for job in jobs for n in job["metrics"]))
    if progress:
        progress(f"Preflight: {len(jobs)} job(s), {len(names)} metric(s); no downloads")
    artifacts = check_artifacts(names, paths)
    runtime = config.runtime()
    if device:
        runtime["device"] = device
    if backend_factory is create_backend:
        preflight_backends(names, runtime)
    for key, record in artifacts.items():
        record["sha256"] = fingerprint(paths[key])
        known = catalog()[key]
        short_hash = re.search(r"-([0-9a-f]{8})\.", known["filenames"][0])
        expected = known.get("sha256") or (short_hash.group(1) if short_hash else None)
        if expected and not record["sha256"].startswith(expected):
            raise EvaluationError(
                f"Artifact checksum mismatch: {key}; expected prefix {expected}, got {record['sha256']}"
            )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = (
        Path(output).resolve()
        if output
        else config.resolve(config.data.get("output_root", "runs")) / timestamp
    )
    destination.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": 1,
        "tool_version": __version__,
        "created_utc": timestamp,
        "environment": environment(),
        "runtime": runtime,
        "artifacts": artifacts,
        "source_pins": source_pins(),
        "config": config.data,
        "jobs": [],
        "validation": {
            "source_audit": "see docs/AUDIT.md",
            "real_weight_numerical_validation": "pending-user-local-weights",
        },
    }
    failed = False
    with (destination / "details.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "dataset",
                "clip_id",
                "metric",
                "frame_or_pair",
                "value",
                "status",
                "details",
            ]
        )
        for job in jobs:
            records = {
                c["id"]: {
                    "clip_id": c["id"],
                    "prediction": c["pred"].describe(),
                    "reference": c["reference"].describe() if c["reference"] else None,
                    "transformations": c["transformations"],
                    "metrics": {},
                }
                for c in job["clips"]
            }
            item = {
                "model": job["model"],
                "dataset": job["dataset"],
                "protocol": job["protocol"],
                "clips": list(records.values()),
                "summary": {},
            }
            report["jobs"].append(item)
            for name in job["metrics"]:
                if progress:
                    progress(
                        f"[{job['model']}/{job['dataset']}] {name}: {len(job['clips'])} clip(s)"
                    )
                backend = None
                try:
                    with network_disabled():
                        backend = backend_factory(name, paths, runtime, job["protocol"])
                except Exception as exc:
                    failed = True
                    for record in records.values():
                        record["metrics"][name] = {
                            "status": "error",
                            "value": None,
                            "error": str(exc),
                        }
                        writer.writerow(
                            [
                                job["model"],
                                job["dataset"],
                                record["clip_id"],
                                name,
                                "initialization",
                                "",
                                "error",
                                str(exc),
                            ]
                        )
                else:
                    for clip in job["clips"]:
                        seed = (
                            job["protocol"]["seed"]
                            + int(
                                hashlib.sha256(clip["id"].encode()).hexdigest()[:8], 16
                            )
                        ) % (2**32)
                        try:
                            with network_disabled():
                                result = _clip_metric(
                                    clip,
                                    name,
                                    backend,
                                    job["protocol"],
                                    writer,
                                    job,
                                    seed,
                                )
                            if result["status"] != "ok":
                                failed = True
                        except NotApplicable as exc:
                            result = {
                                "status": "not_applicable",
                                "value": None,
                                "reason": str(exc),
                                "count": 0,
                            }
                            writer.writerow(
                                [
                                    job["model"],
                                    job["dataset"],
                                    clip["id"],
                                    name,
                                    "clip",
                                    "",
                                    "not_applicable",
                                    str(exc),
                                ]
                            )
                        except Exception as exc:
                            failed = True
                            result = {
                                "status": "error",
                                "value": None,
                                "error": str(exc),
                            }
                            writer.writerow(
                                [
                                    job["model"],
                                    job["dataset"],
                                    clip["id"],
                                    name,
                                    "clip",
                                    "",
                                    "error",
                                    str(exc),
                                ]
                            )
                        records[clip["id"]]["metrics"][name] = result
                finally:
                    del backend
                    gc.collect()
                    if "torch" in __import__("sys").modules:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                valid = [
                    c for c in item["clips"] if c["metrics"][name]["status"] == "ok"
                ]
                all_na = all(
                    c["metrics"][name]["status"] == "not_applicable"
                    for c in item["clips"]
                )
                complete = len(valid) == len(item["clips"])
                if not complete and not all_na:
                    failed = True
                numbers = [c["metrics"][name]["value"] for c in valid]
                value = (
                    (
                        float("inf")
                        if float("inf") in numbers
                        else math.fsum(numbers) / len(numbers)
                    )
                    if complete
                    else None
                )
                signature = {
                    "protocol": job["protocol"],
                    "versions": report["environment"]["packages"],
                    "tool_version": __version__,
                    "source_commits": {
                        k: v["commit"] for k, v in report["source_pins"].items()
                    },
                    "weights": {
                        a: artifacts[a]["sha256"] for a in METRICS[name].artifacts
                    },
                    "clips": [
                        (
                            c["clip_id"],
                            c["prediction"]["frames"],
                            c["prediction"]["height"],
                            c["prediction"]["width"],
                            c["prediction"]["fps"],
                            c["reference"]["path"] if c["reference"] else None,
                            c["metrics"][name].get("expected"),
                        )
                        for c in item["clips"]
                    ],
                }
                item["summary"][name] = {
                    "value": value,
                    "status": "ok"
                    if complete
                    else "not_applicable"
                    if all_na
                    else "incomplete",
                    "valid_clips": len(valid),
                    "total_clips": len(item["clips"]),
                    "comparison_group": digest(signature),
                    "lower_better": METRICS[name].lower_better,
                }
                if name == "ewarp" and complete:
                    item["summary"][name]["scaled_1000"] = value * 1000
                save_json(destination / "report.json", report)
    report["status"] = "incomplete" if failed else "complete"
    save_json(destination / "report.json", report)
    with (destination / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model",
                "dataset",
                "protocol",
                "metric",
                "value",
                "status",
                "valid_clips",
                "total_clips",
                "comparison_group",
            ]
        )
        lines = [
            "# Evaluation results",
            "",
            "Compare rows only within the same dataset, metric and comparison_group. No automatic ranking.",
            "",
            "| Model | Dataset | Metric | Value | Status | Coverage | Protocol / group |",
            "|---|---|---|---:|---|---|---|",
        ]
        for job in report["jobs"]:
            for metric, row in job["summary"].items():
                writer.writerow(
                    [
                        job["model"],
                        job["dataset"],
                        job["protocol"]["id"],
                        metric,
                        row["value"],
                        row["status"],
                        row["valid_clips"],
                        row["total_clips"],
                        row["comparison_group"],
                    ]
                )
                display = f"{row['value']:.8g}" if row["value"] is not None else "—"
                lines.append(
                    f"| {job['model']} | {job['dataset']} | {metric} | {display} | {row['status']} | {row['valid_clips']}/{row['total_clips']} | {job['protocol']['id']} / {row['comparison_group']} |"
                )
    (destination / "comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return report, destination
