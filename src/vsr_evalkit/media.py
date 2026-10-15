import csv
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .errors import EvaluationError

IMAGES = {".png", ".jpg", ".jpeg"}
VIDEOS = {".mp4", ".mkv", ".mov", ".avi"}


def natural_key(value):
    return [
        (0, int(v)) if v.isdigit() else (1, v.casefold())
        for v in re.split(r"(\d+)", str(value))
    ]


@dataclass
class MediaInfo:
    frames: int
    height: int
    width: int
    fps: float | None
    timestamps: list | None
    format: str
    color: dict


class Sequence:
    def __init__(self, path, fps=None):
        self.path = Path(path)
        self.fps = float(fps) if fps is not None else None
        self._info = None
        if not self.path.exists():
            raise EvaluationError(f"Input not found: {self.path}")
        self.image_paths = None
        if self.path.is_dir():
            self.image_paths = sorted(
                (p for p in self.path.iterdir() if p.suffix.lower() in IMAGES),
                key=natural_key,
            )
            if not self.image_paths:
                raise EvaluationError(f"No image frames: {self.path}")
        elif self.path.suffix.lower() in IMAGES:
            self.image_paths = [self.path]
        elif self.path.suffix.lower() not in VIDEOS:
            raise EvaluationError(f"Unsupported media: {self.path}")

    def _decode(self):
        if self.image_paths is not None:
            for path in self.image_paths:
                with Image.open(path) as image:
                    if image.mode not in ("RGB", "L", "P"):
                        raise EvaluationError(
                            f"Only 8-bit RGB/grayscale/palette images supported, got {image.mode}: {path}"
                        )
                    yield np.array(image.convert("RGB")), None, {}
            return
        try:
            import av

            with av.open(str(self.path)) as container:
                if len(container.streams.video) != 1:
                    raise EvaluationError(f"Expected one video stream: {self.path}")
                stream = container.streams.video[0]
                if int(stream.codec_context.color_trc) in (16, 18):
                    raise EvaluationError(
                        f"HDR transfer tag is outside the SDR8 protocol: {self.path}"
                    )
                color = {
                    "pixel_format": stream.codec_context.format.name
                    if stream.codec_context.format
                    else None,
                    "color_range": str(stream.codec_context.color_range),
                    "colorspace": str(stream.codec_context.colorspace),
                    "color_primaries": str(stream.codec_context.color_primaries),
                    "color_trc": str(stream.codec_context.color_trc),
                    "nominal_fps": float(stream.average_rate)
                    if stream.average_rate
                    else None,
                    "time_base": float(stream.time_base) if stream.time_base else 0.0,
                    "decoder": "PyAV/FFmpeg rgb24",
                }
                for frame in container.decode(stream):
                    if any(c.bits > 8 for c in frame.format.components):
                        raise EvaluationError(
                            f"High-bit-depth input is outside the SDR8 protocol: {self.path}"
                        )
                    # FFmpeg enum: SMPTE2084=16, ARIB_STD_B67=18.
                    if int(getattr(frame, "color_trc", 0)) in (16, 18):
                        raise EvaluationError(
                            f"HDR input is outside the SDR8 protocol: {self.path}"
                        )
                    pts = (
                        float(frame.pts * frame.time_base)
                        if frame.pts is not None
                        else None
                    )
                    yield frame.to_ndarray(format="rgb24"), pts, color
        except EvaluationError:
            raise
        except Exception as exc:
            raise EvaluationError(
                f"Video decode failed for {self.path}: {exc}"
            ) from exc

    @property
    def info(self):
        if self._info is None:
            shape, times, count, color = None, [], 0, {}
            for frame, pts, metadata in self._decode():
                if shape is None:
                    shape = frame.shape
                if frame.shape != shape:
                    raise EvaluationError(
                        f"Frame dimensions change inside sequence: {self.path}"
                    )
                times.append(pts)
                color = metadata
                count += 1
            if count == 0:
                raise EvaluationError(f"Empty or unreadable sequence: {self.path}")
            fps = self.fps
            nominal_fps = color.get("nominal_fps")
            if fps is None and nominal_fps:
                fps = nominal_fps
            valid_pts = all(t is not None for t in times)
            if valid_pts and count > 1:
                deltas = np.diff(times)
                decoded_fps = nominal_fps or float(1.0 / np.mean(deltas))
                elapsed = np.asarray(times) - times[0]
                expected = np.arange(count) / decoded_fps
                tolerance = max(float(color.get("time_base", 0.0)) * 1.01, 1e-5)
                if np.any(deltas <= 0) or not np.allclose(
                    elapsed, expected, atol=tolerance, rtol=0
                ):
                    raise EvaluationError(
                        f"Nonuniform/nonmonotonic timestamps (VFR unsupported): {self.path}"
                    )
                if fps is not None and not np.isclose(fps, decoded_fps, rtol=1e-3):
                    raise EvaluationError(
                        f"Configured FPS disagrees with timestamps: {self.path}"
                    )
                fps = decoded_fps
            self._info = MediaInfo(
                count,
                shape[0],
                shape[1],
                fps,
                times if valid_pts else None,
                "frames" if self.image_paths is not None else "video",
                color,
            )
        return self._info

    def __iter__(self):
        for frame, _, _ in self._decode():
            yield frame

    def read_indices(self, indices):
        wanted = set(int(i) for i in indices)
        if not wanted or min(wanted) < 0 or max(wanted) >= self.info.frames:
            raise EvaluationError("Invalid sampled frame indices")
        # Retain only sampled frames, never the full video.
        found = {}
        for i, frame in enumerate(self):
            if i in wanted:
                found[i] = frame
            if len(found) == len(wanted):
                break
        if set(found) != wanted:
            raise EvaluationError(
                "Input changed or decoding ended before sampled frames"
            )
        return found

    def describe(self):
        return {"path": str(self.path), **asdict(self.info)}


def discover(root):
    root = Path(root)
    if not root.exists():
        raise EvaluationError(f"Sequence root not found: {root}")
    if root.is_file():
        return {root.stem: root}
    children = sorted(root.iterdir(), key=natural_key)
    images = [p for p in children if p.suffix.lower() in IMAGES and p.is_file()]
    sequences = [
        p
        for p in children
        if p.is_dir()
        and any(f.suffix.lower() in IMAGES for f in p.iterdir())
        or p.is_file()
        and p.suffix.lower() in VIDEOS
    ]
    if images and sequences:
        raise EvaluationError(f"Ambiguous mixed flat frames and clips: {root}")
    if images:
        return {root.name: root}
    result = {}
    for path in sequences:
        key = path.name if path.is_dir() else path.stem
        if key in result:
            raise EvaluationError(
                f"Duplicate clip ID {key!r} in {root}; use an explicit mapping CSV"
            )
        result[key] = path
    if not result:
        raise EvaluationError(f"No sequences discovered: {root}")
    return result


def pair_paths(pred_root, gt_root=None, mapping=None):
    if mapping:
        rows = []
        with Path(mapping).open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if not row.get("clip_id") or not row.get("pred"):
                    raise EvaluationError(
                        "Mapping CSV requires clip_id,pred and optional gt columns"
                    )
                pred = Path(row["pred"])
                gt = Path(row["gt"]) if row.get("gt") else None
                if not pred.is_absolute():
                    pred = Path(pred_root) / pred
                if gt is not None and not gt.is_absolute():
                    if gt_root is None:
                        raise EvaluationError("Relative mapping GT requires gt_root")
                    gt = Path(gt_root) / gt
                rows.append((row["clip_id"], pred, gt))
        if not rows or len({r[0] for r in rows}) != len(rows):
            raise EvaluationError("Mapping CSV is empty or contains duplicate clip_id")
        if any(r[2] is not None for r in rows) and not all(
            r[2] is not None for r in rows
        ):
            raise EvaluationError("Mapping CSV must supply GT for all clips or none")
        return sorted(rows, key=lambda r: natural_key(r[0]))
    pred = discover(pred_root)
    gt = discover(gt_root) if gt_root else {}
    if gt_root and pred.keys() != gt.keys():
        raise EvaluationError(
            f"Clip IDs differ; missing predictions={sorted(gt.keys() - pred.keys())}, missing GT={sorted(pred.keys() - gt.keys())}"
        )
    return [
        (key, path, gt.get(key))
        for key, path in sorted(pred.items(), key=lambda kv: natural_key(kv[0]))
    ]


def alignment(pred, reference, protocol):
    p, g = pred.info, reference.info if reference else None
    changes = []
    count = p.frames
    size = (p.height, p.width)
    if g:
        if p.frames != g.frames:
            if protocol["temporal_alignment"] == "strict":
                raise EvaluationError(
                    f"Frame counts differ: pred={p.frames}, GT={g.frames}"
                )
            count = min(p.frames, g.frames)
            changes.append({"truncate_frames": count, "original": [p.frames, g.frames]})
        if (p.height, p.width) != (g.height, g.width):
            if protocol["alignment"] == "strict":
                raise EvaluationError(
                    f"Dimensions differ: pred={p.height}x{p.width}, GT={g.height}x{g.width}"
                )
            size = min(p.height, g.height), min(p.width, g.width)
            changes.append({"crop_to": size, "policy": protocol["alignment"]})
        if (
            pred.image_paths is not None
            and reference.image_paths is not None
            and protocol.get("frame_pairing", "identity") == "identity"
        ):

            def key(path):
                numbers = re.findall(r"\d+", path.stem)
                return int(numbers[-1]) if numbers else path.stem

            pkeys, gkeys = (
                [key(v) for v in pred.image_paths[:count]],
                [key(v) for v in reference.image_paths[:count]],
            )
            if len(set(pkeys)) != len(pkeys) or pkeys != gkeys:
                raise EvaluationError(
                    "Ordered frame IDs differ or repeat; set protocol.frame_pairing: ordinal explicitly if order-only pairing is intended"
                )
        if p.fps and g.fps and not np.isclose(p.fps, g.fps, rtol=1e-3):
            raise EvaluationError(f"FPS differs: pred={p.fps}, GT={g.fps}")
        if p.timestamps is not None and g.timestamps is not None:
            # Compare elapsed time, preserving valid container start offsets separately.
            pt = np.array(p.timestamps[:count]) - p.timestamps[0]
            gt = np.array(g.timestamps[:count]) - g.timestamps[0]
            tolerance = max(
                1e-4, p.color.get("time_base", 0.0) + g.color.get("time_base", 0.0)
            )
            if not np.allclose(pt, gt, atol=tolerance, rtol=0):
                raise EvaluationError(
                    "Prediction and reference presentation timelines differ"
                )
    return count, size, changes


def crop_common(frame, size, policy):
    h, w = size
    top, left = (
        ((frame.shape[0] - h) // 2, (frame.shape[1] - w) // 2)
        if policy == "center-common"
        else (0, 0)
    )
    return frame[top : top + h, left : left + w]
