import numpy as np
from scipy.ndimage import correlate1d

from ..errors import EvaluationError


def prepare(image, color_space="rgb", crop_border=0):
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise EvaluationError("Classical metric input must be HWC RGB uint8")
    array = array.astype(np.float64) / 255.0
    if crop_border:
        if min(array.shape[:2]) <= 2 * crop_border:
            raise EvaluationError("crop_border removes the entire frame")
        array = array[crop_border:-crop_border, crop_border:-crop_border]
    if color_space == "bt601-y":
        # BasicSR/MATLAB studio-range BT.601: Y8 = 16 + 65.481 R + 128.553 G + 24.966 B.
        array = (array @ np.array([65.481, 128.553, 24.966]) / 255.0 + 16.0 / 255.0)[
            ..., None
        ]
    elif color_space == "yiq-y":
        array = (array @ np.array([0.299, 0.587, 0.114]))[..., None]
    elif color_space != "rgb":
        raise EvaluationError(f"Unsupported color space: {color_space}")
    return array


def psnr(pred, reference, color_space="rgb", crop_border=0, eps=0.0):
    if pred.shape != reference.shape:
        raise EvaluationError("PSNR requires equal shapes")
    x, y = (prepare(a, color_space, crop_border) for a in (pred, reference))
    mse = np.mean((x - y) ** 2, dtype=np.float64)
    return float("inf") if mse + eps == 0 else float(-10.0 * np.log10(mse + eps))


def ssim(
    pred,
    reference,
    color_space="rgb",
    crop_border=0,
    clip_negative_cs=False,
    round_values=False,
):
    if pred.shape != reference.shape:
        raise EvaluationError("SSIM requires equal shapes")
    x, y = (prepare(a, color_space, crop_border) for a in (pred, reference))
    if round_values:
        x, y = np.rint(x * 255) / 255, np.rint(y * 255) / 255
    if min(x.shape[:2]) < 11:
        raise EvaluationError("SSIM requires at least 11x11 pixels after cropping")
    coords = np.arange(-5, 6, dtype=np.float64)
    kernel = np.exp(-(coords**2) / (2 * 1.5**2))
    kernel /= kernel.sum()

    def filt(a):
        return correlate1d(correlate1d(a, kernel, axis=0), kernel, axis=1)[5:-5, 5:-5]

    ux, uy = filt(x), filt(y)
    vx, vy = filt(x * x) - ux * ux, filt(y * y) - uy * uy
    cov = filt(x * y) - ux * uy
    c1, c2 = 0.01**2, 0.03**2
    cs = (2 * cov + c2) / (vx + vy + c2)
    if clip_negative_cs:
        cs = np.maximum(cs, 0)
    score = ((2 * ux * uy + c1) / (ux * ux + uy * uy + c1)) * cs
    return float(score.mean(dtype=np.float64))
