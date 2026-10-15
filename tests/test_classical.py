import math

import cv2
import numpy as np
import pytest

from vsr_evalkit.errors import EvaluationError
from vsr_evalkit.metrics.classical import prepare, psnr, ssim


def test_psnr_closed_form_and_identity():
    zero = np.zeros((17, 23, 3), np.uint8)
    ten = np.full_like(zero, 10)
    assert psnr(zero, ten) == pytest.approx(20 * math.log10(255 / 10), abs=1e-12)
    assert psnr(ten, ten) == float("inf")
    assert ssim(ten, ten) == pytest.approx(1, abs=1e-12)


def test_ssim_constant_luminance_closed_form():
    a, b = np.full((12, 17, 3), 20, np.uint8), np.full((12, 17, 3), 80, np.uint8)
    expected = (2 * (20 / 255) * (80 / 255) + 0.01**2) / (
        (20 / 255) ** 2 + (80 / 255) ** 2 + 0.01**2
    )
    assert ssim(a, b) == pytest.approx(expected, abs=1e-12)


def test_bt601_y_color_bars():
    colors = np.array(
        [[[0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255]]], np.uint8
    )
    assert prepare(colors, "bt601-y")[0, :, 0] * 255 == pytest.approx(
        [16, 235, 81.481, 144.553, 40.966]
    )


@pytest.mark.parametrize("shape", [(17, 23, 3), (77, 31, 3), (31, 77, 3)])
@pytest.mark.parametrize("color", ["rgb", "bt601-y"])
def test_ssim_independent_opencv_valid_window(shape, color):
    rng = np.random.default_rng(40)
    a, b = (rng.integers(0, 256, shape, dtype=np.uint8) for _ in range(2))
    x, y = (prepare(v, color) for v in (a, b))
    kernel = cv2.getGaussianKernel(11, 1.5)
    filt = lambda v: cv2.filter2D(v, -1, kernel @ kernel.T)[5:-5, 5:-5]
    values = []
    for c in range(x.shape[-1]):
        p, q = x[:, :, c], y[:, :, c]
        up, uq = filt(p), filt(q)
        cov = filt(p * q) - up * uq
        varp, varq = filt(p * p) - up * up, filt(q * q) - uq * uq
        values.append(
            np.mean(
                ((2 * up * uq + 0.01**2) * (2 * cov + 0.03**2))
                / ((up * up + uq * uq + 0.01**2) * (varp + varq + 0.03**2))
            )
        )
    assert ssim(a, b, color) == pytest.approx(np.mean(values), abs=2e-14)


def test_crop_and_invalid_inputs():
    a = np.zeros((21, 21, 3), np.uint8)
    b = a.copy()
    b[0] = 255
    assert psnr(a, b, crop_border=1) == float("inf")
    assert ssim(a, b, crop_border=1) == 1
    with pytest.raises(EvaluationError):
        ssim(a[:10], b[:10])
    with pytest.raises(EvaluationError):
        psnr(a, b, crop_border=11)


@pytest.mark.parametrize("shape", [(1080, 1920, 3), (2160, 3840, 3), (1921, 1081, 3)])
def test_large_and_odd_dimensions(shape):
    a = np.zeros(shape, np.uint8)
    assert psnr(a, a) == float("inf")
    assert ssim(a, a) == pytest.approx(1)
