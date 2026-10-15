import numpy as np
import pytest

from vsr_evalkit.config import Config
from vsr_evalkit.errors import EvaluationError
from vsr_evalkit.media import Sequence, alignment
from vsr_evalkit.metrics.classical import psnr, ssim


def test_epsilon_and_clipped_ssim_are_explicit_variants():
    image = np.tile(np.array([0, 255], np.uint8), (32, 16))[..., None].repeat(3, axis=2)
    assert ssim(image, 255 - image) < 0
    assert ssim(image, 255 - image, clip_negative_cs=True) == 0
    assert psnr(image, image) == float("inf")
    assert psnr(image, image, eps=1e-8) == 80


def test_frame_identity_mismatch_and_explicit_ordinal(frames, config_path):
    pred = frames("pred/a", [0, 0])
    gt = frames("gt/a", [0, 0])
    (gt / "frame1.png").rename(gt / "frame9.png")
    config = Config(config_path({"m": pred.parent}, {"d": {"gt_root": str(gt.parent)}}))
    with pytest.raises(EvaluationError, match="frame IDs"):
        alignment(Sequence(pred), Sequence(gt), config.protocol())
    policy = config.protocol()
    policy["frame_pairing"] = "ordinal"
    assert alignment(Sequence(pred), Sequence(gt), policy)[0] == 2


def test_unknown_fps_is_preserved_and_mismatch_rejected(frames, config_path):
    pred = frames("pred/a", [0, 0])
    gt = frames("gt/a", [0, 0])
    config = Config(config_path({"m": pred.parent}, {"d": {"gt_root": str(gt.parent)}}))
    assert Sequence(pred).info.fps is None
    with pytest.raises(EvaluationError, match="FPS differs"):
        alignment(Sequence(pred, 25), Sequence(gt, 30), config.protocol())
