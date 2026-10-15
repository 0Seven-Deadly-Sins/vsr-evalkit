import json
import math

import pytest

from vsr_evalkit.cli import main
from vsr_evalkit.config import Config
from vsr_evalkit.runner import build_jobs, evaluate


def test_macro_averaging_no_early_rounding_and_matrix(frames, config_path, tmp_path):
    pred = frames("pred/short", [10])
    frames("pred/long", [20, 20, 20])
    gt = frames("gt/short", [0])
    frames("gt/long", [0, 0, 0])
    config = config_path(
        {"first": pred.parent, "second": pred.parent},
        {"test": {"gt_root": str(gt.parent)}},
    )
    report, destination = evaluate(
        config, ["first", "second"], ["test"], output=tmp_path / "run"
    )
    expected = (20 * math.log10(255 / 10) + 20 * math.log10(255 / 20)) / 2
    assert report["jobs"][0]["summary"]["psnr"]["value"] == pytest.approx(
        expected, abs=1e-12
    )
    assert (
        report["jobs"][0]["summary"]["psnr"]["comparison_group"]
        == report["jobs"][1]["summary"]["psnr"]["comparison_group"]
    )
    assert all(
        (destination / name).is_file()
        for name in ["report.json", "details.csv", "summary.csv", "comparison.md"]
    )


def test_real_reference_requirement_and_protocol_ids(frames, config_path):
    pred = frames("pred/clip", [0])
    path = config_path({"m": pred.parent}, {"d": {}}, metrics="auto")
    config = Config(path)
    assert not any(
        n in ["psnr", "ssim", "lpips", "dists"]
        for n in build_jobs(config, ["m"], ["d"])[0]["metrics"]
    )
    with pytest.raises(ValueError, match="reference"):
        build_jobs(config, ["m"], ["d"], metrics="psnr")
    assert config.protocol()["id"] != config.protocol("flow-960-v1")["id"]


def test_infinity_serializes_as_valid_json(frames, config_path, tmp_path):
    pred = frames("pred/a", [0])
    gt = frames("gt/a", [0])
    path = config_path({"m": pred.parent}, {"d": {"gt_root": str(gt.parent)}})
    report, out = evaluate(path, ["m"], ["d"], output=tmp_path / "result")
    assert report["jobs"][0]["summary"]["psnr"]["value"] == float("inf")
    decoded = json.loads(
        (out / "report.json").read_text(),
        parse_constant=lambda v: pytest.fail("nonstandard JSON"),
    )
    assert decoded["jobs"][0]["summary"]["psnr"]["value"] == "+inf"


def test_metric_failure_does_not_average_a_subset(frames, config_path, tmp_path):
    pred = frames("pred/a", [0], shape=(9, 9, 3))
    gt = frames("gt/a", [1], shape=(9, 9, 3))
    path = config_path({"m": pred.parent}, {"d": {"gt_root": str(gt.parent)}})
    report, _ = evaluate(path, ["m"], ["d"], output=tmp_path / "bad")
    assert report["status"] == "incomplete"
    assert report["jobs"][0]["summary"]["ssim"]["value"] is None
    assert report["jobs"][0]["summary"]["ssim"]["status"] == "incomplete"


def test_single_frame_temporal_is_not_zero(frames, config_path, tmp_path, monkeypatch):
    pred = frames("pred/a", [0])
    path = config_path({"m": pred.parent}, {"d": {}}, metrics=["ewarp"])
    monkeypatch.setattr("vsr_evalkit.runner.check_artifacts", lambda *args: {})
    # Weight metadata is unnecessary for a mock backend; real preflight still requires the file.
    from vsr_evalkit.registry import METRICS, MetricSpec

    monkeypatch.setitem(METRICS, "ewarp", MetricSpec("ewarp", "pair", False, True))
    report, _ = evaluate(
        path,
        ["m"],
        ["d"],
        output=tmp_path / "single",
        backend_factory=lambda *args: None,
    )
    result = report["jobs"][0]["clips"][0]["metrics"]["ewarp"]
    assert result["status"] == "not_applicable" and result["value"] is None


def test_cli_and_no_overwrite(frames, config_path, tmp_path, capsys):
    pred = frames("pred/a", [0])
    gt = frames("gt/a", [1])
    path = config_path({"m": pred.parent}, {"d": {"gt_root": str(gt.parent)}})
    assert (
        main(["doctor", "--config", str(path), "--model", "m", "--dataset", "d"]) == 0
    )
    assert main(["list"]) == 0
    assert main(["weights", "info", "--metric", "psnr"]) == 0
    assert (
        main(
            [
                "run",
                "--config",
                str(path),
                "--model",
                "m",
                "--dataset",
                "d",
                "--output",
                str(tmp_path / "out"),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "run",
                "--config",
                str(path),
                "--model",
                "m",
                "--dataset",
                "d",
                "--output",
                str(tmp_path / "out"),
            ]
        )
        == 2
    )
