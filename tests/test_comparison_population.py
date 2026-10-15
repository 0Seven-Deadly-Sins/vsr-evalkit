from vsr_evalkit.config import Config
from vsr_evalkit.runner import evaluate


def test_model_specific_mapping_to_different_gt_is_not_comparable(
    frames, config_path, tmp_path
):
    pred = frames("pred/a", [20])
    first = frames("gt1/a", [0])
    second = frames("gt2/a", [10])
    path = config_path(
        {"one": pred.parent, "two": pred.parent},
        {"d": {"gt_root": str(first.parent)}},
        metrics=["psnr"],
    )
    mapping = tmp_path / "two.csv"
    mapping.write_text(f"clip_id,pred,gt\na,a,{second.as_posix()}\n")
    config = Config(path)
    config.data["models"]["two"]["mappings"] = {"d": str(mapping)}
    report, _ = evaluate(config, ["one", "two"], ["d"], output=tmp_path / "comparison")
    left, right = [job["summary"]["psnr"] for job in report["jobs"]]
    assert left["status"] == right["status"] == "ok"
    assert left["comparison_group"] != right["comparison_group"]
