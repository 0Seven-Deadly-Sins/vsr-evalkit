import numpy as np
import pytest
from PIL import Image

from vsr_evalkit.config import Config
from vsr_evalkit.errors import EvaluationError
from vsr_evalkit.media import Sequence, alignment, discover, pair_paths


def test_natural_sort_and_sparse_reads(tmp_path):
    for index in [10, 2, 1]:
        Image.fromarray(np.full((15, 17, 3), index, np.uint8)).save(
            tmp_path / f"{index}.png"
        )
    sequence = Sequence(tmp_path)
    assert [int(f[0, 0, 0]) for f in sequence] == [1, 2, 10]
    assert sequence.info.frames == 3
    assert set(sequence.read_indices([2, 0, 2])) == {0, 2}


def test_mismatches_are_not_silently_cropped_or_truncated(frames, config_path):
    p = Sequence(frames("pred/a", [0, 1], shape=(17, 19, 3)))
    g = Sequence(frames("gt/a", [0, 1, 2], shape=(18, 20, 3)))
    config = Config(
        config_path({"m": p.path.parent}, {"d": {"gt_root": str(g.path.parent)}})
    )
    with pytest.raises(EvaluationError, match="Frame counts differ"):
        alignment(p, g, config.protocol())
    count, size, changes = alignment(p, g, config.protocol("dove-code-alignment-v1"))
    assert (count, size) == (2, (17, 19)) and len(changes) == 2


def test_duplicate_ids_and_missing_clips(frames, tmp_path):
    root = frames("pred/a", [0]).parent
    (root / "a.mp4").write_bytes(b"bad")
    with pytest.raises(EvaluationError, match="Duplicate"):
        discover(root)
    gt = frames("gt/b", [0]).parent
    (root / "a.mp4").unlink()
    with pytest.raises(EvaluationError, match="Clip IDs differ"):
        pair_paths(root, gt)


def test_mapping_file(frames, tmp_path):
    pred = frames("pred/output_01", [0])
    gt = frames("gt/truth_001", [0])
    mapping = tmp_path / "map.csv"
    mapping.write_text("clip_id,pred,gt\nclip1,output_01,truth_001\n")
    assert pair_paths(pred.parent, gt.parent, mapping) == [("clip1", pred, gt)]


def test_empty_corrupt_and_high_bit_depth(tmp_path):
    with pytest.raises(EvaluationError):
        Sequence(tmp_path)
    bad = tmp_path / "bad.mkv"
    bad.write_bytes(b"bad")
    with pytest.raises(EvaluationError, match="decode"):
        Sequence(bad).info
    high = tmp_path / "high.png"
    Image.fromarray(np.zeros((13, 17), np.uint16)).save(high)
    with pytest.raises(EvaluationError, match="8-bit"):
        Sequence(high).info


def test_lossless_video_matches_frames(frames, tmp_path):
    import av

    rng = np.random.default_rng(8)
    arrays = [rng.integers(0, 256, (17, 23, 3), np.uint8) for _ in range(3)]
    folder = frames("sequence", arrays)
    video = tmp_path / "sequence.mkv"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("ffv1", rate=25)
        stream.width = 23
        stream.height = 17
        stream.pix_fmt = "bgr0"
        for array in arrays:
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(array, format="rgb24")
            ):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    decoded = Sequence(video)
    assert decoded.info.frames == 3 and decoded.info.fps == pytest.approx(25)
    for a, b in zip(Sequence(folder), decoded):
        np.testing.assert_array_equal(a, b)


def test_dimensions_changing_inside_sequence_fail(frames):
    path = frames(
        "variable", [np.zeros((13, 17, 3), np.uint8), np.zeros((15, 17, 3), np.uint8)]
    )
    with pytest.raises(EvaluationError, match="dimensions change"):
        Sequence(path).info
