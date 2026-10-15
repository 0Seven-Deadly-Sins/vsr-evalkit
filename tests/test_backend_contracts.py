from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from vsr_evalkit.backends import RAFTBackend, VQABackend
from vsr_evalkit.config import Config
from vsr_evalkit.media import MediaInfo
from vsr_evalkit.offline import LocalResolver


class SampleSequence:
    info = MediaInfo(3, 240, 320, None, None, "mock", {})

    def read_indices(self, indices):
        return {int(i): np.full((240, 320, 3), 30 + int(i), np.uint8) for i in indices}


class Sampler:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, frames):
        return np.array([0, 1, 2, 0])


def datasets():
    return SimpleNamespace(
        UnifiedFrameSampler=Sampler,
        SampleFrames=Sampler,
        FragmentSampleFrames=Sampler,
        get_single_view=lambda clip, key, **opts: clip[:, :, :4, :5],
        get_spatial_fragments=lambda clip, **opts: clip[:, :, :4, :5],
    )


@pytest.mark.parametrize(
    "order", [["aesthetic", "technical"], ["technical", "aesthetic"]]
)
def test_dover_branch_order_and_frame_ids(order):
    backend = VQABackend.__new__(VQABackend)
    backend.torch = torch
    backend.device = "cpu"
    backend.name = "dover"
    backend.datasets = datasets()
    opts = {key: {"clip_len": 4, "num_clips": 1, "frame_interval": 1} for key in order}
    backend.opt = {"data": {"val-l1080p": {"args": {"sample_types": opts}}}}

    def model(views, reduce_scores):
        assert reduce_scores is False
        return [
            torch.tensor(
                [-0.08285 if key == "aesthetic" else 0.1107], dtype=torch.float64
            )
            for key in views
        ]

    backend.model = model
    value, detail = backend.video(SampleSequence(), 9)
    assert value == pytest.approx(0.5)
    assert detail["sampled_indices"] == {key: [0, 1, 2, 0] for key in order}


@pytest.mark.parametrize(
    "name,mean", [("fastvqa", -0.110198185), ("fastervqa", 0.14759505)]
)
def test_distinct_fast_and_faster_calibration(name, mean):
    backend = VQABackend.__new__(VQABackend)
    backend.torch = torch
    backend.device = "cpu"
    backend.name = name
    backend.datasets = datasets()
    backend.opt = {
        "data": {
            "val-kv1k": {
                "args": {"sample_types": {"fragments": {"clip_len": 4, "num_clips": 1}}}
            }
        }
    }
    backend.model = lambda views: torch.tensor(mean, dtype=torch.float64)
    value, detail = backend.video(SampleSequence(), 12)
    assert value == pytest.approx(0.5)
    assert detail["sampled_indices"]["fragments"] == [0, 1, 2, 0]


def test_raft_input_scaling_and_odd_size_padding(config_path, tmp_path):
    backend = RAFTBackend.__new__(RAFTBackend)
    backend.torch = torch
    backend.device = "cpu"
    backend.protocol = Config(config_path({"m": tmp_path}, {"d": {}})).protocol()
    calls = []

    def model(first, second, iters, test_mode):
        assert first.shape[-2:] == (136, 136)
        assert (
            first.min() == 200 and first.max() == 200
        )  # never normalized to [0,1] before original RAFT
        assert iters == 20 and test_mode
        calls.append(1)
        flow = torch.zeros(1, 2, *first.shape[-2:])
        return None, flow

    backend.model = model
    image = np.full((129, 133, 3), 200, np.uint8)
    value, detail = backend.pair(image, image)
    assert value == pytest.approx(0, abs=1e-12) and len(calls) == 2
    assert detail["valid_pixels"] == 129 * 133


def test_torchvision_imported_download_alias_is_local(tmp_path, monkeypatch):
    import torchvision.models._api as api

    path = tmp_path / "alexnet.pth"
    path.write_bytes(b"fixture")
    monkeypatch.setattr(torch, "load", lambda path, **kwargs: {"local": str(path)})
    with LocalResolver({"alexnet": path}).activate():
        result = api.load_state_dict_from_url(
            "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"
        )
    assert result == {"local": str(path)}
