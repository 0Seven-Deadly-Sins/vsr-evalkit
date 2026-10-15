import socket

import numpy as np
import pytest

from vsr_evalkit.backends import PyIQABackend, fuse_dover
from vsr_evalkit.errors import EvaluationError
from vsr_evalkit.offline import LocalResolver, check_artifacts, network_disabled


def test_network_is_blocked():
    with network_disabled():
        with pytest.raises(EvaluationError, match="disabled"):
            socket.create_connection(("example.com", 443))
        with socket.socket() as sock:
            with pytest.raises(EvaluationError):
                sock.connect(("127.0.0.1", 80))


def test_missing_implicit_backbone_fails_before_construction(tmp_path):
    head = tmp_path / "lpips.pth"
    head.write_bytes(b"local")
    with pytest.raises(EvaluationError, match="alexnet"):
        with network_disabled():
            check_artifacts(["lpips"], {"lpips_alex": head})


def test_local_resolver_does_not_fetch_unknown_artifacts(tmp_path):
    path = tmp_path / "local.pth"
    path.write_bytes(b"local")
    resolver = LocalResolver({"alexnet": path})
    with network_disabled():
        assert resolver.file(
            "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth"
        ) == str(path)
        with pytest.raises(EvaluationError, match="Unmapped"):
            resolver.file("https://example.com/unknown.pth")


def test_dover_branch_names_not_list_positions():
    first = fuse_dover({"technical": 0.1107, "aesthetic": -0.08285})
    second = fuse_dover({"aesthetic": -0.08285, "technical": 0.1107})
    assert first == second == {"aesthetic": 0.5, "technical": 0.5, "overall": 0.5}
    assert fuse_dover({"technical": 0.1107 + 0.07355, "aesthetic": -0.08285})[
        "overall"
    ] == pytest.approx(1 / (1 + np.exp(-0.6104)))


def test_perceptual_input_stays_rgb_and_zero_to_one():
    torch = pytest.importorskip("torch")
    backend = PyIQABackend.__new__(PyIQABackend)
    backend.torch = torch
    backend.device = "cpu"
    backend.name = "lpips"
    backend.resolver = LocalResolver({})

    def model(a, b):
        assert a.shape == (1, 3, 15, 17) and a.max() == 1 and b.max() == 0
        return torch.tensor([0.125])

    backend.model = model
    assert (
        backend.image(
            np.full((15, 17, 3), 255, np.uint8), np.zeros((15, 17, 3), np.uint8)
        )
        == 0.125
    )
