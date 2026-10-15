import pytest

torch = pytest.importorskip("torch")
from vsr_evalkit.errors import NotApplicable
from vsr_evalkit.metrics.temporal import masked_warp_mse, resize_flow, sample_with_flow


def test_translation_direction_and_valid_boundary():
    torch.manual_seed(7)
    first = torch.rand(1, 3, 9, 11)
    second = torch.zeros_like(first)
    second[:, :, :, 1:] = first[:, :, :, :-1]
    forward = torch.zeros(1, 2, 9, 11)
    forward[:, 0] = 1
    backward = -forward
    value, detail = masked_warp_mse(first, second, forward, backward)
    assert value < 1e-13
    assert detail["valid_pixels"] == 9 * 10
    wrong, _ = masked_warp_mse(first, second, -forward, -backward)
    assert wrong > 0.01


def test_rgb_normalization_and_occlusion():
    first = torch.zeros(1, 3, 7, 9)
    second = torch.ones_like(first)
    flow = torch.zeros(1, 2, 7, 9)
    value, _ = masked_warp_mse(first, second, flow, flow)
    assert value == pytest.approx(1)
    inconsistent = flow.clone()
    inconsistent[:, 0] = 100
    with pytest.raises(NotApplicable):
        masked_warp_mse(first, second, flow, inconsistent)


def test_scale_flow_components_and_border():
    field = torch.ones(1, 2, 4, 8)
    restored = resize_flow(field, 12, 16)
    assert torch.allclose(restored[:, 0], torch.full((1, 12, 16), 2.0))
    assert torch.allclose(restored[:, 1], torch.full((1, 12, 16), 3.0))
    image = torch.ones(1, 3, 5, 7)
    flow = torch.full((1, 2, 5, 7), 100.0)
    _, valid = sample_with_flow(image, flow)
    assert not valid.any()
