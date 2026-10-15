"""Explicit masked RGB MSE using forward/backward flow (pixels, not normalized grids)."""

import torch
import torch.nn.functional as F

from ..errors import NotApplicable


def sample_with_flow(value, flow):
    b, _, h, w = flow.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=flow.device),
        torch.arange(w, device=flow.device),
        indexing="ij",
    )
    x, y = xx[None] + flow[:, 0], yy[None] + flow[:, 1]
    valid = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    grid = torch.stack((2 * x / max(w - 1, 1) - 1, 2 * y / max(h - 1, 1) - 1), dim=-1)
    sampled = F.grid_sample(
        value, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    return sampled, valid[:, None]


def resize_flow(flow, height, width):
    old_h, old_w = flow.shape[-2:]
    output = F.interpolate(
        flow, size=(height, width), mode="bilinear", align_corners=False
    )
    return (
        output * flow.new_tensor([width / old_w, height / old_h])[None, :, None, None]
    )


def masked_warp_mse(first, second, forward, backward, relative=0.01, absolute=0.5):
    if (
        first.shape != second.shape
        or forward.shape != backward.shape
        or first.shape[-2:] != forward.shape[-2:]
    ):
        raise ValueError(
            "Images and forward/backward fields must share spatial dimensions"
        )
    warped, valid = sample_with_flow(second, forward)
    reverse, reverse_valid = sample_with_flow(backward, forward)
    residual = (forward + reverse).square().sum(1, keepdim=True)
    magnitude = forward.square().sum(1, keepdim=True) + reverse.square().sum(
        1, keepdim=True
    )
    mask = valid & reverse_valid & (residual <= relative * magnitude + absolute)
    pixels = int(mask.sum().item())
    if pixels == 0:
        raise NotApplicable("No valid non-occluded pixels in this frame pair")
    value = ((first - warped).square() * mask).sum(dtype=torch.float64) / (3 * pixels)
    return float(value.item()), {"valid_pixels": pixels, "total_pixels": mask.numel()}
