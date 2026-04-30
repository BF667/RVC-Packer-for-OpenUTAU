"""Monkey-patch RVC attention for ONNX-compatible dynamic shapes.

The original attentions.py uses .view() with length-dependent constants
that get baked during torch.onnx.export tracing. This module replaces
those methods with torch.jit.script versions that export as dynamic
ONNX shape operations.
"""

import torch
import torch.nn.functional as F


@torch.jit.script
def rel_to_abs(x: torch.Tensor) -> torch.Tensor:
    B, H, L, _ = x.shape
    x = F.pad(x, [0, 1, 0, 0, 0, 0, 0, 0])
    x_flat = x.reshape(B, H, -1)
    x_flat = F.pad(x_flat, [0, L - 1, 0, 0, 0, 0])
    x_final = x_flat.reshape(B, H, L + 1, 2 * L - 1)[:, :, :L, L - 1 :]
    return x_final


@torch.jit.script
def abs_to_rel(x: torch.Tensor) -> torch.Tensor:
    B, H, L, _ = x.shape
    x = F.pad(x, [0, L - 1, 0, 0, 0, 0, 0, 0])
    x_flat = x.reshape(B, H, -1)
    x_flat = F.pad(x_flat, [L, 0, 0, 0, 0, 0])
    x_final = x_flat.reshape(B, H, L, 2 * L)[:, :, :, 1:]
    return x_final


def patch_attention_for_onnx(module):
    """Replace baked-constant attention reshape methods with dynamic versions."""
    for child in module.modules():
        if hasattr(child, "_relative_position_to_absolute_position"):
            child._relative_position_to_absolute_position = staticmethod(rel_to_abs)
            child._absolute_position_to_relative_position = staticmethod(abs_to_rel)
