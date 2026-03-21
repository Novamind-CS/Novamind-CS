"""
NovaMind — Surgical Gradient Attribution.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from novamind.core.device_manager import is_high_perf_mode


def apply_surgical_mask(loss, model, failing_line_index: int, total_tokens: int) -> Dict[str, Any]:
    """
    Attach a backward hook that dampens gradients before the failure boundary
    and amplifies gradients at or after the failure boundary.

    `failing_line_index` is interpreted as the token boundary to start
    amplification from. The caller is responsible for mapping a source-code line
    number to a token index.
    """
    if not is_high_perf_mode():
        return {"applied": False, "reason": "non_cuda"}

    target_tensor = getattr(model, "_sga_target_tensor", None)
    if not isinstance(target_tensor, torch.Tensor) or not target_tensor.requires_grad:
        return {"applied": False, "reason": "missing_target_tensor"}

    if total_tokens <= 0:
        return {"applied": False, "reason": "empty_sequence"}

    token_boundary = max(0, min(int(failing_line_index), int(total_tokens - 1)))

    def _hook(grad: torch.Tensor):
        if grad is None or grad.dim() < 2:
            return grad

        scaled = grad.clone()
        safe_slice = slice(0, token_boundary)
        poison_slice = slice(token_boundary, grad.shape[-2])

        if token_boundary > 0:
            scaled[..., safe_slice, :] *= 0.1
        scaled[..., poison_slice, :] *= 10.0
        return scaled

    handle = target_tensor.register_hook(_hook)
    return {
        "applied": True,
        "handle": handle,
        "token_boundary": token_boundary,
    }
