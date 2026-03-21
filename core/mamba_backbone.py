"""
NovaMind — 硬件感知 Mamba backbone

高性能设备上优先走官方 `mamba_ssm` 实现。
在 macOS / CPU 上退化到纯 PyTorch fallback，但保持相同张量形状接口。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from novamind.core.device_manager import is_high_perf_mode
from novamind.core.ssm import MultiHeadSSM

try:
    from mamba_ssm import Mamba2 as OfficialMamba2
except Exception:
    try:
        from mamba_ssm.modules.mamba2 import Mamba2 as OfficialMamba2
    except Exception:
        OfficialMamba2 = None


class VanillaPyTorchMambaFallback(nn.Module):
    """
    Debug fallback:
    用 GRU + 线性投影模拟 Mamba 的形状与残差行为。
    不追求速度，只追求在 Mac/CPU 上稳定调试。
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.implementation_type = "gru_fallback"
        self.gru = nn.GRU(d_model, d_model, batch_first=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.RMSNorm(d_model)

    def forward(self, x: torch.Tensor,
                inference_mode: bool = False,
                states=None,
                return_states: bool = False):
        h0 = None
        if states is not None:
            h0 = states[0] if isinstance(states, list) and states else states
            if h0 is not None and h0.dim() == 2:
                h0 = h0.unsqueeze(0)

        out, hn = self.gru(x, h0)
        out = self.out_proj(self.norm(out))
        next_state = [hn.squeeze(0)] if hn is not None else [None]

        if return_states:
            return out, next_state
        return out


class OfficialMambaBackbone(nn.Module):
    """
    包装官方 Mamba2，并在不支持的调用路径上回退到 MultiHeadSSM。
    """

    def __init__(self, d_model: int, d_state: int, d_conv: int, expand: int, num_heads: int):
        super().__init__()
        self.implementation_type = "official_mamba2"
        self.optimized = OfficialMamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.stateful_fallback = MultiHeadSSM(
            d_model=d_model,
            num_heads=num_heads,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            backend="mamba2",
        )

    def forward(self, x: torch.Tensor,
                inference_mode: bool = False,
                states=None,
                return_states: bool = False):
        if inference_mode or states is not None or return_states:
            return self.stateful_fallback(
                x,
                inference_mode=inference_mode,
                states=states,
                return_states=return_states,
            )

        out = self.optimized(x)
        if return_states:
            return out, [None]
        return out


def build_mamba_backbone(d_model: int,
                         d_state: int,
                         d_conv: int,
                         expand: int,
                         num_heads: int):
    backbone = None
    if is_high_perf_mode() and OfficialMamba2 is not None:
        backbone = OfficialMambaBackbone(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            num_heads=num_heads,
        )
    elif is_high_perf_mode():
        backbone = MultiHeadSSM(
            d_model=d_model,
            num_heads=num_heads,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            backend="mamba2",
        )
        backbone.implementation_type = "multihead_ssm"
    else:
        backbone = VanillaPyTorchMambaFallback(d_model)

    print(f"[Backbone] Loaded: {backbone.implementation_type}")
    return backbone
