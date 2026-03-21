"""Unified hardware tier detection and device management."""

from __future__ import annotations

from contextlib import nullcontext
from enum import Enum
from typing import Optional

import torch


class HardwareTier(str, Enum):
    CUDA_EXTREME = "CUDA_EXTREME"
    MAC_DEBUG = "MAC_DEBUG"
    CPU_FALLBACK = "CPU_FALLBACK"


def detect_hardware_tier() -> HardwareTier:
    if torch.cuda.is_available():
        return HardwareTier.CUDA_EXTREME
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return HardwareTier.MAC_DEBUG
    return HardwareTier.CPU_FALLBACK


def get_device(preferred: str = "auto") -> torch.device:
    if preferred != "auto":
        return torch.device(preferred)

    tier = detect_hardware_tier()
    if tier == HardwareTier.CUDA_EXTREME:
        return torch.device("cuda")
    if tier == HardwareTier.MAC_DEBUG:
        return torch.device("mps")
    return torch.device("cpu")


def get_dtype(tier: Optional[HardwareTier] = None) -> torch.dtype:
    tier = tier or detect_hardware_tier()
    if tier == HardwareTier.CUDA_EXTREME:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if tier == HardwareTier.MAC_DEBUG:
        return torch.float16
    return torch.float32


def is_high_perf_mode(tier: Optional[HardwareTier] = None) -> bool:
    return (tier or detect_hardware_tier()) == HardwareTier.CUDA_EXTREME


def supports_triton(tier: Optional[HardwareTier] = None) -> bool:
    return is_high_perf_mode(tier)


def get_hardware_banner(device: Optional[torch.device] = None) -> str:
    tier = detect_hardware_tier()
    device = device or get_device()
    dtype = get_dtype(tier)
    return f"[Hardware] tier={tier.value} device={device.type} dtype={dtype}"


def get_autocast_context(device: torch.device, enabled: bool = True):
    if not enabled:
        return nullcontext()

    tier = detect_hardware_tier()
    if tier == HardwareTier.CUDA_EXTREME:
        return torch.autocast(device_type="cuda", dtype=get_dtype(tier))
    if tier == HardwareTier.MAC_DEBUG:
        return torch.autocast(device_type="cpu", enabled=False)
    return nullcontext()
