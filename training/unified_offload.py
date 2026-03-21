"""
NovaMind — 硬件感知 Unified Memory Offloading

CUDA_EXTREME:
- 启用基础 ZeRO 风格梯度卸载 hook

MAC_DEBUG / CPU_FALLBACK:
- 提供完全无副作用的 no-op 实现
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import torch
import torch.nn as nn

from novamind.core.device_manager import get_device, is_high_perf_mode


class UnifiedMemoryOffloader:
    """
    轻量层级流式搬运工具。

    这个类保留原有测试接口，与是否高性能硬件无关。
    """

    def __init__(self, execution_device: str = "cuda", offload_device: str = "cpu"):
        self.execution_device = execution_device
        self.offload_device = offload_device
        self._managed: List[nn.Module] = []

    def register_layers(self, layers: Iterable[nn.Module]):
        self._managed = list(layers)
        for layer in self._managed:
            layer.to(self.offload_device)
        return self

    def prefetch(self, layer: nn.Module):
        layer.to(self.execution_device, non_blocking=True)

    def release(self, layer: nn.Module):
        layer.to(self.offload_device, non_blocking=True)

    def stream_step(self, layer: nn.Module, *args, **kwargs):
        self.prefetch(layer)
        output = layer(*args, **kwargs)
        self.release(layer)
        return output

    def model_resident_bytes(self) -> int:
        total = 0
        for layer in self._managed:
            for param in layer.parameters(recurse=True):
                total += param.numel() * param.element_size()
        return total


class _NoOpZeROMemoryManager:
    def __init__(self, *args, **kwargs):
        self.handles: List = []
        self.offloaded_grads: Dict[str, torch.Tensor] = {}

    def register_offload_hooks(self, model: nn.Module, optimizer=None):
        return self

    def remove_hooks(self):
        self.handles.clear()

    def offload_optimizer_state(self, optimizer=None):
        return None

    def restore_optimizer_state(self, optimizer=None):
        return None

    def clear(self):
        self.offloaded_grads.clear()


class _CUDAZeROMemoryManager:
    def __init__(self, offload_device: str = "cpu"):
        self.offload_device = offload_device
        self.handles: List = []
        self.offloaded_grads: Dict[str, torch.Tensor] = {}

    def register_offload_hooks(self, model: nn.Module, optimizer=None):
        self.remove_hooks()
        self.offloaded_grads.clear()

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            def _make_hook(param_name: str):
                def _hook(grad: torch.Tensor):
                    self.offloaded_grads[param_name] = grad.detach().to(
                        self.offload_device, non_blocking=True
                    )
                    return grad
                return _hook

            self.handles.append(param.register_hook(_make_hook(name)))

        return self

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def offload_optimizer_state(self, optimizer=None):
        if optimizer is None:
            return
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(self.offload_device, non_blocking=True)

    def restore_optimizer_state(self, optimizer=None):
        if optimizer is None:
            return
        device = get_device()
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                if torch.is_tensor(value):
                    state[key] = value.to(device, non_blocking=True)

    def clear(self):
        self.offloaded_grads.clear()


class ZeROMemoryManager:
    def __init__(self, *args, **kwargs):
        if is_high_perf_mode():
            self._impl = _CUDAZeROMemoryManager(*args, **kwargs)
        else:
            self._impl = _NoOpZeROMemoryManager(*args, **kwargs)

    def register_offload_hooks(self, model: nn.Module, optimizer=None):
        return self._impl.register_offload_hooks(model, optimizer)

    def remove_hooks(self):
        return self._impl.remove_hooks()

    def offload_optimizer_state(self, optimizer=None):
        return self._impl.offload_optimizer_state(optimizer)

    def restore_optimizer_state(self, optimizer=None):
        return self._impl.restore_optimizer_state(optimizer)

    def clear(self):
        return self._impl.clear()
