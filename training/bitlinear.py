"""
Hardware-aware BitLinear module.

CUDA_EXTREME enables 1.58-bit ternary QAT with a custom STE path. On macOS and
CPU the implementation safely falls back to standard nn.Linear while preserving
the same interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from novamind.core.device_manager import is_high_perf_mode
from novamind.core.lod_router import get_lod_tier


_FALLBACK_WARNING_EMITTED = False
_LAST_LOGGED_LOD: Optional[str] = None


def _warn_fallback_once():
    global _FALLBACK_WARNING_EMITTED
    if not _FALLBACK_WARNING_EMITTED:
        print(
            "[BitLinear] macOS/CPU detected. Falling back to standard "
            "FP32/FP16 nn.Linear to prevent MPS crashes."
        )
        _FALLBACK_WARNING_EMITTED = True


def _log_lod_switch_once(tier: str):
    global _LAST_LOGGED_LOD
    if tier == _LAST_LOGGED_LOD:
        return
    label = "High-Fi" if tier == "high_fi" else "Low-Fi"
    print(f"[LOD] Routing to {label}")
    _LAST_LOGGED_LOD = tier


class AbsMeanTernarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor, threshold: float):
        scale = weight.abs().mean().clamp(min=1e-6)
        normalized = weight / scale
        ternary = normalized.sign() * (normalized.abs() > threshold).to(weight.dtype)
        return ternary * scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


@dataclass
class CompiledTernaryCircuit:
    pos_idx: torch.Tensor
    neg_idx: torch.Tensor
    pos_mask: torch.Tensor
    neg_mask: torch.Tensor
    ternary_weight: torch.Tensor


class BitLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 threshold: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold
        self.high_perf = is_high_perf_mode()
        self.compiled_circuit: Optional[CompiledTernaryCircuit] = None

        if self.high_perf:
            self.weight_fp = nn.Parameter(torch.empty(out_features, in_features))
            nn.init.kaiming_uniform_(self.weight_fp, a=5**0.5)
            if bias:
                self.bias_param = nn.Parameter(torch.zeros(out_features))
            else:
                self.register_parameter("bias_param", None)
            self.linear = None
        else:
            _warn_fallback_once()
            self.linear = nn.Linear(in_features, out_features, bias=bias)
            self.register_parameter("weight_fp", None)
            self.register_parameter("bias_param", None)

    @property
    def weight(self) -> torch.Tensor:
        if self.high_perf:
            return self.weight_fp
        return self.linear.weight

    @property
    def bias(self) -> Optional[torch.Tensor]:
        if self.high_perf:
            return self.bias_param
        return self.linear.bias

    @classmethod
    def from_linear(cls, linear: nn.Linear, threshold: float = 0.5) -> "BitLinear":
        layer = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
            threshold=threshold,
        )
        with torch.no_grad():
            if layer.high_perf:
                layer.weight_fp.copy_(linear.weight.data)
                if linear.bias is not None:
                    layer.bias_param.copy_(linear.bias.data)
            else:
                layer.linear.weight.copy_(linear.weight.data)
                if linear.bias is not None:
                    layer.linear.bias.copy_(linear.bias.data)
        return layer

    def quantize_weight(self) -> torch.Tensor:
        if self.high_perf:
            return AbsMeanTernarySTE.apply(self.weight_fp, self.threshold)

        weight = self.linear.weight
        scale = weight.abs().mean().clamp(min=1e-6)
        normalized = weight / scale
        ternary = normalized.sign() * (normalized.abs() > self.threshold).to(weight.dtype)
        return ternary * scale

    def compile_circuit(self, device: Optional[torch.device] = None) -> CompiledTernaryCircuit:
        ternary = self.quantize_weight().detach()
        ternary_sign = torch.sign(ternary)

        pos_lists = []
        neg_lists = []
        max_pos = 1
        max_neg = 1
        for out_idx in range(self.out_features):
            pos = torch.nonzero(ternary_sign[out_idx] > 0, as_tuple=False).flatten()
            neg = torch.nonzero(ternary_sign[out_idx] < 0, as_tuple=False).flatten()
            pos_lists.append(pos)
            neg_lists.append(neg)
            max_pos = max(max_pos, pos.numel())
            max_neg = max(max_neg, neg.numel())

        pos_idx = torch.zeros(self.out_features, max_pos, dtype=torch.long)
        neg_idx = torch.zeros(self.out_features, max_neg, dtype=torch.long)
        pos_mask = torch.zeros(self.out_features, max_pos, dtype=torch.bool)
        neg_mask = torch.zeros(self.out_features, max_neg, dtype=torch.bool)

        for out_idx, pos in enumerate(pos_lists):
            if pos.numel():
                pos_idx[out_idx, :pos.numel()] = pos
                pos_mask[out_idx, :pos.numel()] = True
        for out_idx, neg in enumerate(neg_lists):
            if neg.numel():
                neg_idx[out_idx, :neg.numel()] = neg
                neg_mask[out_idx, :neg.numel()] = True

        if device is not None:
            pos_idx = pos_idx.to(device)
            neg_idx = neg_idx.to(device)
            pos_mask = pos_mask.to(device)
            neg_mask = neg_mask.to(device)
            ternary = ternary.to(device)

        self.compiled_circuit = CompiledTernaryCircuit(
            pos_idx=pos_idx,
            neg_idx=neg_idx,
            pos_mask=pos_mask,
            neg_mask=neg_mask,
            ternary_weight=ternary,
        )
        return self.compiled_circuit

    def forward_train(self, x: torch.Tensor) -> torch.Tensor:
        if not self.high_perf:
            return self.linear(x)

        weight_q = self.quantize_weight()
        return F.linear(
            x,
            weight_q.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )

    def forward_additive(self, x: torch.Tensor) -> torch.Tensor:
        if self.compiled_circuit is None or self.compiled_circuit.pos_idx.device != x.device:
            self.compile_circuit(x.device)

        circuit = self.compiled_circuit
        x_flat = x.reshape(-1, self.in_features)
        pos_terms = x_flat[:, circuit.pos_idx]
        neg_terms = x_flat[:, circuit.neg_idx]
        pos_sum = pos_terms.masked_fill(~circuit.pos_mask.unsqueeze(0), 0).sum(dim=-1)
        neg_sum = neg_terms.masked_fill(~circuit.neg_mask.unsqueeze(0), 0).sum(dim=-1)

        out = pos_sum - neg_sum
        scale = circuit.ternary_weight.abs().amax(dim=-1).clamp(min=1e-6).view(1, -1)
        out = out * scale.to(out.dtype)

        bias = self.bias
        if bias is not None:
            out = out + bias.to(out.dtype)
        return out.view(*x.shape[:-1], self.out_features)

    def forward(self, x: torch.Tensor, additive_inference: bool = False) -> torch.Tensor:
        if not self.high_perf:
            return self.linear(x)

        lod_tier = get_lod_tier()
        _log_lod_switch_once(lod_tier)

        if lod_tier == "high_fi":
            return F.linear(
                x,
                self.weight_fp.to(x.dtype),
                None if self.bias is None else self.bias.to(x.dtype),
            )

        if self.training or not additive_inference:
            return self.forward_train(x)
        return self.forward_additive(x)


class BitLinear158(BitLinear):
    pass


def replace_with_bitlinear(model: nn.Module,
                           target_keywords: Sequence[str] = (
                               "gate_proj", "up_proj", "down_proj",
                               "q_proj", "k_proj", "v_proj", "out_proj",
                           ),
                           threshold: float = 0.5):
    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(keyword in name for keyword in target_keywords):
            continue

        parts = name.rsplit(".", 1)
        if len(parts) == 1:
            parent, attr = model, parts[0]
        else:
            parent = model
            for part in parts[0].split("."):
                parent = getattr(parent, part)
            attr = parts[1]

        setattr(parent, attr, BitLinear.from_linear(module, threshold=threshold))
        replaced += 1

    return model, replaced
