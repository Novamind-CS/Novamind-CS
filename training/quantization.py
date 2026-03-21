"""Asymmetric mixed-precision and 1.58-bit QAT helpers."""

from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


SENSITIVE_KEYWORDS = ("embed", "lm_head", "norm", "logic_layer", "wsra")
DEFAULT_TARGET_KEYWORDS = (
    "gate_proj", "up_proj", "down_proj",
    "q_proj", "k_proj", "v_proj", "out_proj",
)


class QuantizedLinearQAT(nn.Module):
    """
    """

    def __init__(self, original: nn.Linear,
                 clip_value: float = 6.0,
                 smooth_alpha: float = 0.5):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.clip_value = clip_value
        self.smooth_alpha = smooth_alpha

        self.weight_fp = nn.Parameter(original.weight.data.clone().float())
        self.bias = None
        if original.bias is not None:
            self.bias = nn.Parameter(original.bias.data.clone().float())

        self.register_buffer("running_input_abs", torch.ones(self.in_features))
        self.register_buffer("quant_levels", torch.tensor([-1.0, 0.0, 1.0]))

    def _smooth_input(self, x: torch.Tensor):
        reduce_dims = tuple(range(x.dim() - 1))
        current_abs = x.detach().abs().mean(dim=reduce_dims).float()
        self.running_input_abs.mul_(0.95).add_(current_abs, alpha=0.05)
        smooth = torch.clamp(self.running_input_abs, min=1e-4).pow(self.smooth_alpha)
        x = x / smooth.to(x.dtype)
        x = torch.clamp(x, -self.clip_value, self.clip_value)
        return x, smooth

    @staticmethod
    def _ternary_quantize(weight: torch.Tensor) -> torch.Tensor:
        scale = weight.abs().mean().clamp(min=1e-6)
        normalized = weight / scale
        quantized = normalized.sign() * (normalized.abs() > 0.5).to(weight.dtype)
        return quantized * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_smooth, smooth = self._smooth_input(x)
        weight_source = self.weight_fp * smooth.unsqueeze(0)
        weight_q = self._ternary_quantize(weight_source)

        # Straight-through estimator
        weight_ste = weight_source + (weight_q - weight_source).detach()
        return F.linear(x_smooth, weight_ste.to(x.dtype),
                        None if self.bias is None else self.bias.to(x.dtype))


def preserve_sensitive_precision(model: nn.Module,
                                 norm_dtype: torch.dtype = torch.float32,
                                 sensitive_dtype: torch.dtype = torch.float16):
    for name, module in model.named_modules():
        if isinstance(module, (nn.LayerNorm, nn.RMSNorm)):
            module.to(dtype=norm_dtype)
        elif any(keyword in name for keyword in SENSITIVE_KEYWORDS):
            module.to(dtype=sensitive_dtype)
    return model


def apply_qat_quantization(model: nn.Module,
                           clip_value: float = 6.0,
                           smooth_alpha: float = 0.5,
                           target_keywords=DEFAULT_TARGET_KEYWORDS):
    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if any(keyword in name for keyword in SENSITIVE_KEYWORDS):
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

        setattr(
            parent,
            attr,
            QuantizedLinearQAT(module, clip_value=clip_value, smooth_alpha=smooth_alpha)
        )
        replaced += 1

    return model, replaced


def qat_parameter_groups(model: nn.Module):
    high_precision = []
    normal = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(keyword in name for keyword in ("weight_fp", "embed", "lm_head", "norm")):
            high_precision.append(param)
        else:
            normal.append(param)
    return [
        {"params": high_precision, "lr_scale": 1.0},
        {"params": normal, "lr_scale": 1.0},
    ]


def distillation_loss(student_logits: torch.Tensor,
                      teacher_logits: torch.Tensor,
                      student_hidden: Optional[torch.Tensor] = None,
                      teacher_hidden: Optional[torch.Tensor] = None,
                      temperature: float = 2.0,
                      alpha: float = 0.5):
    s_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    t_probs = F.softmax(teacher_logits / temperature, dim=-1)
    logits_loss = F.kl_div(s_log_probs, t_probs, reduction="batchmean") * (temperature ** 2)

    hidden_loss = student_logits.new_tensor(0.0)
    if student_hidden is not None and teacher_hidden is not None:
        hidden_dim = min(student_hidden.shape[-1], teacher_hidden.shape[-1])
        hidden_loss = F.mse_loss(
            student_hidden[..., :hidden_dim],
            teacher_hidden[..., :hidden_dim].detach()
        )

    total = alpha * logits_loss + (1.0 - alpha) * hidden_loss
    return {
        "total": total,
        "logits": logits_loss,
        "hidden": hidden_loss,
    }


def estimate_qat_vram(num_params: int,
                      quantized_fraction: float = 0.8,
                      master_dtype_bytes: int = 2,
                      optimizer_state_bytes: int = 4):
    quantized_bytes = num_params * quantized_fraction * 1.58 / 8
    sensitive_bytes = num_params * (1 - quantized_fraction) * master_dtype_bytes
    optimizer_bytes = num_params * quantized_fraction * optimizer_state_bytes * 2
    total = quantized_bytes + sensitive_bytes + optimizer_bytes
    return {
        "quantized_weights_gb": round(quantized_bytes / 1e9, 2),
        "sensitive_weights_gb": round(sensitive_bytes / 1e9, 2),
        "optimizer_state_gb": round(optimizer_bytes / 1e9, 2),
        "total_estimated_gb": round(total / 1e9, 2),
    }
