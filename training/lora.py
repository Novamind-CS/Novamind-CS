"""
LoRA utilities and 16 GB training helpers.

Includes LoRA layers, automatic injection utilities, activation offloading, and
memory tracking helpers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict
import gc



# LoRA


class LoRALinear(nn.Module):
    """
    
    
    """

    def __init__(self, original: nn.Linear, rank: int = 64,
                 alpha: int = 128, dropout: float = 0.05):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.rank = rank
        self.scale = alpha / rank


        self.weight = nn.Parameter(original.weight.data.clone(), requires_grad=False)
        self.bias = None
        if original.bias is not None:
            self.bias = nn.Parameter(original.bias.data.clone(), requires_grad=False)


        self.lora_A = nn.Parameter(
            torch.empty(rank, self.in_features).normal_(std=1 / rank ** 0.5)
        )
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        base = F.linear(x, self.weight, self.bias)

        lora_out = F.linear(self.lora_dropout(x), self.lora_A)
        lora_out = F.linear(lora_out, self.lora_B)
        return base + lora_out * self.scale

    def merge_weights(self):
        with torch.no_grad():
            delta = (self.lora_B @ self.lora_A) * self.scale
            self.weight.data += delta
        self.lora_A.requires_grad_(False)
        self.lora_B.requires_grad_(False)

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, rank={self.rank}"


def inject_lora(model: nn.Module, target_modules: List[str],
                rank: int = 64, alpha: int = 128,
                dropout: float = 0.05) -> nn.Module:
    """
    
    
    """
    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(t in name for t in target_modules):
            continue


        parts = name.rsplit(".", 1)
        if len(parts) == 1:
            parent, attr = model, parts[0]
        else:
            parent = model
            for part in parts[0].split("."):
                parent = getattr(parent, part)
            attr = parts[1]

        lora_layer = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, attr, lora_layer)
        replaced += 1

    print(f"[LoRA] Injected into {replaced} linear layers. "
          f"Trainable params: only LoRA A/B matrices.")
    return model


def freeze_base_model(model: nn.Module) -> nn.Module:
    frozen = 0
    trainable = 0
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad_(True)
            trainable += param.numel()
        else:
            param.requires_grad_(False)
            frozen += param.numel()
    total = frozen + trainable
    print(f"[LoRA] Frozen: {frozen/1e6:.1f}M params | "
          f"Trainable: {trainable/1e6:.2f}M params | "
          f"Ratio: {100*trainable/total:.2f}%")
    return model






class ActivationOffloader:
    """
    
    
        offloader = ActivationOffloader(threshold_mb=50)
        offloader.register(model)
        

    """

    def __init__(self, threshold_mb: float = 50.0):
        self.threshold_bytes = threshold_mb * 1024 * 1024
        self.cpu_stash: Dict[int, torch.Tensor] = {}
        self.hooks = []

    def _size_mb(self, tensor: torch.Tensor) -> float:
        return tensor.nelement() * tensor.element_size() / 1024 / 1024

    def register(self, model: nn.Module):
        for layer_id, module in enumerate(model.modules()):
            if not isinstance(module, (nn.Linear, nn.LayerNorm, nn.RMSNorm)):
                continue

            def make_hooks(lid):
                def fwd_hook(module, input, output):
                    if isinstance(output, torch.Tensor):
                        if output.nelement() * output.element_size() > self.threshold_bytes:

                            cpu_tensor = torch.empty(
                                output.shape, dtype=output.dtype,
                                pin_memory=True
                            )
                            cpu_tensor.copy_(output.detach(), non_blocking=True)
                            self.cpu_stash[lid] = cpu_tensor

                            return output.detach()
                    return output

                def bwd_hook(module, grad_input, grad_output):
                    if lid in self.cpu_stash:

                        gpu_tensor = self.cpu_stash.pop(lid).to(
                            grad_output[0].device if grad_output[0] is not None
                            else "cuda",
                            non_blocking=True
                        )
                        return (gpu_tensor,) + grad_input[1:]
                    return grad_input

                return fwd_hook, bwd_hook

            fh, bh = make_hooks(layer_id)
            self.hooks.append(module.register_forward_hook(fh))
            self.hooks.append(module.register_full_backward_hook(bh))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def clear_stash(self):
        self.cpu_stash.clear()
        gc.collect()






class MemoryTracker:

    def __init__(self, device="cuda"):
        self.device = device
        self.baseline = 0

    def reset(self):
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.baseline = torch.cuda.memory_allocated(self.device)

    def current_gb(self) -> float:
        return torch.cuda.memory_allocated(self.device) / 1e9

    def peak_gb(self) -> float:
        return torch.cuda.max_memory_allocated(self.device) / 1e9

    def report(self, label: str = ""):
        prefix = f"[{label}] " if label else ""
        print(f"{prefix}VRAM: current={self.current_gb():.2f}GB | "
              f"peak={self.peak_gb():.2f}GB")


def estimate_vram(num_params: int, dtype: str = "bfloat16",
                  batch_size: int = 1, seq_len: int = 2048,
                  hidden_dim: int = 4096, num_layers: int = 32,
                  use_activation_checkpointing: bool = True) -> dict:
    """
    
    """
    bytes_per_param = 2 if "16" in dtype else 4

    weights_gb = num_params * bytes_per_param / 1e9


    optimizer_gb = num_params * 4 * 2 / 1e9


    act_bytes_per_layer = batch_size * seq_len * hidden_dim * bytes_per_param
    if use_activation_checkpointing:

        activations_gb = act_bytes_per_layer * num_layers * 0.1 / 1e9
    else:
        activations_gb = act_bytes_per_layer * num_layers / 1e9

    grad_gb = weights_gb

    total_gb = weights_gb + optimizer_gb + activations_gb + grad_gb

    return {
        "weights_gb": round(weights_gb, 2),
        "optimizer_gb": round(optimizer_gb, 2),
        "activations_gb": round(activations_gb, 2),
        "gradients_gb": round(grad_gb, 2),
        "total_estimated_gb": round(total_gb, 2),
        "fits_16gb": total_gb < 15.5,
    }
