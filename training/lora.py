"""
NovaMind — LoRA 实现 + 16GB 显存训练工具链

包含:
1. LoRALinear: 低秩适配层，直接替换 nn.Linear
2. LoRAModel: 自动为目标模块注入 LoRA
3. ActivationOffloader: 激活卸载到 CPU RAM
4. MemoryTracker: 显存使用监控
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Dict
import gc


# ─────────────────────────────────────────────────────────
# LoRA
# ─────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """
    LoRA 包装的线性层
    
    原始权重 W 冻结，只训练低秩分解 A, B
    输出: W @ x + (B @ A) @ x * (alpha / rank)
    
    显存节省: 只需存 A, B 的梯度，而不是 W 的梯度
    """

    def __init__(self, original: nn.Linear, rank: int = 64,
                 alpha: int = 128, dropout: float = 0.05):
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.rank = rank
        self.scale = alpha / rank

        # 冻结原始权重
        self.weight = nn.Parameter(original.weight.data.clone(), requires_grad=False)
        self.bias = None
        if original.bias is not None:
            self.bias = nn.Parameter(original.bias.data.clone(), requires_grad=False)

        # LoRA 参数（只有这两个需要梯度）
        self.lora_A = nn.Parameter(
            torch.empty(rank, self.in_features).normal_(std=1 / rank ** 0.5)
        )
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始前向（不需要梯度）
        base = F.linear(x, self.weight, self.bias)
        # LoRA 增量（需要梯度）
        lora_out = F.linear(self.lora_dropout(x), self.lora_A)
        lora_out = F.linear(lora_out, self.lora_B)
        return base + lora_out * self.scale

    def merge_weights(self):
        """推理时将 LoRA 权重合并回原始权重（提速）"""
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
    自动为目标模块注入 LoRA
    
    target_modules: 模块名称的关键词列表
                    例如 ["q_proj", "k_proj", "v_proj"]
    
    返回注入后的模型（原地修改）
    """
    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(t in name for t in target_modules):
            continue

        # 找到父模块和属性名
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
    """冻结所有非 LoRA 参数"""
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


# ─────────────────────────────────────────────────────────
# 激活内存卸载
# ─────────────────────────────────────────────────────────

class ActivationOffloader:
    """
    Token 级激活卸载器
    
    在 forward pass 中将超过阈值大小的激活卸载到 CPU pinned memory
    在 backward pass 需要时再异步预取回 GPU
    
    用法（手动 hook 注册模式）:
        offloader = ActivationOffloader(threshold_mb=50)
        offloader.register(model)
        
        # 然后正常训练，offloader 自动管理卸载/预取
    """

    def __init__(self, threshold_mb: float = 50.0):
        self.threshold_bytes = threshold_mb * 1024 * 1024
        self.cpu_stash: Dict[int, torch.Tensor] = {}  # layer_id → tensor
        self.hooks = []

    def _size_mb(self, tensor: torch.Tensor) -> float:
        return tensor.nelement() * tensor.element_size() / 1024 / 1024

    def register(self, model: nn.Module):
        """为模型中的关键层注册 forward/backward hook"""
        for layer_id, module in enumerate(model.modules()):
            if not isinstance(module, (nn.Linear, nn.LayerNorm, nn.RMSNorm)):
                continue

            def make_hooks(lid):
                def fwd_hook(module, input, output):
                    if isinstance(output, torch.Tensor):
                        if output.nelement() * output.element_size() > self.threshold_bytes:
                            # 卸载到 CPU pinned memory（异步）
                            cpu_tensor = torch.empty(
                                output.shape, dtype=output.dtype,
                                pin_memory=True
                            )
                            cpu_tensor.copy_(output.detach(), non_blocking=True)
                            self.cpu_stash[lid] = cpu_tensor
                            # 返回占位符（空梯度将触发预取）
                            return output.detach()  # 停止梯度追踪
                    return output

                def bwd_hook(module, grad_input, grad_output):
                    if lid in self.cpu_stash:
                        # 需要时预取回 GPU
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


# ─────────────────────────────────────────────────────────
# 显存监控
# ─────────────────────────────────────────────────────────

class MemoryTracker:
    """简单的 GPU 显存使用跟踪器"""

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
    估算训练时显存用量（粗略估算）
    
    返回各组成部分的 GB 估算值
    """
    bytes_per_param = 2 if "16" in dtype else 4

    weights_gb = num_params * bytes_per_param / 1e9

    # AdamW 优化器状态（m + v，fp32）
    optimizer_gb = num_params * 4 * 2 / 1e9

    # 激活（取决于是否用 checkpointing）
    act_bytes_per_layer = batch_size * seq_len * hidden_dim * bytes_per_param
    if use_activation_checkpointing:
        # checkpointing 后只需保留每层的输入，大幅减少
        activations_gb = act_bytes_per_layer * num_layers * 0.1 / 1e9
    else:
        activations_gb = act_bytes_per_layer * num_layers / 1e9

    grad_gb = weights_gb  # 梯度与权重同大小

    total_gb = weights_gb + optimizer_gb + activations_gb + grad_gb

    return {
        "weights_gb": round(weights_gb, 2),
        "optimizer_gb": round(optimizer_gb, 2),
        "activations_gb": round(activations_gb, 2),
        "gradients_gb": round(grad_gb, 2),
        "total_estimated_gb": round(total_gb, 2),
        "fits_16gb": total_gb < 15.5,
    }
