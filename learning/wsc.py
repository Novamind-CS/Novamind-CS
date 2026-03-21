"""
Weight Space Consolidation (WSC).

Addresses catastrophic forgetting in continual learning with two mechanisms:
SVD-based low-rank resets to recover plasticity and EMA weight averaging to
preserve stability.
"""

import torch
import torch.nn as nn
from copy import deepcopy
from typing import Dict, Optional
import math


class WSCOptimizer:
    """
    
    
        base_opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
        wsc = WSCOptimizer(model, base_opt, reset_freq=500, ema_decay=0.999)
        

        loss.backward()
        wsc.step()
        wsc.zero_grad()
    """

    def __init__(self,
                 model: nn.Module,
                 base_optimizer: torch.optim.Optimizer,
                 reset_freq: int = 500,
                 reset_ratio: float = 0.05,
                 ema_decay: float = 0.999,
                 min_rank_ratio: float = 0.1):
        self.model = model
        self.base_optimizer = base_optimizer
        self.reset_freq = reset_freq
        self.reset_ratio = reset_ratio
        self.ema_decay = ema_decay
        self.min_rank_ratio = min_rank_ratio

        self.step_count = 0


        self.ema_model = deepcopy(model)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)


        self.grad_accum: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.dim() >= 2:
                self.grad_accum[name] = torch.zeros_like(param.data)

    @torch.no_grad()
    def _update_ema(self):
        decay = self.ema_decay
        for (name, param), (_, ema_param) in zip(
            self.model.named_parameters(),
            self.ema_model.named_parameters()
        ):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

    @torch.no_grad()
    def _accumulate_grad_signal(self):
        for name, param in self.model.named_parameters():
            if name in self.grad_accum and param.grad is not None:

                self.grad_accum[name] += (param.grad.abs() * param.data.abs()).detach()

    @torch.no_grad()
    def _svd_reset_low_rank(self):
        """
        
        W = U @ S @ V^T
        """
        reset_count = 0
        total_count = 0

        for name, param in self.model.named_parameters():
            if name not in self.grad_accum:
                continue
            if param.dim() < 2:
                continue


            original_shape = param.shape
            W = param.data.view(original_shape[0], -1).float()

            if W.shape[0] < 4 or W.shape[1] < 4:
                continue

            try:
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            except Exception:
                continue

            total_rank = S.shape[0]
            min_rank = max(1, int(total_rank * self.min_rank_ratio))


            grad_signal = self.grad_accum[name].view(original_shape[0], -1).float()

            rank_importance = S * (S / (S.sum() + 1e-8))


            n_reset = max(0, int(total_rank * self.reset_ratio))
            n_keep = max(min_rank, total_rank - n_reset)

            if n_keep < total_rank:

                noise_scale = S[n_keep - 1].item() * 0.1
                S[n_keep:] = torch.randn_like(S[n_keep:]) * noise_scale
                reset_count += (total_rank - n_keep)

            total_count += total_rank


            W_new = (U @ torch.diag(S) @ Vh)
            param.data.copy_(W_new.view(original_shape).to(param.dtype))


        for k in self.grad_accum:
            self.grad_accum[k].zero_()

        return reset_count, total_count

    @torch.no_grad()
    def _weight_space_average(self):
        """
        
        """
        alpha = 1.0 - self.ema_decay
        for (name, param), (_, ema_param) in zip(
            self.model.named_parameters(),
            self.ema_model.named_parameters()
        ):
            if param.requires_grad:
                param.data.lerp_(ema_param.data, alpha * 0.1)

    def step(self):
        self._accumulate_grad_signal()
        self.base_optimizer.step()
        self.step_count += 1


        self._update_ema()


        if self.step_count % self.reset_freq == 0:
            reset_count, total_count = self._svd_reset_low_rank()
            self._weight_space_average()
            print(f"[WSC] Step {self.step_count}: "
                  f"Reset {reset_count}/{total_count} singular directions "
                  f"({100*reset_count/max(total_count,1):.1f}%)")

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def state_dict(self):
        return {
            "base_optimizer": self.base_optimizer.state_dict(),
            "step_count": self.step_count,
            "ema_model": self.ema_model.state_dict(),
            "grad_accum": {k: v.cpu() for k, v in self.grad_accum.items()},
        }

    def load_state_dict(self, state: dict):
        self.base_optimizer.load_state_dict(state["base_optimizer"])
        self.step_count = state["step_count"]
        self.ema_model.load_state_dict(state["ema_model"])
        for k, v in state["grad_accum"].items():
            if k in self.grad_accum:
                self.grad_accum[k].copy_(v)

    @property
    def ema_weights(self) -> nn.Module:
        return self.ema_model
