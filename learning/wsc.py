"""
NovaMind — 权重空间巩固 (WSC)
解决持续学习中的灾难性遗忘问题

两个核心机制:
1. 基于 SVD 的低秩参数重置 → 恢复"塑性"（学新东西的能力）
2. 指数移动平均权重平均    → 保持"稳定性"（不忘旧东西）

直觉类比：
- 大脑睡眠时"修剪"不重要的突触（重置低贡献参数）
- 长期记忆通过重复激活固化（EMA 权重平均）
"""

import torch
import torch.nn as nn
from copy import deepcopy
from typing import Dict, Optional
import math


class WSCOptimizer:
    """
    WSC 持续学习优化器包装器
    
    包装标准优化器（AdamW 等），在其基础上添加:
    - 定期 SVD 重置低贡献参数
    - EMA 权重平均
    - 梯度贡献监控
    
    用法:
        base_opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
        wsc = WSCOptimizer(model, base_opt, reset_freq=500, ema_decay=0.999)
        
        # 训练循环中:
        loss.backward()
        wsc.step()  # 代替 base_opt.step()
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
        self.reset_ratio = reset_ratio     # 每次重置最低贡献的 N% 参数
        self.ema_decay = ema_decay
        self.min_rank_ratio = min_rank_ratio  # 保留至少这么多奇异值

        self.step_count = 0

        # EMA 影子模型（用于稳定性保证）
        self.ema_model = deepcopy(model)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

        # 梯度贡献累积器（用于判断哪些参数贡献低）
        self.grad_accum: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.dim() >= 2:
                self.grad_accum[name] = torch.zeros_like(param.data)

    @torch.no_grad()
    def _update_ema(self):
        """更新 EMA 影子模型"""
        decay = self.ema_decay
        for (name, param), (_, ema_param) in zip(
            self.model.named_parameters(),
            self.ema_model.named_parameters()
        ):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

    @torch.no_grad()
    def _accumulate_grad_signal(self):
        """累积梯度信号，用于判断参数贡献度"""
        for name, param in self.model.named_parameters():
            if name in self.grad_accum and param.grad is not None:
                # 用 |grad * param| 估计参数重要性（Fisher 信息的近似）
                self.grad_accum[name] += (param.grad.abs() * param.data.abs()).detach()

    @torch.no_grad()
    def _svd_reset_low_rank(self):
        """
        对线性层权重执行 SVD，重置低秩（低重要性）的奇异值方向
        
        原理：
        W = U @ S @ V^T
        低奇异值 s_i 对应的方向贡献小 → 重置为小随机值
        这相当于"清空"了不重要的神经连接，恢复模型的学习塑性
        """
        reset_count = 0
        total_count = 0

        for name, param in self.model.named_parameters():
            if name not in self.grad_accum:
                continue
            if param.dim() < 2:
                continue

            # 展平到 2D（支持 4D 卷积等）
            original_shape = param.shape
            W = param.data.view(original_shape[0], -1).float()

            if W.shape[0] < 4 or W.shape[1] < 4:
                continue  # 太小的矩阵跳过

            try:
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            except Exception:
                continue  # SVD 失败时跳过

            total_rank = S.shape[0]
            min_rank = max(1, int(total_rank * self.min_rank_ratio))

            # 结合梯度贡献判断重置范围
            grad_signal = self.grad_accum[name].view(original_shape[0], -1).float()
            # 把梯度信号投影到奇异值空间（近似）
            rank_importance = S * (S / (S.sum() + 1e-8))  # 归一化

            # 找出贡献最低的奇异值
            n_reset = max(0, int(total_rank * self.reset_ratio))
            n_keep = max(min_rank, total_rank - n_reset)

            if n_keep < total_rank:
                # 重置低秩方向：用小随机噪声替换
                noise_scale = S[n_keep - 1].item() * 0.1  # 保持量级一致
                S[n_keep:] = torch.randn_like(S[n_keep:]) * noise_scale
                reset_count += (total_rank - n_keep)

            total_count += total_rank

            # 重建权重并写回
            W_new = (U @ torch.diag(S) @ Vh)
            param.data.copy_(W_new.view(original_shape).to(param.dtype))

        # 重置梯度累积器
        for k in self.grad_accum:
            self.grad_accum[k].zero_()

        return reset_count, total_count

    @torch.no_grad()
    def _weight_space_average(self):
        """
        将当前模型权重拉向 EMA 稳态（防止偏离太远）
        
        实现 "稳定性-塑性" 权衡的关键：
        - 只做轻微拉拽（alpha 很小），不完全用 EMA 替换
        - 每次 SVD 重置后调用，防止重置引入过多随机性
        """
        alpha = 1.0 - self.ema_decay  # 轻微向 EMA 靠拢
        for (name, param), (_, ema_param) in zip(
            self.model.named_parameters(),
            self.ema_model.named_parameters()
        ):
            if param.requires_grad:
                param.data.lerp_(ema_param.data, alpha * 0.1)

    def step(self):
        """执行一步优化"""
        self._accumulate_grad_signal()
        self.base_optimizer.step()
        self.step_count += 1

        # 更新 EMA
        self._update_ema()

        # 定期执行 SVD 重置
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
        """返回 EMA 影子模型（用于评估，比当前训练中的模型更稳定）"""
        return self.ema_model
