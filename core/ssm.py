"""
NovaMind — 选择性状态空间模型 (Mamba-2 风格)

核心方程:
    h_t = A_t * h_{t-1} + B_t * x_t
    y_t = C_t^T * h_t

关键点:
- A_t 是对角矩阵 (高效) + 输入选择性 (不同 token 有不同的遗忘率)
- 训练时并行扫描 (parallel scan)，推理时递归 (O(1) 内存)
- 与 Mamba-2 的结构化状态矩阵分解对齐
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as mamba2_selective_scan_fn
except Exception:
    mamba2_selective_scan_fn = None


def selective_scan_sequential(x, dt, A, B, C, D,
                              initial_state=None,
                              return_final_state: bool = False):
    """
    顺序扫描实现 (推理用，O(1) 内存)
    
    x:  (B, L, d_inner)
    dt: (B, L, d_inner)    — 离散化步长，输入依赖
    A:  (d_inner, d_state)  — 对角衰减矩阵（log 参数化）
    B:  (B, L, d_state)    — 输入投影
    C:  (B, L, d_state)    — 输出投影
    D:  (d_inner,)          — 跳跃连接
    
    返回: (B, L, d_inner)
    """
    B_batch, L, d_inner = x.shape
    d_state = A.shape[-1]

    # 离散化 A, B (Zero-order hold)
    # dA: (B, L, d_inner, d_state)
    dt = F.softplus(dt)  # 确保正值
    dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
    dB = dt.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, d_inner, d_state)

    # 顺序递归
    if initial_state is None:
        h = torch.zeros(B_batch, d_inner, d_state, device=x.device, dtype=x.dtype)
    else:
        h = initial_state
    ys = []
    for t in range(L):
        # h_t = dA_t * h_{t-1} + dB_t * x_t
        h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
        # y_t = C_t^T * h_t
        y = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # (B, d_inner)
        ys.append(y)

    y = torch.stack(ys, dim=1)   # (B, L, d_inner)
    y = y + x * D.unsqueeze(0).unsqueeze(0)
    if return_final_state:
        return y, h
    return y


def selective_scan_parallel(x, dt, A, B, C, D):
    """
    并行扫描实现 (训练用，GPU 高效)
    使用 blelloch scan 的简化版本。
    实际生产中替换为 cuda-optimized mamba_ssm.ops.selective_scan_interface
    
    当前实现: 朴素循环但使用 torch.compile 可自动并行化
    """
    # 对于完整的 CUDA 并行实现，建议直接调用:
    # from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    # 这里提供不依赖额外包的参考实现
    return selective_scan_sequential(x, dt, A, B, C, D)


class SelectiveSSM(nn.Module):
    """
    单个 SSM 头
    
    实现要点:
    1. A 以 log 形式参数化，确保稳定性 (A < 0 → 衰减)
    2. B, C, dt 都是输入依赖的（这是"选择性"的核心）
    3. D 是简单的跳跃连接
    """

    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4,
                 expand: int = 2, backend: str = "auto"):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        self.backend = backend

        # 输入投影
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 局部卷积（捕获短程依赖）
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True
        )

        # SSM 参数（输入依赖）
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        # A: 固定结构，log 参数化
        A = torch.arange(1, d_state + 1).unsqueeze(0).repeat(self.d_inner, 1).float()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # D: 跳跃连接
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.RMSNorm(self.d_inner)

    def forward(self, x: torch.Tensor,
                inference_mode: bool = False,
                state: torch.Tensor = None,
                return_state: bool = False):
        """
        x: (B, L, d_model)
        返回: (B, L, d_model)
        """
        B, L, _ = x.shape

        # 输入投影 + 门控
        xz = self.in_proj(x)          # (B, L, 2 * d_inner)
        x_ssm, z = xz.chunk(2, dim=-1)  # 各 (B, L, d_inner)

        # 局部卷积（因果）
        x_conv = x_ssm.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[..., :L]   # 截断保因果
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)

        # 计算 B, C, dt（全部输入依赖）
        x_dbc = self.x_proj(x_conv)   # (B, L, d_state*2 + d_inner)
        B_ssm, C_ssm, dt_raw = x_dbc.split(
            [self.d_state, self.d_state, self.d_inner], dim=-1
        )
        dt = self.dt_proj(dt_raw)     # (B, L, d_inner)

        A = -torch.exp(self.A_log)    # 确保 A < 0 → 稳定衰减

        # 状态空间扫描
        if inference_mode or state is not None or return_state:
            y, next_state = selective_scan_sequential(
                x_conv, dt, A, B_ssm, C_ssm, self.D,
                initial_state=state,
                return_final_state=True
            )
        elif self.backend in ("auto", "mamba2") and mamba2_selective_scan_fn is not None:
            y = mamba2_selective_scan_fn(x_conv, dt, A, B_ssm, C_ssm, self.D)
            next_state = None
        else:
            y = selective_scan_parallel(x_conv, dt, A, B_ssm, C_ssm, self.D)
            next_state = None

        # 门控输出
        y = self.norm(y) * F.silu(z)
        y = self.out_proj(y)
        if return_state:
            return y, next_state
        return y


class MultiHeadSSM(nn.Module):
    """
    多头 SSM — 并行运行多个 SSM 头，类似多头注意力
    不同头可以学习不同时间尺度的依赖
    """

    def __init__(self, d_model: int, num_heads: int, d_state: int = 64,
                 d_conv: int = 4, expand: int = 2, backend: str = "auto"):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.heads = nn.ModuleList([
            SelectiveSSM(self.head_dim, d_state, d_conv, expand, backend=backend)
            for _ in range(num_heads)
        ])
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor,
                inference_mode: bool = False,
                states=None,
                return_states: bool = False):
        """x: (B, L, d_model)"""
        # 拆分到各头
        chunks = x.chunk(self.num_heads, dim=-1)  # num_heads × (B, L, head_dim)
        if states is None:
            states = [None] * self.num_heads

        outputs = []
        new_states = []
        for head, chunk, state in zip(self.heads, chunks, states):
            if return_states:
                out, next_state = head(
                    chunk,
                    inference_mode=inference_mode,
                    state=state,
                    return_state=True
                )
                new_states.append(next_state)
            else:
                out = head(chunk, inference_mode=inference_mode, state=state)
            outputs.append(out)

        y = self.out_proj(torch.cat(outputs, dim=-1))
        if return_states:
            return y, new_states
        return y
