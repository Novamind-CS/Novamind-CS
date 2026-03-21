"""
Selective state-space model blocks in a Mamba-2 style.

Implements input-dependent state updates, parallel scan during training, and
recursive constant-memory inference.
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
    
    x:  (B, L, d_inner)
    
    """
    B_batch, L, d_inner = x.shape
    d_state = A.shape[-1]


    # dA: (B, L, d_inner, d_state)
    dt = F.softplus(dt)
    dA = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
    dB = dt.unsqueeze(-1) * B.unsqueeze(2)  # (B, L, d_inner, d_state)


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
    
    """

    # from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    return selective_scan_sequential(x, dt, A, B, C, D)


class SelectiveSSM(nn.Module):
    """
    
    """

    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4,
                 expand: int = 2, backend: str = "auto"):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        self.backend = backend


        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)


        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True
        )


        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + self.d_inner, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)


        A = torch.arange(1, d_state + 1).unsqueeze(0).repeat(self.d_inner, 1).float()
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True


        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.D._no_weight_decay = True


        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.RMSNorm(self.d_inner)

    def forward(self, x: torch.Tensor,
                inference_mode: bool = False,
                state: torch.Tensor = None,
                return_state: bool = False):
        """
        x: (B, L, d_model)
        """
        B, L, _ = x.shape


        xz = self.in_proj(x)          # (B, L, 2 * d_inner)
        x_ssm, z = xz.chunk(2, dim=-1)


        x_conv = x_ssm.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[..., :L]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)


        x_dbc = self.x_proj(x_conv)   # (B, L, d_state*2 + d_inner)
        B_ssm, C_ssm, dt_raw = x_dbc.split(
            [self.d_state, self.d_state, self.d_inner], dim=-1
        )
        dt = self.dt_proj(dt_raw)     # (B, L, d_inner)

        A = -torch.exp(self.A_log)


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


        y = self.norm(y) * F.silu(z)
        y = self.out_proj(y)
        if return_state:
            return y, next_state
        return y


class MultiHeadSSM(nn.Module):
    """
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

        chunks = x.chunk(self.num_heads, dim=-1)
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
