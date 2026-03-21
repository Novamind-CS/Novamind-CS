"""
xLSTM matrix memory blocks.

Upgrades recurrent hidden state from a vector to a matrix, enabling higher-rank
memory with constant-memory inference and attention-like recall behavior.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class mLSTMCell(nn.Module):
    """
    
    """

    def __init__(self, d_model: int, d_head: int = 64, num_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.num_heads = num_heads
        self.d_inner = d_head * num_heads


        self.q_proj = nn.Linear(d_model, self.d_inner, bias=False)
        self.k_proj = nn.Linear(d_model, self.d_inner, bias=False)
        self.v_proj = nn.Linear(d_model, self.d_inner, bias=False)



        self.i_gate = nn.Linear(d_model, num_heads, bias=True)
        self.f_gate = nn.Linear(d_model, num_heads, bias=True)
        self.o_gate = nn.Linear(d_model, self.d_inner, bias=True)


        self.group_norm = nn.GroupNorm(num_heads, self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)


        nn.init.constant_(self.f_gate.bias, 3.0)

    def forward(self, x: torch.Tensor,
                C_prev: torch.Tensor,
                n_prev: torch.Tensor,
                m_prev: torch.Tensor):
        """
        
        """
        B, L, _ = x.shape
        H = self.num_heads
        d = self.d_head


        q = self.q_proj(x).view(B, L, H, d)   # (B, L, H, d)
        k = self.k_proj(x).view(B, L, H, d)
        v = self.v_proj(x).view(B, L, H, d)



        log_i = self.i_gate(x)   # (B, L, H)
        log_f = self.f_gate(x)   # (B, L, H)
        o = torch.sigmoid(self.o_gate(x)).view(B, L, H, d)  # (B, L, H, d)


        m_new_candidates = torch.maximum(log_f + m_prev.unsqueeze(1), log_i)  # (B, L, H)

        outputs = []
        C, n, m = C_prev, n_prev, m_prev

        for t in range(L):

            m_t = m_new_candidates[:, t]            # (B, H)
            f_t = torch.exp(log_f[:, t] + m - m_t)
            i_t = torch.exp(log_i[:, t] - m_t)
            m = m_t

            k_t = k[:, t] / math.sqrt(d)


            # (B, H, d, d) = f * (B, H, d, d) + i * outer(v, k)
            outer = torch.einsum("bhd,bhe->bhde", v[:, t], k_t)  # (B, H, d, d)
            C = (f_t[:, :, None, None] * C +
                 i_t[:, :, None, None] * outer)


            n = f_t[:, :, None] * n + i_t[:, :, None] * k_t


            recall = torch.einsum("bhde,bhe->bhd", C, q[:, t])  # (B, H, d)
            denom = torch.clamp(
                (n * q[:, t]).sum(dim=-1, keepdim=True).abs(),
                min=1.0
            )  # (B, H, 1)
            h_t = o[:, t] * (recall / denom)   # (B, H, d)
            outputs.append(h_t)


        out = torch.stack(outputs, dim=1)         # (B, L, H, d)
        out = out.reshape(B, L, H * d)             # (B, L, d_inner)
        out = self.group_norm(out.transpose(1, 2)).transpose(1, 2)
        out = F.gelu(out)

        return self.out_proj(out), C, n, m

    def init_state(self, batch_size: int, device, dtype):
        H, d = self.num_heads, self.d_head
        return (
            torch.zeros(batch_size, H, d, d, device=device, dtype=dtype),  # C
            torch.zeros(batch_size, H, d, device=device, dtype=dtype),      # n
            torch.zeros(batch_size, H, device=device, dtype=dtype),         # m
        )


class xLSTMBlock(nn.Module):
    """
    """

    def __init__(self, d_model: int, d_head: int = 64, num_heads: int = 4,
                 expand_ffn: int = 4):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.mlstm = mLSTMCell(d_model, d_head, num_heads)


        self.norm2 = nn.RMSNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * expand_ffn, bias=False),
            nn.GELU(),
            nn.Linear(d_model * expand_ffn, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor,
                state=None):
        """
        x: (B, L, d_model)
        """
        B = x.shape[0]
        device, dtype = x.device, x.dtype

        if state is None:
            state = self.mlstm.init_state(B, device, dtype)

        C, n, m = state


        mlstm_out, C_new, n_new, m_new = self.mlstm(self.norm1(x), C, n, m)
        x = x + mlstm_out


        x = x + self.ffn(self.norm2(x))

        return x, (C_new, n_new, m_new)
