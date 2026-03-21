"""
NovaMind — xLSTM 矩阵记忆

核心方程:
    C_t = f_t * C_{t-1} + i_t * (k_t ⊗ v_t)   ← 矩阵记忆更新
    h_t = o_t * (C_t * q_t) / max(|n_t^T q_t|, 1)  ← 归一化召回

相比标准 LSTM:
- 隐藏状态从向量升级为矩阵，存储高秩信息
- 指数门控（exp gating）提升数值稳定性
- 协方差更新规则类似注意力但代价是 O(N)

相比注意力机制:
- 不需要保存所有历史 KV，O(1) 内存
- 通过矩阵记忆实现类注意力的精确召回
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class mLSTMCell(nn.Module):
    """
    单个 mLSTM 单元（矩阵 LSTM）
    
    d_model: 输入/输出维度
    d_head:  矩阵记忆的 key/value 维度
    num_heads: 并行记忆头数
    """

    def __init__(self, d_model: int, d_head: int = 64, num_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.num_heads = num_heads
        self.d_inner = d_head * num_heads

        # Q, K, V 投影
        self.q_proj = nn.Linear(d_model, self.d_inner, bias=False)
        self.k_proj = nn.Linear(d_model, self.d_inner, bias=False)
        self.v_proj = nn.Linear(d_model, self.d_inner, bias=False)

        # 门控：输入门 i, 遗忘门 f, 输出门 o
        # 使用指数参数化提升稳定性（xLSTM 核心改进）
        self.i_gate = nn.Linear(d_model, num_heads, bias=True)   # 标量门，每头一个
        self.f_gate = nn.Linear(d_model, num_heads, bias=True)   # 标量门，每头一个
        self.o_gate = nn.Linear(d_model, self.d_inner, bias=True)

        # 输出归一化与投影
        self.group_norm = nn.GroupNorm(num_heads, self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # 初始化遗忘门偏置为正（高遗忘率 → 稳定训练初期）
        nn.init.constant_(self.f_gate.bias, 3.0)

    def forward(self, x: torch.Tensor,
                C_prev: torch.Tensor,
                n_prev: torch.Tensor,
                m_prev: torch.Tensor):
        """
        x:      (B, L, d_model)    — 输入序列
        C_prev: (B, H, d_head, d_head)  — 矩阵记忆
        n_prev: (B, H, d_head)     — 归一化向量
        m_prev: (B, H)             — 数值稳定最大值（log-sum-exp trick）
        
        返回: output (B, L, d_model), 更新后的 (C, n, m)
        """
        B, L, _ = x.shape
        H = self.num_heads
        d = self.d_head

        # 投影 Q, K, V
        q = self.q_proj(x).view(B, L, H, d)   # (B, L, H, d)
        k = self.k_proj(x).view(B, L, H, d)
        v = self.v_proj(x).view(B, L, H, d)

        # 门控值（指数化，xLSTM 的核心技巧）
        # log(i) 和 log(f) 直接作为 logit，用 log-sum-exp 稳定
        log_i = self.i_gate(x)   # (B, L, H)
        log_f = self.f_gate(x)   # (B, L, H)
        o = torch.sigmoid(self.o_gate(x)).view(B, L, H, d)  # (B, L, H, d)

        # 数值稳定化的最大值更新
        m_new_candidates = torch.maximum(log_f + m_prev.unsqueeze(1), log_i)  # (B, L, H)

        outputs = []
        C, n, m = C_prev, n_prev, m_prev

        for t in range(L):
            # 稳定化后的门控值
            m_t = m_new_candidates[:, t]            # (B, H)
            f_t = torch.exp(log_f[:, t] + m - m_t)  # 遗忘门，归一化
            i_t = torch.exp(log_i[:, t] - m_t)      # 输入门，归一化
            m = m_t

            k_t = k[:, t] / math.sqrt(d)   # (B, H, d) — 缩放 key

            # 矩阵记忆更新: C_t = f_t * C_{t-1} + i_t * (v_t ⊗ k_t)
            # (B, H, d, d) = f * (B, H, d, d) + i * outer(v, k)
            outer = torch.einsum("bhd,bhe->bhde", v[:, t], k_t)  # (B, H, d, d)
            C = (f_t[:, :, None, None] * C +
                 i_t[:, :, None, None] * outer)

            # 归一化向量更新: n_t = f_t * n_{t-1} + i_t * k_t
            n = f_t[:, :, None] * n + i_t[:, :, None] * k_t

            # 召回: h_t = o_t * (C_t @ q_t) / max(|n_t^T q_t|, 1)
            recall = torch.einsum("bhde,bhe->bhd", C, q[:, t])  # (B, H, d)
            denom = torch.clamp(
                (n * q[:, t]).sum(dim=-1, keepdim=True).abs(),
                min=1.0
            )  # (B, H, 1)
            h_t = o[:, t] * (recall / denom)   # (B, H, d)
            outputs.append(h_t)

        # 拼接输出
        out = torch.stack(outputs, dim=1)         # (B, L, H, d)
        out = out.reshape(B, L, H * d)             # (B, L, d_inner)
        out = self.group_norm(out.transpose(1, 2)).transpose(1, 2)
        out = F.gelu(out)

        return self.out_proj(out), C, n, m

    def init_state(self, batch_size: int, device, dtype):
        """初始化零状态"""
        H, d = self.num_heads, self.d_head
        return (
            torch.zeros(batch_size, H, d, d, device=device, dtype=dtype),  # C
            torch.zeros(batch_size, H, d, device=device, dtype=dtype),      # n
            torch.zeros(batch_size, H, device=device, dtype=dtype),         # m
        )


class xLSTMBlock(nn.Module):
    """
    完整的 xLSTM block，包含残差连接和 RMS 归一化
    可以独立替换任意 Transformer block 中的 FFN
    """

    def __init__(self, d_model: int, d_head: int = 64, num_heads: int = 4,
                 expand_ffn: int = 4):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model)
        self.mlstm = mLSTMCell(d_model, d_head, num_heads)

        # 可选的 FFN（用于特征变换）
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
        state: 可选的 (C, n, m) 元组，用于推理时的状态延续
        """
        B = x.shape[0]
        device, dtype = x.device, x.dtype

        if state is None:
            state = self.mlstm.init_state(B, device, dtype)

        C, n, m = state

        # mLSTM 子层 + 残差
        mlstm_out, C_new, n_new, m_new = self.mlstm(self.norm1(x), C, n, m)
        x = x + mlstm_out

        # FFN 子层 + 残差
        x = x + self.ffn(self.norm2(x))

        return x, (C_new, n_new, m_new)
