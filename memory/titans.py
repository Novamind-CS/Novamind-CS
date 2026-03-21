"""
NovaMind — TITANS 推断期联想记忆

核心思想: 记忆不是静态缓存，而是一个在推断时可以学习的小神经网络

工作流程:
    1. 输入经过惊奇度评估 (Surprise metric)
    2. 惊奇度高的信息写入联想记忆网络权重
    3. 查询时直接从权重中提取关联信息
    4. 整个过程不改变主模型权重，只更新记忆网络

相比标准 KV Cache:
- KV Cache: 内存 O(L)，精确但有界
- TITANS:   内存 O(1)，基于关联度召回，支持无限长上下文
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad


class AssociativeMemoryNetwork(nn.Module):
    """
    联想记忆网络 — 一个超小型 MLP，权重就是记忆的载体
    在推断时通过梯度更新来"记住"新信息
    """

    def __init__(self, d_model: int, memory_hidden: int = 256):
        super().__init__()
        self.d_model = d_model
        # 故意设计成很小的网络（记忆容量有限但高效）
        self.W1 = nn.Linear(d_model, memory_hidden, bias=False)
        self.W2 = nn.Linear(memory_hidden, d_model, bias=False)
        nn.init.normal_(self.W1.weight, std=0.02)
        nn.init.zeros_(self.W2.weight)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        """query: (B, d_model) → 联想召回结果 (B, d_model)"""
        h = F.gelu(self.W1(query))
        return self.W2(h)


class SurpriseGate(nn.Module):
    """
    惊奇度评估器
    
    衡量当前输入 x_t 与记忆网络 M 的预期之间的差距
    差距越大 = 越"惊奇" = 越值得写入记忆
    
    surprise(x_t) = ||M(x_t) - target_t||^2
    """

    def __init__(self, d_model: int):
        super().__init__()
        # 预测器：从当前输入预测"应该召回什么"
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.GELU(),
            nn.Linear(d_model, d_model, bias=False),
        )
        self.threshold = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, memory_recall: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, d_model)
        memory_recall: (B, L, d_model) — 记忆网络当前的召回结果

        返回: surprise_scores (B, L) — 越高越值得写入
        """
        expected = self.predictor(x)
        # 均方误差作为惊奇度
        surprise = ((expected - memory_recall) ** 2).mean(dim=-1)  # (B, L)
        return surprise


class MemorySummaryBank(nn.Module):
    """
    固定容量的多时间尺度记忆摘要库。

    用少量原型向量压缩长期经验，补足联想网络只靠权重召回时
    对稀有离散事实不够稳定的问题。
    """

    def __init__(self, d_model: int, num_slots: int = 32,
                 momentum: float = 0.95, temperature: float = 0.35):
        super().__init__()
        self.num_slots = num_slots
        self.momentum = momentum
        self.temperature = temperature

        self.register_buffer("slots", torch.zeros(num_slots, d_model))
        self.register_buffer("usage", torch.zeros(num_slots))

    def retrieve(self, x: torch.Tensor) -> torch.Tensor:
        if self.usage.sum() <= 0:
            return torch.zeros_like(x)

        slots = F.normalize(self.slots, dim=-1)
        queries = F.normalize(x, dim=-1)
        scores = torch.einsum("bld,sd->bls", queries, slots) / self.temperature
        weights = torch.softmax(scores, dim=-1)
        return torch.einsum("bls,sd->bld", weights, self.slots)

    @torch.no_grad()
    def update(self, write_batch: torch.Tensor):
        if write_batch.numel() == 0:
            return

        for token in write_batch:
            if self.usage.sum() <= 0:
                slot_idx = 0
            else:
                sims = F.cosine_similarity(
                    token.unsqueeze(0), self.slots.to(token.dtype), dim=-1
                )
                empty_slots = (self.usage == 0).nonzero(as_tuple=False)
                if empty_slots.numel() > 0:
                    slot_idx = empty_slots[0, 0].item()
                else:
                    slot_idx = torch.argmax(sims).item()

            self.slots[slot_idx].mul_(self.momentum).add_(
                token.detach(), alpha=1.0 - self.momentum
            )
            self.usage[slot_idx] += 1


class TITANSMemoryModule(nn.Module):
    """
    完整的 TITANS 记忆模块
    
    在每个 forward 中:
    1. 短期记忆：FIFO 队列缓存最近 fifo_size 个 token
    2. 长期记忆：联想网络，通过推断期梯度更新写入
    3. 融合：短期 + 长期 + 当前输入 → 最终输出
    """

    def __init__(self, d_model: int, fifo_size: int = 512,
                 memory_hidden: int = 256, memory_lr: float = 0.01,
                 surprise_threshold: float = 0.1,
                 summary_slots: int = 32,
                 summary_momentum: float = 0.95,
                 summary_temperature: float = 0.35):
        super().__init__()
        self.d_model = d_model
        self.fifo_size = fifo_size
        self.memory_lr = memory_lr
        self.surprise_threshold = surprise_threshold

        # 长期联想记忆网络
        self.memory_net = AssociativeMemoryNetwork(d_model, memory_hidden)

        # 惊奇度门控
        self.surprise_gate = SurpriseGate(d_model)
        self.summary_bank = MemorySummaryBank(
            d_model=d_model,
            num_slots=summary_slots,
            momentum=summary_momentum,
            temperature=summary_temperature,
        )

        # 短期 FIFO 缓冲的注意力（局部窗口）
        self.fifo_attn = nn.MultiheadAttention(d_model, num_heads=4,
                                               dropout=0.0, batch_first=True)

        # 融合门：决定短期记忆与长期记忆的比例
        self.fusion_gate = nn.Linear(d_model * 4, 3, bias=True)

        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.RMSNorm(d_model)

    def _write_to_memory(self, x: torch.Tensor, surprise_mask: torch.Tensor):
        """
        对惊奇度高的 token，通过梯度步骤将其写入联想记忆

        x:             (B, L, d_model)
        surprise_mask: (B, L) bool — True 表示需要写入
        
        注意：这是推断期学习的核心，在 no_grad 上下文外执行
        """
        # 只处理有惊奇内容的 batch
        if not surprise_mask.any():
            return

        # 提取需要写入的 token
        # 简化：用平均池化（实际可以用更复杂的写入策略）
        write_tokens = []
        for b in range(x.shape[0]):
            mask_b = surprise_mask[b]  # (L,)
            if mask_b.any():
                write_tokens.append(x[b][mask_b].mean(dim=0, keepdim=True))

        if not write_tokens:
            return

        write_batch = torch.cat(write_tokens, dim=0)  # (K, d_model)

        # 推断期梯度更新（写入）
        # 目标：记忆网络能从 key 中召回 value
        # 这里用自编码目标：M(x) ≈ x
        with torch.enable_grad():
            # 临时为记忆网络参数开启梯度
            for p in self.memory_net.parameters():
                p.requires_grad_(True)

            recall = self.memory_net(write_batch.detach())
            loss = F.mse_loss(recall, write_batch.detach())
            loss.backward()

            # 手动梯度步骤（不影响主模型优化器）
            with torch.no_grad():
                for p in self.memory_net.parameters():
                    if p.grad is not None:
                        p.data -= self.memory_lr * p.grad
                        p.grad = None

        self.summary_bank.update(write_batch.detach())

    def forward(self, x: torch.Tensor, fifo_buffer: torch.Tensor = None):
        """
        x:           (B, L, d_model) — 当前输入
        fifo_buffer: (B, S, d_model) — 短期 FIFO 缓冲，S ≤ fifo_size

        返回: output (B, L, d_model), 更新后的 fifo_buffer
        """
        B, L, D = x.shape
        x_norm = self.norm(x)

        # ① 长期记忆召回
        # 将序列展平后批量查询
        x_flat = x_norm.view(B * L, D)
        with torch.no_grad():
            lt_recall = self.memory_net(x_flat).view(B, L, D)
            summary_recall = self.summary_bank.retrieve(x_norm)

        # ② 惊奇度评估 → 决定写入哪些 token
        surprise = self.surprise_gate(x_norm, lt_recall)    # (B, L)
        surprise_mask = surprise > self.surprise_threshold  # (B, L) bool

        # ③ 写入高惊奇度 token 到长期记忆
        self._write_to_memory(x_norm, surprise_mask)

        # ④ 短期记忆（FIFO 局部注意力）
        if fifo_buffer is not None and fifo_buffer.shape[1] > 0:
            # 用 FIFO 中的历史 token 作为 KV，当前 x 作为 Query
            st_recall, _ = self.fifo_attn(
                query=x_norm,
                key=fifo_buffer,
                value=fifo_buffer,
                need_weights=False
            )
        else:
            st_recall = torch.zeros_like(x_norm)

        # ⑤ 更新 FIFO 缓冲（FIFO: 先进先出）
        x_detach = x_norm.detach()
        if fifo_buffer is None:
            new_fifo = x_detach
        else:
            new_fifo = torch.cat([fifo_buffer, x_detach], dim=1)
            if new_fifo.shape[1] > self.fifo_size:
                new_fifo = new_fifo[:, -self.fifo_size:]  # 保留最近的

        # ⑥ 融合：短期 + 长期 + 原始输入
        fusion_input = torch.cat([x_norm, st_recall, lt_recall, summary_recall], dim=-1)
        gates = torch.softmax(self.fusion_gate(fusion_input), dim=-1)

        # 加权融合
        fused = (
            gates[..., 0:1] * st_recall +
            gates[..., 1:2] * lt_recall +
            gates[..., 2:3] * summary_recall
        )

        output = self.out_proj(fused) + x  # 残差连接

        return output, new_fifo
