"""
TITANS inference-time associative memory.

Treats memory as a small learnable network that updates during inference rather
than a static cache, enabling constant-memory recall over long contexts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, grad


class AssociativeMemoryNetwork(nn.Module):
    """
    """

    def __init__(self, d_model: int, memory_hidden: int = 256):
        super().__init__()
        self.d_model = d_model

        self.W1 = nn.Linear(d_model, memory_hidden, bias=False)
        self.W2 = nn.Linear(memory_hidden, d_model, bias=False)
        nn.init.normal_(self.W1.weight, std=0.02)
        nn.init.zeros_(self.W2.weight)

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.W1(query))
        return self.W2(h)


class SurpriseGate(nn.Module):
    """
    
    
    surprise(x_t) = ||M(x_t) - target_t||^2
    """

    def __init__(self, d_model: int):
        super().__init__()

        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.GELU(),
            nn.Linear(d_model, d_model, bias=False),
        )
        self.threshold = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor, memory_recall: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, d_model)

        """
        expected = self.predictor(x)

        surprise = ((expected - memory_recall) ** 2).mean(dim=-1)  # (B, L)
        return surprise


class MemorySummaryBank(nn.Module):
    """

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


        self.memory_net = AssociativeMemoryNetwork(d_model, memory_hidden)


        self.surprise_gate = SurpriseGate(d_model)
        self.summary_bank = MemorySummaryBank(
            d_model=d_model,
            num_slots=summary_slots,
            momentum=summary_momentum,
            temperature=summary_temperature,
        )


        self.fifo_attn = nn.MultiheadAttention(d_model, num_heads=4,
                                               dropout=0.0, batch_first=True)


        self.fusion_gate = nn.Linear(d_model * 4, 3, bias=True)


        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.RMSNorm(d_model)

    def _write_to_memory(self, x: torch.Tensor, surprise_mask: torch.Tensor):
        """

        x:             (B, L, d_model)
        
        """

        if not surprise_mask.any():
            return



        write_tokens = []
        for b in range(x.shape[0]):
            mask_b = surprise_mask[b]  # (L,)
            if mask_b.any():
                write_tokens.append(x[b][mask_b].mean(dim=0, keepdim=True))

        if not write_tokens:
            return

        write_batch = torch.cat(write_tokens, dim=0)  # (K, d_model)




        with torch.enable_grad():

            for p in self.memory_net.parameters():
                p.requires_grad_(True)

            recall = self.memory_net(write_batch.detach())
            loss = F.mse_loss(recall, write_batch.detach())
            loss.backward()


            with torch.no_grad():
                for p in self.memory_net.parameters():
                    if p.grad is not None:
                        p.data -= self.memory_lr * p.grad
                        p.grad = None

        self.summary_bank.update(write_batch.detach())

    def forward(self, x: torch.Tensor, fifo_buffer: torch.Tensor = None):
        """

        """
        B, L, D = x.shape
        x_norm = self.norm(x)



        x_flat = x_norm.view(B * L, D)
        with torch.no_grad():
            lt_recall = self.memory_net(x_flat).view(B, L, D)
            summary_recall = self.summary_bank.retrieve(x_norm)


        surprise = self.surprise_gate(x_norm, lt_recall)    # (B, L)
        surprise_mask = surprise > self.surprise_threshold  # (B, L) bool


        self._write_to_memory(x_norm, surprise_mask)


        if fifo_buffer is not None and fifo_buffer.shape[1] > 0:

            st_recall, _ = self.fifo_attn(
                query=x_norm,
                key=fifo_buffer,
                value=fifo_buffer,
                need_weights=False
            )
        else:
            st_recall = torch.zeros_like(x_norm)


        x_detach = x_norm.detach()
        if fifo_buffer is None:
            new_fifo = x_detach
        else:
            new_fifo = torch.cat([fifo_buffer, x_detach], dim=1)
            if new_fifo.shape[1] > self.fifo_size:
                new_fifo = new_fifo[:, -self.fifo_size:]


        fusion_input = torch.cat([x_norm, st_recall, lt_recall, summary_recall], dim=-1)
        gates = torch.softmax(self.fusion_gate(fusion_input), dim=-1)


        fused = (
            gates[..., 0:1] * st_recall +
            gates[..., 1:2] * lt_recall +
            gates[..., 2:3] * summary_recall
        )

        output = self.out_proj(fused) + x

        return output, new_fifo
