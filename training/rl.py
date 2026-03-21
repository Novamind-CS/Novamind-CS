"""
NovaMind — 轻量 RL / reward-weighted objective
"""

import torch
import torch.nn.functional as F


def reward_weighted_ce_loss(logits: torch.Tensor,
                            labels: torch.Tensor,
                            rewards: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    token_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.shape[-1]),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)

    valid_mask = (shift_labels != -100).float()
    seq_loss = (token_loss * valid_mask).sum(dim=-1) / valid_mask.sum(dim=-1).clamp(min=1.0)
    normalized_reward = rewards / rewards.abs().mean().clamp(min=1.0)
    return (seq_loss * normalized_reward).mean()
