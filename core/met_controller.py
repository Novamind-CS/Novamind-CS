"""
NovaMind — Metacognitive Entropy Throttling.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


def calculate_entropy(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits.float(), dim=-1)
    probs = probs.clamp_min(1e-12)
    entropy = -(probs * probs.log()).sum(dim=-1)
    return float(entropy.mean().item())


@dataclass
class METGate:
    entropy_threshold: float = 3.5

    def should_trigger_system2(self, logits: torch.Tensor) -> bool:
        return calculate_entropy(logits) > self.entropy_threshold

    def inspect(self, logits: torch.Tensor) -> dict:
        entropy = calculate_entropy(logits)
        return {
            "entropy": entropy,
            "trigger_system2": entropy > self.entropy_threshold,
        }


@dataclass
class MetStateTracker:
    entropy_threshold: float = 3.5
    caution_window: int = 5
    remaining_system2_steps: int = 0
    system1_tokens: int = 0
    system2_tokens: int = 0

    def __post_init__(self):
        self.gate = METGate(self.entropy_threshold)

    def reset(self):
        self.remaining_system2_steps = 0
        self.system1_tokens = 0
        self.system2_tokens = 0

    def observe(self, logits: torch.Tensor) -> dict:
        entropy = calculate_entropy(logits)
        high_entropy = entropy > self.entropy_threshold
        forced_by_inertia = self.remaining_system2_steps > 0 and not high_entropy

        if high_entropy:
            self.remaining_system2_steps = self.caution_window

        trigger_system2 = high_entropy or self.remaining_system2_steps > 0
        if trigger_system2:
            self.system2_tokens += 1
            self.remaining_system2_steps = max(0, self.remaining_system2_steps - 1)
        else:
            self.system1_tokens += 1

        return {
            "entropy": entropy,
            "trigger_system2": trigger_system2,
            "forced_by_inertia": forced_by_inertia,
            "remaining_system2_steps": self.remaining_system2_steps,
        }

    def compression_ratio(self) -> dict:
        total = self.system1_tokens + self.system2_tokens
        if total == 0:
            return {
                "system1_tokens": 0,
                "system2_tokens": 0,
                "system1_pct": 0.0,
                "system2_pct": 0.0,
            }
        return {
            "system1_tokens": self.system1_tokens,
            "system2_tokens": self.system2_tokens,
            "system1_pct": 100.0 * self.system1_tokens / total,
            "system2_pct": 100.0 * self.system2_tokens / total,
        }
