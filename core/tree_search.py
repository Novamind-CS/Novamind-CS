"""
NovaMind — 测试时树状搜索

用于 CATTS 的 deep 模式，把“多次抖动平均”替换成真正的路径搜索。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
import copy

import torch
import torch.nn.functional as F


@dataclass
class SearchNode:
    tokens: List[int]
    logprob: float
    reward: float
    state: Dict

    @property
    def score(self) -> float:
        return self.logprob + self.reward


def clone_state_tree(obj):
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.detach().clone()
    if isinstance(obj, dict):
        return {k: clone_state_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clone_state_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(clone_state_tree(v) for v in obj)
    return copy.deepcopy(obj)


class RewardGuidedTreeSearch:
    def __init__(self, depth: int = 3, branching: int = 4, beam_width: int = 4):
        self.depth = depth
        self.branching = branching
        self.beam_width = beam_width

    @staticmethod
    def _extract_reward(model_out: Dict) -> float:
        wsra = model_out.get("wsra")
        if not wsra:
            return 0.0

        consistency = wsra["physics_consistency"].mean().item()
        proof = wsra["proof_confidence"].mean().item()
        compiled = wsra["compiled"].float().mean().item()
        return 0.6 * consistency + 0.3 * proof + 0.1 * compiled

    def search(self,
               initial_logits: torch.Tensor,
               initial_state: Dict,
               step_fn: Callable[[torch.Tensor, Dict], Dict]) -> SearchNode:
        log_probs = F.log_softmax(initial_logits, dim=-1)
        top_log_probs, top_tokens = torch.topk(log_probs, k=min(self.branching, log_probs.shape[-1]), dim=-1)

        frontier: List[SearchNode] = []
        for idx in range(top_tokens.shape[-1]):
            token_id = int(top_tokens[0, idx].item())
            token = torch.tensor([[token_id]], device=initial_logits.device)
            state = clone_state_tree(initial_state)
            out = step_fn(token, state)
            frontier.append(SearchNode(
                tokens=[token_id],
                logprob=float(top_log_probs[0, idx].item()),
                reward=self._extract_reward(out),
                state=out,
            ))

        for _ in range(1, self.depth):
            candidates: List[SearchNode] = []
            for node in frontier:
                next_logits = node.state["logits"][:, -1, :]
                node_log_probs = F.log_softmax(next_logits, dim=-1)
                child_log_probs, child_tokens = torch.topk(
                    node_log_probs, k=min(self.branching, node_log_probs.shape[-1]), dim=-1
                )

                for idx in range(child_tokens.shape[-1]):
                    token_id = int(child_tokens[0, idx].item())
                    token = torch.tensor([[token_id]], device=next_logits.device)
                    state = {
                        "layer_states": clone_state_tree(node.state["layer_states"]),
                        "titans_fifos": clone_state_tree(node.state["titans_fifos"]),
                        "wsra_state": clone_state_tree(node.state["wsra_state"]),
                    }
                    out = step_fn(token, state)
                    candidates.append(SearchNode(
                        tokens=node.tokens + [token_id],
                        logprob=node.logprob + float(child_log_probs[0, idx].item()),
                        reward=node.reward + self._extract_reward(out),
                        state=out,
                    ))

            if not candidates:
                break
            candidates.sort(key=lambda n: n.score, reverse=True)
            frontier = candidates[:self.beam_width]

        frontier.sort(key=lambda n: n.score, reverse=True)
        return frontier[0]
