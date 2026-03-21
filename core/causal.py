"""
Causal reasoning layers and differentiable logic modules.

Includes a simplified ECAM-style causal attention mechanism together with
Logic Tensor Network style T-norm operators for soft logical constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import List, Optional






class TNorm(nn.Module):
    """
    
    - lukasiewicz: T(a,b) = max(a+b-1, 0)
    """

    def __init__(self, tnorm_type: str = "product"):
        super().__init__()
        assert tnorm_type in ("product", "lukasiewicz", "minmax")
        self.tnorm_type = tnorm_type

    def and_(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.tnorm_type == "product":
            return a * b
        elif self.tnorm_type == "lukasiewicz":
            return torch.clamp(a + b - 1, min=0.0)
        else:
            return torch.min(a, b)

    def or_(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.tnorm_type == "product":
            return a + b - a * b
        elif self.tnorm_type == "lukasiewicz":
            return torch.clamp(a + b, max=1.0)
        else:
            return torch.max(a, b)

    def implies(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        
        """
        if self.tnorm_type == "product":
            return 1 - a + a * b
        else:
            return self.or_(1 - a, b)

    def not_(self, a: torch.Tensor) -> torch.Tensor:
        return 1 - a

    def forall(self, x: torch.Tensor, dim: int = -1, p: float = -6.0) -> torch.Tensor:
        """
        
        pMean(x, p) = (1/n * sum(x^p))^(1/p)
        """
        x = torch.clamp(x, min=1e-6)
        return torch.pow(torch.mean(torch.pow(x, p), dim=dim), 1.0 / p)

    def exists(self, x: torch.Tensor, dim: int = -1, p: float = 6.0) -> torch.Tensor:
        return torch.pow(torch.mean(torch.pow(x, p), dim=dim), 1.0 / p)


class LogicConstraintLayer(nn.Module):
    """
    
    
        constraint = LogicConstraintLayer(d_model, rules=my_rules)
        output, satisfaction = constraint(hidden_states)
        loss += (1 - satisfaction).mean() * lambda_logic
    """

    def __init__(self, d_model: int, num_predicates: int = 32,
                 tnorm_type: str = "product"):
        super().__init__()
        self.d_model = d_model
        self.num_predicates = num_predicates
        self.tnorm = TNorm(tnorm_type)


        self.predicate_net = nn.Sequential(
            nn.Linear(d_model, num_predicates * 2, bias=False),
            nn.GELU(),
            nn.Linear(num_predicates * 2, num_predicates, bias=True),
            nn.Sigmoid()
        )


        self.rule_weights = nn.Parameter(torch.ones(num_predicates))

    def forward(self, x: torch.Tensor):
        """
        x: (B, L, d_model)
        
        """

        pred_values = self.predicate_net(x)  # (B, L, num_predicates)


        rule_satisfaction = self.tnorm.forall(pred_values, dim=1)  # (B, num_predicates)


        weights = F.softmax(self.rule_weights, dim=0)
        satisfaction = (rule_satisfaction * weights).sum(dim=-1)  # (B,)


        penalty = (1 - satisfaction).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
        x_out = x * (1 - 0.1 * penalty.detach())

        return x_out, satisfaction






class LocalCausalGraph(nn.Module):
    """
    
    
    """

    def __init__(self, d_model: int, causal_dim: int = 32, max_len: int = 512):
        super().__init__()
        self.causal_dim = causal_dim
        self.max_len = max_len


        self.cause_proj = nn.Linear(d_model, causal_dim, bias=False)
        self.effect_proj = nn.Linear(d_model, causal_dim, bias=False)


        self.causal_score = nn.Sequential(
            nn.Linear(causal_dim * 2, causal_dim, bias=True),
            nn.GELU(),
            nn.Linear(causal_dim, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, d_model)
        
        """
        B, L, _ = x.shape


        win = min(L, self.max_len)
        x_win = x[:, :win]

        cause = self.cause_proj(x_win)   # (B, win, causal_dim)
        effect = self.effect_proj(x_win)  # (B, win, causal_dim)


        cause_exp = cause.unsqueeze(2).expand(-1, -1, win, -1)  # (B, win, win, d)
        effect_exp = effect.unsqueeze(1).expand(-1, win, -1, -1)  # (B, win, win, d)

        pair_feat = torch.cat([cause_exp, effect_exp], dim=-1)  # (B, win, win, 2d)
        G = self.causal_score(pair_feat).squeeze(-1)  # (B, win, win)


        if L > win:
            pad = torch.full((B, L, L), 0.5, device=x.device, dtype=x.dtype)
            pad[:, :win, :win] = G
            return pad

        return G


@dataclass
class CounterfactualIntervention:
    source_index: int
    target_index: Optional[int] = None
    strength: float = 1.0
    mode: str = "set"


class CausalModulatedAttention(nn.Module):
    """
    
    
    score_{causal}(i,j) = score_{std}(i,j) * (1 + alpha * G[i,j])
    
    """

    def __init__(self, d_model: int, num_heads: int = 8,
                 causal_dim: int = 32, causal_alpha: float = 0.3,
                 attention_window: int = 256):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = math.sqrt(self.head_dim)
        self.causal_alpha = causal_alpha
        self.attention_window = attention_window


        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)


        self.causal_graph = LocalCausalGraph(d_model, causal_dim)

        self.norm = nn.RMSNorm(d_model)

    @staticmethod
    def _build_causal_mask(query_len: int, key_len: int, memory_len: int,
                           device) -> torch.Tensor:
        local_mask = torch.triu(
            torch.full((query_len, query_len), float("-inf"), device=device),
            diagonal=1
        )
        if memory_len <= 0:
            return local_mask

        memory_mask = torch.zeros(query_len, memory_len, device=device)
        return torch.cat([memory_mask, local_mask], dim=-1)

    @staticmethod
    def _apply_interventions(G: torch.Tensor,
                             interventions: Optional[List[CounterfactualIntervention]]):
        if not interventions:
            return G

        G_mod = G.clone()
        L = G.shape[-1]
        for intervention in interventions:
            src = max(0, min(L - 1, intervention.source_index))
            strength = float(max(0.0, min(1.0, intervention.strength)))
            if intervention.target_index is None:
                if intervention.mode == "suppress":
                    G_mod[:, src, :] = G_mod[:, src, :] * (1.0 - strength)
                elif intervention.mode == "boost":
                    G_mod[:, src, :] = torch.clamp(
                        G_mod[:, src, :] + strength * (1.0 - G_mod[:, src, :]),
                        0.0, 1.0
                    )
                else:
                    G_mod[:, src, :] = strength
            else:
                tgt = max(0, min(L - 1, intervention.target_index))
                if intervention.mode == "suppress":
                    G_mod[:, src, tgt] = G_mod[:, src, tgt] * (1.0 - strength)
                elif intervention.mode == "boost":
                    G_mod[:, src, tgt] = torch.clamp(
                        G_mod[:, src, tgt] + strength * (1.0 - G_mod[:, src, tgt]),
                        0.0, 1.0
                    )
                else:
                    G_mod[:, src, tgt] = strength
        return G_mod

    def forward(self, x: torch.Tensor,
                use_causal_modulation: bool = True,
                memory: torch.Tensor = None,
                interventions: Optional[List[CounterfactualIntervention]] = None,
                return_causal_graph: bool = False):
        """x: (B, L, d_model)"""
        B, L, D = x.shape
        H = self.num_heads
        d = self.head_dim
        memory_len = 0 if memory is None else memory.shape[1]

        if memory is not None and memory_len > 0:
            memory = memory[:, -self.attention_window:]
            kv_input = torch.cat([memory, x], dim=1)
            memory_len = memory.shape[1]
        else:
            kv_input = x


        q = self.q_proj(x).view(B, L, H, d).transpose(1, 2)  # (B, H, L, d)
        k = self.k_proj(kv_input).view(B, kv_input.shape[1], H, d).transpose(1, 2)
        v = self.v_proj(kv_input).view(B, kv_input.shape[1], H, d).transpose(1, 2)


        scores = (q @ k.transpose(-2, -1)) / self.scale  # (B, H, L, S)


        G_local = None
        if use_causal_modulation:
            G = self.causal_graph(kv_input)  # (B, S, S)
            G = self._apply_interventions(G, interventions)
            G_local = G[:, -L:, :]

            causal_bias = self.causal_alpha * (G_local.unsqueeze(1) - 0.5)
            scores = scores + causal_bias


        causal_mask = self._build_causal_mask(L, kv_input.shape[1], memory_len, x.device)
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)


        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)

        if return_causal_graph:
            return out, G_local
        return out

    @torch.no_grad()
    def simulate_counterfactual(self, x: torch.Tensor,
                                interventions: List[CounterfactualIntervention],
                                memory: torch.Tensor = None) -> dict:
        factual, graph = self.forward(
            x,
            use_causal_modulation=True,
            memory=memory,
            return_causal_graph=True
        )
        counterfactual, cf_graph = self.forward(
            x,
            use_causal_modulation=True,
            memory=memory,
            interventions=interventions,
            return_causal_graph=True
        )
        delta = (counterfactual - factual).norm(dim=-1).mean(dim=-1)
        return {
            "factual": factual,
            "counterfactual": counterfactual,
            "graph": graph,
            "counterfactual_graph": cf_graph,
            "delta_norm": delta,
        }
