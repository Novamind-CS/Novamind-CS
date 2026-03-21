"""
WSRA prototype reasoning stack.

Combines four runtime mechanisms into an integrated reasoning chain:
1. Physics-grounded logic engine
2. Just-in-time compiled circuits
3. Dynamic frustum-culling memory
4. Adversarial logic proof trees
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsGroundedLogicEngine(nn.Module):

    def __init__(self, d_model: int, num_slots: int = 8):
        super().__init__()
        self.num_slots = num_slots
        self.slot_proj = nn.Linear(d_model, num_slots, bias=False)
        self.delta_proj = nn.Linear(d_model, d_model, bias=False)
        self.read_proj = nn.Linear(d_model, d_model, bias=False)
        self.coord_head = nn.Linear(d_model, 4, bias=True)

    def forward(self, x: torch.Tensor, world_state: Optional[torch.Tensor] = None):
        B, _, D = x.shape
        pooled = x.mean(dim=1)

        if world_state is None:
            world_state = torch.zeros(B, self.num_slots, D, device=x.device, dtype=x.dtype)

        slot_weights = torch.softmax(self.slot_proj(x), dim=1)          # (B, L, S)
        slot_updates = torch.einsum("bls,bld->bsd", slot_weights, self.delta_proj(x))
        new_world = 0.92 * world_state + 0.08 * slot_updates

        coords = self.coord_head(new_world.float())
        positions = torch.tanh(coords[..., :3])
        radii = torch.sigmoid(coords[..., 3]) * 0.25 + 0.05

        pairwise = positions.unsqueeze(2) - positions.unsqueeze(1)
        distances = pairwise.pow(2).sum(dim=-1).sqrt() + 1e-6
        min_distance = radii.unsqueeze(2) + radii.unsqueeze(1)
        overlap = torch.relu(min_distance - distances)
        overlap = overlap * (1 - torch.eye(self.num_slots, device=x.device).unsqueeze(0))

        velocity_penalty = (new_world - world_state).pow(2).mean(dim=-1)
        physics_loss = overlap.mean(dim=(1, 2)) + 0.1 * velocity_penalty.mean(dim=-1)
        consistency = torch.exp(-physics_loss)

        world_context = self.read_proj(new_world.mean(dim=1)).unsqueeze(1)
        x = x + world_context * consistency.view(B, 1, 1)

        return x, {
            "world_state": new_world,
            "physics_loss": physics_loss,
            "consistency": consistency,
        }


class JITCompiledCircuit(nn.Module):

    def __init__(self, d_model: int, rank: int = 32, compile_threshold: float = 0.55):
        super().__init__()
        self.rank = rank
        self.compile_threshold = compile_threshold
        self.signature_proj = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.GELU(),
        )
        self.compile_gate = nn.Linear(d_model, 1, bias=True)
        self.hyper_a = nn.Linear(d_model, d_model * rank, bias=False)
        self.hyper_b = nn.Linear(d_model, rank * d_model, bias=False)
        self.merge_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, difficulty: torch.Tensor):
        B, _, D = x.shape
        signature = self.signature_proj(x.mean(dim=1))
        compile_score = torch.sigmoid(self.compile_gate(signature)).squeeze(-1)
        should_compile = (compile_score > self.compile_threshold) | (difficulty > 0.5)

        if not should_compile.any():
            return x, {
                "compiled": should_compile,
                "compile_score": compile_score,
                "circuit_summary": signature.detach(),
            }

        A = self.hyper_a(signature).view(B, D, self.rank)
        Bm = self.hyper_b(signature).view(B, self.rank, D)
        jit_hidden = torch.einsum("bld,bdr->blr", x, A)
        jit_hidden = F.gelu(jit_hidden)
        jit_out = torch.einsum("blr,brd->bld", jit_hidden, Bm)

        compile_mask = should_compile.to(x.dtype).view(B, 1, 1)
        x = x + self.merge_proj(jit_out) * compile_mask

        return x, {
            "compiled": should_compile,
            "compile_score": compile_score,
            "circuit_summary": jit_out.mean(dim=1).detach(),
        }


class DynamicFrustumMemory(nn.Module):

    def __init__(self, d_model: int, active_tokens: int = 256):
        super().__init__()
        self.active_tokens = active_tokens
        self.salience_head = nn.Linear(d_model, 1, bias=False)
        self.read_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, state: Optional[Dict] = None):
        B, L, D = x.shape
        active_memory = None if state is None else state.get("active")
        cold_summary = None if state is None else state.get("cold")

        combined = x if active_memory is None else torch.cat([active_memory.to(x.device), x], dim=1)
        salience = self.salience_head(combined).squeeze(-1)
        keep = min(self.active_tokens, combined.shape[1])
        top_idx = torch.topk(salience, k=keep, dim=1).indices
        top_idx = torch.sort(top_idx, dim=-1).values
        gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, D)
        new_active = torch.gather(combined, 1, gather_idx).detach()

        keep_mask = torch.zeros_like(salience, dtype=torch.bool)
        keep_mask.scatter_(1, top_idx, True)
        dropped = combined.masked_select(~keep_mask.unsqueeze(-1)).view(B, -1, D)

        if dropped.shape[1] > 0:
            dropped_summary = dropped.mean(dim=1)
            if cold_summary is None:
                cold_summary = dropped_summary
            else:
                cold_summary = 0.9 * cold_summary.to(x.device) + 0.1 * dropped_summary
        elif cold_summary is None:
            cold_summary = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        else:
            cold_summary = cold_summary.to(x.device)

        active_context = new_active.mean(dim=1, keepdim=True)
        cold_context = cold_summary.unsqueeze(1)
        x = x + self.read_proj(active_context + cold_context)

        stored_active = new_active.detach()
        if x.is_cuda:
            stored_active = stored_active.to("cpu")
            cold_summary = cold_summary.detach().to("cpu")
        else:
            cold_summary = cold_summary.detach()

        return x, {
            "active": stored_active,
            "cold": cold_summary,
            "active_tokens": keep,
        }


class AdversarialLogicProofTree(nn.Module):

    def __init__(self, d_model: int, num_branches: int = 4):
        super().__init__()
        self.num_branches = num_branches
        self.branch_emb = nn.Parameter(torch.randn(num_branches, d_model) * 0.02)
        self.generator = nn.Linear(d_model, d_model, bias=False)
        self.verifier = nn.Sequential(
            nn.Linear(d_model * 2, d_model, bias=False),
            nn.GELU(),
            nn.Linear(d_model, 1, bias=True),
        )
        self.merge_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, consistency: torch.Tensor):
        pooled = x.mean(dim=1)
        candidates = []
        scores = []

        for branch in self.branch_emb:
            candidate = self.generator(pooled + branch.unsqueeze(0))
            score = self.verifier(torch.cat([pooled, candidate], dim=-1)).squeeze(-1)
            candidates.append(candidate)
            scores.append(score)

        candidate_tensor = torch.stack(candidates, dim=1)   # (B, K, D)
        score_tensor = torch.stack(scores, dim=1)           # (B, K)
        attacker_penalty = (1.0 - consistency).unsqueeze(-1)
        robust_scores = score_tensor - attacker_penalty
        weights = torch.softmax(robust_scores, dim=-1)
        merged = torch.einsum("bk,bkd->bd", weights, candidate_tensor)
        proof_margin = robust_scores.max(dim=-1).values - robust_scores.mean(dim=-1)
        proof_loss = torch.relu(0.25 - proof_margin)

        x = x + self.merge_proj(merged).unsqueeze(1)
        return x, {
            "proof_loss": proof_loss,
            "proof_confidence": weights.max(dim=-1).values,
            "branch_scores": robust_scores,
        }


class WSRAReasoningStack(nn.Module):

    def __init__(self, d_model: int, world_slots: int = 8, jit_rank: int = 32,
                 frustum_tokens: int = 256, proof_branches: int = 4,
                 compile_threshold: float = 0.55):
        super().__init__()
        self.frustum = DynamicFrustumMemory(d_model, frustum_tokens)
        self.physics = PhysicsGroundedLogicEngine(d_model, world_slots)
        self.jit = JITCompiledCircuit(d_model, jit_rank, compile_threshold)
        self.proof_tree = AdversarialLogicProofTree(d_model, proof_branches)
        self.norm = nn.RMSNorm(d_model)

    def forward(self, x: torch.Tensor, state: Optional[Dict] = None):
        state = state or {}

        x, frustum_state = self.frustum(x, state.get("frustum"))
        x, physics_info = self.physics(x, state.get("world_state"))
        difficulty = 1.0 - physics_info["consistency"]
        x, jit_info = self.jit(x, difficulty)
        x, proof_info = self.proof_tree(x, physics_info["consistency"])
        x = self.norm(x)

        wsra_loss = physics_info["physics_loss"] + proof_info["proof_loss"]
        new_state = {
            "frustum": frustum_state,
            "world_state": physics_info["world_state"].detach(),
            "circuit_summary": jit_info["circuit_summary"],
        }

        return x, {
            "loss": wsra_loss,
            "state": new_state,
            "physics_consistency": physics_info["consistency"],
            "proof_confidence": proof_info["proof_confidence"],
            "compiled": jit_info["compiled"],
            "compile_score": jit_info["compile_score"],
            "active_tokens": frustum_state["active_tokens"],
        }
