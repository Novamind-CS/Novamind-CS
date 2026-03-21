"""
NovaMind — 因果推理层 (简化版 ECAM) + 可微逻辑层 (LTN)

ECAM (Endogenous Causal Attention Mechanism):
    标准注意力: score(q,k) = softmax(q @ k^T / sqrt(d))
    ECAM:       score(q,k) = softmax(causal_modulate(q @ k^T, G_local) / sqrt(d))
    
    其中 G_local 是从输入表征中快速推断的局部因果图
    G[i,j] 表示 token_i 对 token_j 的因果影响强度（非对称）

LTN (Logic Tensor Network) T-norms:
    用可微函数近似经典逻辑运算，使约束可以参与反向传播
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import List, Optional


# ─────────────────────────────────────────────
# 可微逻辑层 (T-norms)
# ─────────────────────────────────────────────

class TNorm(nn.Module):
    """
    T-范数逻辑层 — 将神经网络输出解释为模糊真值 [0,1]
    并用可微操作实现逻辑连接词
    
    支持:
    - product: T(a,b) = a*b  (梯度平滑，最常用)
    - lukasiewicz: T(a,b) = max(a+b-1, 0)
    - minmax: T(a,b) = min(a,b)  (用软近似)
    """

    def __init__(self, tnorm_type: str = "product"):
        super().__init__()
        assert tnorm_type in ("product", "lukasiewicz", "minmax")
        self.tnorm_type = tnorm_type

    def and_(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """逻辑 AND"""
        if self.tnorm_type == "product":
            return a * b
        elif self.tnorm_type == "lukasiewicz":
            return torch.clamp(a + b - 1, min=0.0)
        else:  # minmax，用软近似
            return torch.min(a, b)

    def or_(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """逻辑 OR（由 de Morgan 定律推导）"""
        if self.tnorm_type == "product":
            return a + b - a * b
        elif self.tnorm_type == "lukasiewicz":
            return torch.clamp(a + b, max=1.0)
        else:
            return torch.max(a, b)

    def implies(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """
        逻辑蕴含 (a → b)
        product 版本: 1 - a + a*b
        
        关键特性：当 a=1（前提成立）且 b=0（结论不成立）时，
        梯度极大（负方向），强制网络修正
        """
        if self.tnorm_type == "product":
            return 1 - a + a * b
        else:
            return self.or_(1 - a, b)

    def not_(self, a: torch.Tensor) -> torch.Tensor:
        return 1 - a

    def forall(self, x: torch.Tensor, dim: int = -1, p: float = -6.0) -> torch.Tensor:
        """
        全称量词 ∀x: P(x)
        用广义均值近似（p-均值，p 越负越关注最差情况）
        
        pMean(x, p) = (1/n * sum(x^p))^(1/p)
        p=-6 时近似 min 函数，任何一个反例都引发大梯度
        """
        x = torch.clamp(x, min=1e-6)  # 数值稳定
        return torch.pow(torch.mean(torch.pow(x, p), dim=dim), 1.0 / p)

    def exists(self, x: torch.Tensor, dim: int = -1, p: float = 6.0) -> torch.Tensor:
        """存在量词 ∃x: P(x)，用正 p-均值近似 max"""
        return torch.pow(torch.mean(torch.pow(x, p), dim=dim), 1.0 / p)


class LogicConstraintLayer(nn.Module):
    """
    逻辑约束层 — 在网络输出上施加形式逻辑约束
    
    使用场景：
    - 确保输出不违反已知规则（如物理定律、业务约束）
    - 作为辅助损失项，惩罚逻辑不满足度
    - 直接嵌入网络中强制约束（软约束）
    
    用法：
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

        # 谓词网络：将隐藏状态映射到谓词真值 [0,1]
        self.predicate_net = nn.Sequential(
            nn.Linear(d_model, num_predicates * 2, bias=False),
            nn.GELU(),
            nn.Linear(num_predicates * 2, num_predicates, bias=True),
            nn.Sigmoid()  # 输出范围 [0,1]，表示模糊真值
        )

        # 规则权重（可学习的规则重要性）
        self.rule_weights = nn.Parameter(torch.ones(num_predicates))

    def forward(self, x: torch.Tensor):
        """
        x: (B, L, d_model)
        返回: (x_out, satisfaction_score)
        
        satisfaction_score: (B,) — 当前输出满足逻辑约束的程度
        """
        # 计算每个位置的谓词真值
        pred_values = self.predicate_net(x)  # (B, L, num_predicates)

        # 全称量词：每个规则在整个序列上都应该满足
        rule_satisfaction = self.tnorm.forall(pred_values, dim=1)  # (B, num_predicates)

        # 加权综合满足度
        weights = F.softmax(self.rule_weights, dim=0)
        satisfaction = (rule_satisfaction * weights).sum(dim=-1)  # (B,)

        # 满足度作为门控：不满足时轻微惩罚输出（软约束）
        penalty = (1 - satisfaction).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
        x_out = x * (1 - 0.1 * penalty.detach())  # 不通过 penalty 反传到 x

        return x_out, satisfaction


# ─────────────────────────────────────────────
# 简化版 ECAM — 因果调制注意力
# ─────────────────────────────────────────────

class LocalCausalGraph(nn.Module):
    """
    快速局部因果图推断器
    
    从输入表征中推断 token 间的因果影响方向
    输出: G[i,j] ∈ [0,1] 表示 token_i → token_j 的因果强度
    
    实现思路：
    - 用小型 MLP 从每对 token 表征推断因果方向
    - 用不对称性正则化鼓励学出有向图（而非无向相关图）
    - 训练中用 PC 算法等发现的真实因果图作为监督信号
    """

    def __init__(self, d_model: int, causal_dim: int = 32, max_len: int = 512):
        super().__init__()
        self.causal_dim = causal_dim
        self.max_len = max_len

        # 将隐藏状态投影到因果嵌入空间
        self.cause_proj = nn.Linear(d_model, causal_dim, bias=False)  # "原因"方向
        self.effect_proj = nn.Linear(d_model, causal_dim, bias=False)  # "结果"方向

        # 因果强度评分
        self.causal_score = nn.Sequential(
            nn.Linear(causal_dim * 2, causal_dim, bias=True),
            nn.GELU(),
            nn.Linear(causal_dim, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, d_model)
        返回: causal_graph G (B, L, L) — G[b,i,j] = P(i causes j)
        
        注意：L 较大时这会很慢。实际使用时应限制窗口大小或使用稀疏近似。
        """
        B, L, _ = x.shape

        # 只在较短序列或滑动窗口上计算（超长序列限制为局部窗口）
        win = min(L, self.max_len)
        x_win = x[:, :win]  # 截取窗口

        cause = self.cause_proj(x_win)   # (B, win, causal_dim)
        effect = self.effect_proj(x_win)  # (B, win, causal_dim)

        # 构建所有对的特征（朴素实现 O(L^2)，实际应用中用稀疏近似）
        cause_exp = cause.unsqueeze(2).expand(-1, -1, win, -1)  # (B, win, win, d)
        effect_exp = effect.unsqueeze(1).expand(-1, win, -1, -1)  # (B, win, win, d)

        pair_feat = torch.cat([cause_exp, effect_exp], dim=-1)  # (B, win, win, 2d)
        G = self.causal_score(pair_feat).squeeze(-1)  # (B, win, win)

        # 如果序列比窗口长，把窗口外的部分填 0.5（中性）
        if L > win:
            pad = torch.full((B, L, L), 0.5, device=x.device, dtype=x.dtype)
            pad[:, :win, :win] = G
            return pad

        return G


@dataclass
class CounterfactualIntervention:
    """对局部因果图施加 do-operator 风格的干预。"""
    source_index: int
    target_index: Optional[int] = None
    strength: float = 1.0
    mode: str = "set"


class CausalModulatedAttention(nn.Module):
    """
    因果调制注意力 (简化版 ECAM)
    
    核心改变：注意力分数不再只是点积相似度，
    而是被局部因果图 G 调制后的非对称分布
    
    score_{causal}(i,j) = score_{std}(i,j) * (1 + alpha * G[i,j])
    
    G[i,j] 高 → token_j 对 token_i 有强因果影响 → 注意力增强
    G[i,j] 低 → 仅相关但无因果 → 注意力降低
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

        # 标准 QKV 投影
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # 因果图推断器
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

        # 标准 QKV
        q = self.q_proj(x).view(B, L, H, d).transpose(1, 2)  # (B, H, L, d)
        k = self.k_proj(kv_input).view(B, kv_input.shape[1], H, d).transpose(1, 2)
        v = self.v_proj(kv_input).view(B, kv_input.shape[1], H, d).transpose(1, 2)

        # 标准注意力分数
        scores = (q @ k.transpose(-2, -1)) / self.scale  # (B, H, L, S)

        # 因果调制（在推断或需要高精度推理时激活）
        G_local = None
        if use_causal_modulation:
            G = self.causal_graph(kv_input)  # (B, S, S)
            G = self._apply_interventions(G, interventions)
            G_local = G[:, -L:, :]
            # G 调制：增强因果方向的注意力
            causal_bias = self.causal_alpha * (G_local.unsqueeze(1) - 0.5)
            scores = scores + causal_bias

        # 因果掩码（自回归）
        causal_mask = self._build_causal_mask(L, kv_input.shape[1], memory_len, x.device)
        scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        # Softmax + 加权求和
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
