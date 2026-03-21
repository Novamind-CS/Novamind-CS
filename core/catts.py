"""
NovaMind — CATTS 置信度感知自适应算力调度器

System 1 / System 2 切换逻辑:
    - 简单问题 → 早退 (只过前 N 层)，极速输出
    - 难题     → 树状搜索 + 多轨假设验证，多消耗算力换准确性

判断"难不难"的依据:
    1. Token 概率熵 — 熵高 = 模型不确定 = 难
    2. 内部层间表征漂移 — 相邻层表征变化大 = 问题复杂
    3. 逻辑约束满足度 — 满足度低 = 推理不稳定

核心思想来自 OpenAI o1 的 test-time compute scaling，
但 NovaMind 的版本在模型内部动态决策，不需要外部 orchestrator
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass

from novamind.core.tree_search import RewardGuidedTreeSearch


@dataclass
class DispatchDecision:
    """调度决策结果"""
    mode: str            # "fast" | "normal" | "deep"
    confidence: float    # 0~1，越高越确定
    entropy: float       # token 概率熵
    exit_layer: int      # 早退时在哪层退出（fast 模式）
    num_samples: int     # deep 模式采样路径数


class EntropyEstimator(nn.Module):
    """
    轻量级熵估计器
    
    在每一层都可以产生一个对最终输出分布的"提前预测"
    用于判断是否可以提前退出
    """

    def __init__(self, d_model: int, vocab_size: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        
        # 每隔几层设一个早退头（不是每层都设，太贵）
        self.exit_every = 4
        exit_layers = list(range(self.exit_every - 1, num_layers, self.exit_every))
        
        self.exit_heads = nn.ModuleDict({
            str(i): nn.Sequential(
                nn.RMSNorm(d_model),
                nn.Linear(d_model, vocab_size, bias=False)
            )
            for i in exit_layers
        })
        
        # 置信度阈值（可在推理时调整）
        self.fast_threshold = 0.85    # 置信度 > 85% → 直接早退
        self.deep_threshold = 0.45    # 置信度 < 45% → 触发深度推理

    def estimate_at_layer(self, hidden: torch.Tensor,
                          layer_idx: int) -> Optional[Tuple[torch.Tensor, float]]:
        """
        在 layer_idx 层估算输出置信度
        
        返回: (logits, confidence) 或 None（该层没有早退头）
        """
        key = str(layer_idx)
        if key not in self.exit_heads:
            return None
        
        # 用序列平均池化作为"全局语义"估计
        pooled = hidden.mean(dim=1)              # (B, d_model)
        logits = self.exit_heads[key](pooled)    # (B, vocab_size)
        
        probs = F.softmax(logits, dim=-1)
        
        # Shannon 熵（越低 = 越确定）
        entropy = -(probs * (probs + 1e-9).log()).sum(dim=-1).mean().item()
        max_entropy = math.log(logits.shape[-1])
        
        # 归一化置信度 (0~1，1=完全确定)
        confidence = 1.0 - entropy / max_entropy
        
        return logits, confidence


class RepresentationDriftDetector(nn.Module):
    """
    层间表征漂移检测器
    
    如果相邻两层的隐状态变化很大，说明模型还在"思考"
    变化趋于稳定时，可以安全退出
    """

    def __init__(self, d_model: int, window: int = 3):
        super().__init__()
        self.window = window
        self.history: List[torch.Tensor] = []
        
        # 漂移评分头（可学习的漂移敏感度）
        self.drift_proj = nn.Linear(d_model, 1, bias=True)
        nn.init.zeros_(self.drift_proj.weight)
        nn.init.constant_(self.drift_proj.bias, 0.5)

    def update(self, hidden: torch.Tensor) -> float:
        """
        更新历史并返回当前漂移分数 (0~1)
        0 = 稳定（可以退出）
        1 = 快速变化（需要继续计算）
        """
        pooled = hidden.mean(dim=1).detach()  # (B, d_model)
        
        if self.history:
            # 余弦距离（1 - cosine_similarity）
            prev = self.history[-1]
            cos_sim = F.cosine_similarity(pooled, prev, dim=-1).mean().item()
            drift = (1.0 - cos_sim) / 2.0  # 归一化到 [0,1]
        else:
            drift = 1.0  # 第一层，默认"高漂移"
        
        self.history.append(pooled)
        if len(self.history) > self.window:
            self.history.pop(0)
        
        return drift

    def reset(self):
        self.history.clear()


class TreeSearchSampler:
    """
    深度推理时的树状搜索采样器
    
    对于"难题"，生成多条候选推理路径，
    通过多数投票选出最可信的答案
    
    简化实现：Beam search 变体 + 一致性投票
    """

    def __init__(self, num_beams: int = 4, temperature: float = 0.7):
        self.num_beams = num_beams
        self.temperature = temperature

    def sample_candidates(self, logits: torch.Tensor,
                          top_k: int = 50) -> List[int]:
        """
        从 logits 采样多个候选 token
        logits: (vocab_size,) — 单个位置的 logit
        返回: num_beams 个候选 token id
        """
        # Top-K 截断
        top_k_logits, top_k_ids = torch.topk(logits, min(top_k, logits.shape[-1]))
        
        # 温度缩放
        scaled = top_k_logits / self.temperature
        probs = F.softmax(scaled, dim=-1)
        
        # 多次采样（允许重复）
        samples = torch.multinomial(probs, num_samples=self.num_beams, replacement=True)
        return [top_k_ids[s].item() for s in samples]

    @staticmethod
    def majority_vote(candidates: List[int]) -> int:
        """多数投票，返回出现最多的候选"""
        from collections import Counter
        return Counter(candidates).most_common(1)[0][0]


class CATTSDispatcher(nn.Module):
    """
    CATTS — 置信度感知测试时算力调度器
    
    完整的 System 1 / System 2 切换系统：
    
    System 1 (fast):  高置信度 → 早退，只过 exit_layer 层
    System 2 (deep):  低置信度 → 树搜索，多采样路径 + 多数投票
    Normal:           中间情况 → 全量层，标准贪心/采样
    
    使用方式：
        dispatcher = CATTSDispatcher(cfg)
        # 注册到模型中，在每层 forward 后调用 dispatcher.check()
        decision = dispatcher.decide(hidden, layer_idx)
        if decision.mode == "fast": break  # 提前退出
    """

    def __init__(self, d_model: int, vocab_size: int, num_layers: int,
                 fast_threshold: float = 0.85,
                 deep_threshold: float = 0.40,
                 num_deep_samples: int = 4):
        super().__init__()
        self.num_layers = num_layers
        self.fast_threshold = fast_threshold
        self.deep_threshold = deep_threshold
        self.num_deep_samples = num_deep_samples

        # 子模块
        self.entropy_estimator = EntropyEstimator(d_model, vocab_size, num_layers)
        self.drift_detector = RepresentationDriftDetector(d_model)
        self.tree_sampler = TreeSearchSampler(num_beams=num_deep_samples)

        # 统计（可选，用于监控调度分布）
        self._stats: Dict[str, int] = {"fast": 0, "normal": 0, "deep": 0}

    def reset_per_token(self):
        """每个新 token 生成前重置漂移历史"""
        self.drift_detector.reset()

    def check_layer(self, hidden: torch.Tensor,
                    layer_idx: int) -> Optional[DispatchDecision]:
        """
        在每层 forward 结束后调用
        如果应该早退，返回 DispatchDecision；否则返回 None（继续）
        
        hidden: (B, L, d_model) — 当前层的输出
        layer_idx: 0-indexed 当前层号
        """
        # 漂移检测
        drift = self.drift_detector.update(hidden)

        # 熵估计（只在有早退头的层）
        exit_result = self.entropy_estimator.estimate_at_layer(hidden, layer_idx)
        if exit_result is None:
            return None  # 这层没有早退头，继续

        _, confidence = exit_result

        # 综合置信度（熵 + 漂移）
        # 漂移大时，降低置信度（表示还没收敛）
        adjusted_confidence = confidence * (1.0 - 0.3 * drift)

        # System 1: 早退
        if adjusted_confidence >= self.fast_threshold:
            self._stats["fast"] += 1
            return DispatchDecision(
                mode="fast",
                confidence=adjusted_confidence,
                entropy=1.0 - confidence,
                exit_layer=layer_idx,
                num_samples=1
            )

        # 到了最后一层，判断是否需要深度推理
        if layer_idx >= self.num_layers - 1:
            if adjusted_confidence < self.deep_threshold:
                self._stats["deep"] += 1
                return DispatchDecision(
                    mode="deep",
                    confidence=adjusted_confidence,
                    entropy=1.0 - confidence,
                    exit_layer=layer_idx,
                    num_samples=self.num_deep_samples
                )
            else:
                self._stats["normal"] += 1
                return DispatchDecision(
                    mode="normal",
                    confidence=adjusted_confidence,
                    entropy=1.0 - confidence,
                    exit_layer=layer_idx,
                    num_samples=1
                )

        return None  # 还没到早退层，继续

    def get_stats(self) -> Dict[str, float]:
        """返回调度统计（各模式的比例）"""
        total = max(1, sum(self._stats.values()))
        return {k: v / total for k, v in self._stats.items()}

    def reset_stats(self):
        self._stats = {"fast": 0, "normal": 0, "deep": 0}


class AdaptiveNovaMindWrapper(nn.Module):
    """
    带 CATTS 的 NovaMind 推理包装器
    
    将 CATTS 调度器集成到推理循环中
    训练时禁用（全量层，无早退），推理时启用
    
    使用:
        wrapper = AdaptiveNovaMindWrapper(model, cfg)
        output = wrapper.adaptive_forward(input_ids)
    """

    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config

        self.dispatcher = CATTSDispatcher(
            d_model=config.hidden_dim,
            vocab_size=config.vocab_size,
            num_layers=config.num_layers,
            fast_threshold=config.catts_fast_threshold,
            deep_threshold=config.catts_deep_threshold,
            num_deep_samples=config.catts_num_deep_samples,
        )
        self.tree_search = RewardGuidedTreeSearch(
            depth=config.tree_search_depth,
            branching=config.tree_search_branching,
            beam_width=config.tree_search_beam_width,
        )

    @torch.no_grad()
    def adaptive_forward(self, input_ids: torch.Tensor,
                         layer_states=None,
                         titans_fifos=None,
                         wsra_state=None,
                         use_catts: bool = True) -> Dict:
        """
        带 CATTS 的前向推理
        
        返回:
            logits, decision (DispatchDecision), exit_layer
        """
        if not use_catts:
            # 标准前向
            out = self.model(
                input_ids,
                layer_states=layer_states,
                titans_fifos=titans_fifos,
                wsra_state=wsra_state,
                inference_mode=True
            )
            return {
                "logits": out["logits"],
                "decision": None,
                "exit_layer": self.config.num_layers - 1,
                "layer_states": out["layer_states"],
                "titans_fifos": out["titans_fifos"],
                "wsra_state": out["wsra_state"],
            }

        # 逐层手动前向，检查早退
        self.dispatcher.reset_per_token()

        B, L = input_ids.shape
        x = self.model.embed(input_ids)

        if layer_states is None:
            layer_states = [None] * self.config.num_layers
        if titans_fifos is None:
            titans_fifos = {}

        updated_states = list(layer_states)
        updated_fifos = dict(titans_fifos)
        decision = None

        for i, layer in enumerate(self.model.layers):
            # TITANS 记忆
            if str(i) in self.model.titans_modules:
                fifo_buf = updated_fifos.get(str(i))
                x, new_fifo = self.model.titans_modules[str(i)](x, fifo_buf)
                updated_fifos[str(i)] = new_fifo

            # 层前向
            x, updated_states[i] = layer(
                x,
                state=updated_states[i],
                inference_mode=True
            )

            # CATTS 检查
            decision = self.dispatcher.check_layer(x, i)
            if decision is not None and decision.mode in ("fast", "normal"):
                # 早退：直接从当前层的隐状态预测
                break

        wsra_outputs = None
        if self.model.wsra is not None:
            x, wsra_outputs = self.model.wsra(x, wsra_state)
            wsra_state = wsra_outputs["state"]

        # 最终输出
        x = self.model.final_norm(x)

        if decision is not None and decision.mode == "deep":
            # System 2：多采样路径 + 多数投票
            logits_list = []
            for _ in range(decision.num_samples):
                # 每次加一点随机扰动（模拟不同推理路径）
                x_perturb = x + torch.randn_like(x) * 0.01
                logits_list.append(self.model.lm_head(x_perturb))
            # 平均多路 logits（soft voting）
            logits = torch.stack(logits_list, dim=0).mean(dim=0)
        else:
            logits = self.model.lm_head(x)

        return {
            "logits": logits,
            "decision": decision,
            "exit_layer": i,
            "catts_stats": self.dispatcher.get_stats(),
            "layer_states": updated_states,
            "titans_fifos": updated_fifos,
            "wsra_state": wsra_state,
            "wsra": wsra_outputs,
        }

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor,
                 max_new_tokens: int = 200,
                 temperature: float = 1.0,
                 top_p: float = 0.9,
                 top_k: int = 50,
                 use_catts: bool = True):
        layer_states = None
        titans_fifos = {}
        wsra_state = None

        out = self.adaptive_forward(
            input_ids,
            layer_states=layer_states,
            titans_fifos=titans_fifos,
            wsra_state=wsra_state,
            use_catts=use_catts
        )
        layer_states = out["layer_states"]
        titans_fifos = out["titans_fifos"]
        wsra_state = out["wsra_state"]
        next_token_logits = out["logits"][:, -1, :]

        generated = [input_ids]
        decisions = []
        if out["decision"] is not None:
            decisions.append(out["decision"])

        for _ in range(max_new_tokens):
            logits = next_token_logits
            current_decision = decisions[-1] if decisions else None

            if current_decision is not None and current_decision.mode == "deep":
                def step_fn(token, state):
                    return self.adaptive_forward(
                        token,
                        layer_states=state["layer_states"],
                        titans_fifos=state["titans_fifos"],
                        wsra_state=state["wsra_state"],
                        use_catts=False,
                    )

                best = self.tree_search.search(
                    initial_logits=logits,
                    initial_state={
                        "layer_states": layer_states,
                        "titans_fifos": titans_fifos,
                        "wsra_state": wsra_state,
                    },
                    step_fn=step_fn,
                )
                next_token = torch.tensor([[best.tokens[0]]], device=logits.device)
                out = best.state
            else:
                if temperature != 1.0:
                    logits = logits / max(temperature, 1e-5)

                if top_k > 0:
                    top_k_vals = torch.topk(logits, min(top_k, logits.shape[-1]))[0]
                    logits = logits.masked_fill(logits < top_k_vals[:, -1:], float("-inf"))

                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cumprobs = torch.cumsum(sorted_probs, dim=-1)
                    remove_mask = cumprobs - sorted_probs > top_p
                    sorted_logits[remove_mask] = float("-inf")
                    logits = torch.full_like(logits, float("-inf"))
                    logits.scatter_(1, sorted_idx, sorted_logits)

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                out = self.adaptive_forward(
                    next_token,
                    layer_states=layer_states,
                    titans_fifos=titans_fifos,
                    wsra_state=wsra_state,
                    use_catts=use_catts
                )

            generated.append(next_token)
            layer_states = out["layer_states"]
            titans_fifos = out["titans_fifos"]
            wsra_state = out["wsra_state"]
            next_token_logits = out["logits"][:, -1, :]
            if out["decision"] is not None:
                decisions.append(out["decision"])

        return {
            "tokens": torch.cat(generated, dim=1),
            "decisions": decisions,
            "catts_stats": self.dispatcher.get_stats(),
        }

    def profile_difficulty(self, prompts: List[str],
                           tokenizer) -> List[Dict]:
        """
        对一批 prompt 做难度分析
        返回每个 prompt 的预期调度模式
        """
        results = []
        for prompt in prompts:
            tokens = tokenizer(prompt, return_tensors="pt")
            input_ids = tokens["input_ids"].to(next(self.parameters()).device)
            out = self.adaptive_forward(input_ids, use_catts=True)
            d = out["decision"]
            results.append({
                "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
                "mode": d.mode if d else "normal",
                "confidence": round(d.confidence if d else 0.5, 3),
                "exit_layer": out["exit_layer"],
                "max_layers": self.config.num_layers,
                "compute_ratio": (out["exit_layer"] + 1) / self.config.num_layers,
            })
        return results
