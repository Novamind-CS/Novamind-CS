"""
Confidence-aware adaptive test-time scheduling (CATTS).

This module implements internal System 1 / System 2 routing based on entropy,
representation drift, and logic satisfaction so the model can decide when to
exit early and when to spend more test-time compute.
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
    mode: str            # "fast" | "normal" | "deep"
    confidence: float
    entropy: float
    exit_layer: int
    num_samples: int


class EntropyEstimator(nn.Module):
    """
    
    """

    def __init__(self, d_model: int, vocab_size: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        

        self.exit_every = 4
        exit_layers = list(range(self.exit_every - 1, num_layers, self.exit_every))
        
        self.exit_heads = nn.ModuleDict({
            str(i): nn.Sequential(
                nn.RMSNorm(d_model),
                nn.Linear(d_model, vocab_size, bias=False)
            )
            for i in exit_layers
        })
        

        self.fast_threshold = 0.85
        self.deep_threshold = 0.45

    def estimate_at_layer(self, hidden: torch.Tensor,
                          layer_idx: int) -> Optional[Tuple[torch.Tensor, float]]:
        """
        
        """
        key = str(layer_idx)
        if key not in self.exit_heads:
            return None
        

        pooled = hidden.mean(dim=1)              # (B, d_model)
        logits = self.exit_heads[key](pooled)    # (B, vocab_size)
        
        probs = F.softmax(logits, dim=-1)
        

        entropy = -(probs * (probs + 1e-9).log()).sum(dim=-1).mean().item()
        max_entropy = math.log(logits.shape[-1])
        

        confidence = 1.0 - entropy / max_entropy
        
        return logits, confidence


class RepresentationDriftDetector(nn.Module):
    """
    
    """

    def __init__(self, d_model: int, window: int = 3):
        super().__init__()
        self.window = window
        self.history: List[torch.Tensor] = []
        

        self.drift_proj = nn.Linear(d_model, 1, bias=True)
        nn.init.zeros_(self.drift_proj.weight)
        nn.init.constant_(self.drift_proj.bias, 0.5)

    def update(self, hidden: torch.Tensor) -> float:
        """
        """
        pooled = hidden.mean(dim=1).detach()  # (B, d_model)
        
        if self.history:

            prev = self.history[-1]
            cos_sim = F.cosine_similarity(pooled, prev, dim=-1).mean().item()
            drift = (1.0 - cos_sim) / 2.0
        else:
            drift = 1.0
        
        self.history.append(pooled)
        if len(self.history) > self.window:
            self.history.pop(0)
        
        return drift

    def reset(self):
        self.history.clear()


class TreeSearchSampler:
    """
    
    
    """

    def __init__(self, num_beams: int = 4, temperature: float = 0.7):
        self.num_beams = num_beams
        self.temperature = temperature

    def sample_candidates(self, logits: torch.Tensor,
                          top_k: int = 50) -> List[int]:
        """
        """

        top_k_logits, top_k_ids = torch.topk(logits, min(top_k, logits.shape[-1]))
        

        scaled = top_k_logits / self.temperature
        probs = F.softmax(scaled, dim=-1)
        

        samples = torch.multinomial(probs, num_samples=self.num_beams, replacement=True)
        return [top_k_ids[s].item() for s in samples]

    @staticmethod
    def majority_vote(candidates: List[int]) -> int:
        from collections import Counter
        return Counter(candidates).most_common(1)[0][0]


class CATTSDispatcher(nn.Module):
    """
    
    
    
        dispatcher = CATTSDispatcher(cfg)

        decision = dispatcher.decide(hidden, layer_idx)
        if decision.mode == "fast": break
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


        self.entropy_estimator = EntropyEstimator(d_model, vocab_size, num_layers)
        self.drift_detector = RepresentationDriftDetector(d_model)
        self.tree_sampler = TreeSearchSampler(num_beams=num_deep_samples)


        self._stats: Dict[str, int] = {"fast": 0, "normal": 0, "deep": 0}

    def reset_per_token(self):
        self.drift_detector.reset()

    def check_layer(self, hidden: torch.Tensor,
                    layer_idx: int) -> Optional[DispatchDecision]:
        """
        
        """

        drift = self.drift_detector.update(hidden)


        exit_result = self.entropy_estimator.estimate_at_layer(hidden, layer_idx)
        if exit_result is None:
            return None

        _, confidence = exit_result



        adjusted_confidence = confidence * (1.0 - 0.3 * drift)


        if adjusted_confidence >= self.fast_threshold:
            self._stats["fast"] += 1
            return DispatchDecision(
                mode="fast",
                confidence=adjusted_confidence,
                entropy=1.0 - confidence,
                exit_layer=layer_idx,
                num_samples=1
            )


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

        return None

    def get_stats(self) -> Dict[str, float]:
        total = max(1, sum(self._stats.values()))
        return {k: v / total for k, v in self._stats.items()}

    def reset_stats(self):
        self._stats = {"fast": 0, "normal": 0, "deep": 0}


class AdaptiveNovaMindWrapper(nn.Module):
    """
    
    
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
        
            logits, decision (DispatchDecision), exit_layer
        """
        if not use_catts:

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

            if str(i) in self.model.titans_modules:
                fifo_buf = updated_fifos.get(str(i))
                x, new_fifo = self.model.titans_modules[str(i)](x, fifo_buf)
                updated_fifos[str(i)] = new_fifo


            x, updated_states[i] = layer(
                x,
                state=updated_states[i],
                inference_mode=True
            )


            decision = self.dispatcher.check_layer(x, i)
            if decision is not None and decision.mode in ("fast", "normal"):

                break

        wsra_outputs = None
        if self.model.wsra is not None:
            x, wsra_outputs = self.model.wsra(x, wsra_state)
            wsra_state = wsra_outputs["state"]


        x = self.model.final_norm(x)

        if decision is not None and decision.mode == "deep":

            logits_list = []
            for _ in range(decision.num_samples):

                x_perturb = x + torch.randn_like(x) * 0.01
                logits_list.append(self.model.lm_head(x_perturb))

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
