"""
Full NovaMind language model assembly.

Combines embeddings, hybrid reasoning blocks, optional logic constraints, and
the final language-model head into a single end-to-end architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict

from novamind.config import NovaMindConfig
from novamind.core.mamba_backbone import build_mamba_backbone
from novamind.core.xlstm import xLSTMBlock
from novamind.core.causal import CausalModulatedAttention, LogicConstraintLayer
from novamind.core.wsra import WSRAReasoningStack
from novamind.memory.titans import TITANSMemoryModule


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class FeedForward(nn.Module):

    def __init__(self, d_model: int, expand: int = 4):
        super().__init__()
        d_hidden = int(d_model * expand * 2 / 3)
        d_hidden = ((d_hidden + 63) // 64) * 64

        self.gate_proj = nn.Linear(d_model, d_hidden, bias=False)
        self.up_proj = nn.Linear(d_model, d_hidden, bias=False)
        self.down_proj = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class HybridBlock(nn.Module):
    """
    
    """

    def __init__(self, config: NovaMindConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        d = config.hidden_dim

        # Pre-norms
        self.norm_ssm = RMSNorm(d, config.norm_eps)
        self.norm_xlstm = RMSNorm(d, config.norm_eps)
        self.norm_attn = RMSNorm(d, config.norm_eps)
        self.norm_ffn = RMSNorm(d, config.norm_eps)


        self.ssm = build_mamba_backbone(
            d_model=d,
            d_state=config.ssm_state_dim,
            d_conv=config.ssm_conv_kernel,
            expand=config.ssm_expand,
            num_heads=config.num_ssm_heads,
        )

        self.xlstm = xLSTMBlock(
            d_model=d,
            d_head=config.xlstm_matrix_dim,
            num_heads=config.xlstm_num_heads
        )

        self.attn = None
        if config.use_attention_branch and config.num_attn_heads > 0:
            self.attn = CausalModulatedAttention(
                d_model=d,
                num_heads=config.num_attn_heads,
                causal_dim=config.causal_graph_dim,
                attention_window=config.attention_window,
            )

        # FFN
        self.ffn = FeedForward(d)


        self.num_branches = 3 if self.attn is not None else 2
        self.fusion_gate = nn.Linear(d, self.num_branches, bias=True)
        nn.init.constant_(self.fusion_gate.bias, 1 / self.num_branches)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)


        self.use_causal_modulation = config.use_causal_modulation

    def forward(self, x: torch.Tensor,
                state: Optional[Dict] = None,
                xlstm_state=None,
                inference_mode: bool = False) -> Tuple[torch.Tensor, dict]:
        """
        x: (B, L, d_model)
        """
        if state is None:
            state = {}
        if xlstm_state is not None and "xlstm" not in state:
            state["xlstm"] = xlstm_state


        ssm_out, new_ssm_state = self.ssm(
            self.norm_ssm(x),
            inference_mode=inference_mode,
            states=state.get("ssm"),
            return_states=True,
        )


        xlstm_out, new_xlstm_state = self.xlstm(
            self.norm_xlstm(x), state=state.get("xlstm")
        )


        attn_out = None
        new_attn_state = state.get("attn")
        if self.attn is not None:
            attn_out = self.attn(
                self.norm_attn(x),
                use_causal_modulation=self.use_causal_modulation,
                memory=state.get("attn")
            )
            new_attn_state = x.detach() if state.get("attn") is None else torch.cat(
                [state["attn"], x.detach()], dim=1
            )[:, -self.attn.attention_window:]


        gates = F.softmax(self.fusion_gate(x.mean(dim=1, keepdim=True)), dim=-1)
        branch_outputs = [ssm_out, xlstm_out]
        if attn_out is not None:
            branch_outputs.append(attn_out)

        fused = 0
        for idx, branch in enumerate(branch_outputs):
            fused = fused + gates[..., idx:idx+1] * branch

        x = x + self.dropout(fused)


        x = x + self.dropout(self.ffn(self.norm_ffn(x)))

        return x, {
            "ssm": new_ssm_state,
            "xlstm": new_xlstm_state,
            "attn": new_attn_state,
        }


class NovaMind(nn.Module):
    """
    
    """

    def __init__(self, config: NovaMindConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        nn.init.normal_(self.embed.weight, std=0.02)


        self.layers = nn.ModuleList([
            HybridBlock(config, i) for i in range(config.num_layers)
        ])


        self.titans_modules = nn.ModuleDict()
        for i in range(0, config.num_layers, config.titans_memory_layers):
            self.titans_modules[str(i)] = TITANSMemoryModule(
                d_model=config.hidden_dim,
                surprise_threshold=config.titans_surprise_threshold,
                memory_lr=config.titans_memory_lr,
                summary_slots=config.titans_summary_slots,
                summary_momentum=config.titans_summary_momentum,
                summary_temperature=config.titans_summary_temperature,
            )


        self.logic_layer = None
        if config.use_logic_layer:
            self.logic_layer = LogicConstraintLayer(
                d_model=config.hidden_dim,
                tnorm_type=config.logic_tnorm
            )

        self.wsra = None
        if config.use_wsra:
            self.wsra = WSRAReasoningStack(
                d_model=config.hidden_dim,
                world_slots=config.wsra_world_slots,
                jit_rank=config.wsra_jit_rank,
                frustum_tokens=config.wsra_frustum_tokens,
                proof_branches=config.wsra_proof_branches,
                compile_threshold=config.wsra_compile_threshold,
            )


        self.final_norm = RMSNorm(config.hidden_dim, config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)


        if config.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self,
                input_ids: torch.Tensor,
                labels: Optional[torch.Tensor] = None,
                xlstm_states: Optional[List] = None,
                layer_states: Optional[List[Optional[Dict]]] = None,
                titans_fifos: Optional[dict] = None,
                wsra_state: Optional[Dict] = None,
                inference_mode: bool = False,
                return_logic_loss: bool = False):
        """
        input_ids: (B, L)
        
            - logits: (B, L, vocab_size)
        """
        B, L = input_ids.shape
        device = input_ids.device


        if layer_states is None:
            if xlstm_states is not None:
                layer_states = [
                    {"xlstm": state} if state is not None else None
                    for state in xlstm_states
                ]
            else:
                layer_states = [None] * self.config.num_layers
        if titans_fifos is None:
            titans_fifos = {}

        # Embedding
        x = self.embed(input_ids)  # (B, L, hidden_dim)

        new_layer_states = []
        new_titans_fifos = {}
        logic_satisfactions = []
        wsra_outputs = None


        for i, layer in enumerate(self.layers):

            if str(i) in self.titans_modules:
                fifo_key = str(i)
                fifo_buf = titans_fifos.get(fifo_key, None)
                x, new_fifo = self.titans_modules[fifo_key](x, fifo_buf)
                new_titans_fifos[fifo_key] = new_fifo


            x, new_state = layer(
                x,
                state=layer_states[i],
                inference_mode=inference_mode
            )
            new_layer_states.append(new_state)

        logic_loss = None
        if self.wsra is not None:
            x, wsra_outputs = self.wsra(x, wsra_state)
            if return_logic_loss:
                logic_loss = wsra_outputs["loss"].mean()


        if self.logic_layer is not None:
            x, satisfaction = self.logic_layer(x)
            logic_satisfactions.append(satisfaction)
            if return_logic_loss:
                layer_logic_loss = (1 - satisfaction).mean()
                logic_loss = layer_logic_loss if logic_loss is None else logic_loss + layer_logic_loss


        x = self.final_norm(x)
        hidden_states = x
        logits = self.lm_head(x)  # (B, L, vocab_size)


        loss = None
        if labels is not None:

            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
            if logic_loss is not None:
                loss = loss + 0.01 * logic_loss

        return {
            "logits": logits,
            "hidden_states": hidden_states,
            "loss": loss,
            "logic_loss": logic_loss,
            "xlstm_states": [state["xlstm"] for state in new_layer_states],
            "layer_states": new_layer_states,
            "titans_fifos": new_titans_fifos,
            "wsra": wsra_outputs,
            "wsra_state": None if wsra_outputs is None else wsra_outputs["state"],
        }

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor,
                 max_new_tokens: int = 200,
                 temperature: float = 1.0,
                 top_p: float = 0.9,
                 top_k: int = 50) -> torch.Tensor:
        """
        """
        self.eval()
        B = input_ids.shape[0]
        device = input_ids.device

        layer_states = None
        titans_fifos = {}
        wsra_state = None


        out = self.forward(
            input_ids,
            layer_states=layer_states,
            titans_fifos=titans_fifos,
            wsra_state=wsra_state,
            inference_mode=True
        )
        layer_states = out["layer_states"]
        titans_fifos = out["titans_fifos"]
        wsra_state = out["wsra_state"]
        next_token_logits = out["logits"][:, -1, :]

        generated = [input_ids]

        for _ in range(max_new_tokens):

            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature


            if top_k > 0:
                top_k_vals = torch.topk(next_token_logits, top_k)[0]
                next_token_logits = next_token_logits.masked_fill(
                    next_token_logits < top_k_vals[:, -1:], float("-inf")
                )


            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_token_logits, descending=True)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove_mask = cumprobs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove_mask] = float("-inf")
                next_token_logits.scatter_(1, sorted_idx, sorted_logits)

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            generated.append(next_token)


            out = self.forward(
                next_token,
                layer_states=layer_states,
                titans_fifos=titans_fifos,
                wsra_state=wsra_state,
                inference_mode=True
            )
            layer_states = out["layer_states"]
            titans_fifos = out["titans_fifos"]
            wsra_state = out["wsra_state"]
            next_token_logits = out["logits"][:, -1, :]

        return torch.cat(generated, dim=1)

    def num_parameters(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
