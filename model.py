"""
NovaMind — 完整模型

把所有模块组装成完整的语言模型:
    Embedding
    → N × HybridBlock (SSM + xLSTM + 因果注意力)
    → 逻辑约束层 (可选)
    → LM Head

HybridBlock 内部结构:
    Input
    → RMSNorm → MultiHeadSSM (全局压缩)
    → RMSNorm → xLSTMBlock (高秩记忆召回)
    → RMSNorm → CausalModulatedAttention (因果调制局部注意)
    → RMSNorm → FFN (特征变换)
    → TITANS 记忆 (每 titans_memory_layers 层插入一个)
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
    """SwiGLU FFN（Llama 风格，参数效率高）"""

    def __init__(self, d_model: int, expand: int = 4):
        super().__init__()
        d_hidden = int(d_model * expand * 2 / 3)  # SwiGLU 标准比例
        d_hidden = ((d_hidden + 63) // 64) * 64   # 对齐到 64 的倍数，GPU 效率

        self.gate_proj = nn.Linear(d_model, d_hidden, bias=False)
        self.up_proj = nn.Linear(d_model, d_hidden, bias=False)
        self.down_proj = nn.Linear(d_hidden, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class HybridBlock(nn.Module):
    """
    核心混合计算块
    
    三路并行处理，然后门控融合:
    - SSM: 全局上下文压缩 (O(1) 内存)
    - xLSTM: 矩阵记忆精确召回
    - 因果注意力: 高价值局部依赖
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

        # 三路处理器
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

        # 门控融合：三路输出的动态权重（不同层可以学到不同的混合策略）
        self.num_branches = 3 if self.attn is not None else 2
        self.fusion_gate = nn.Linear(d, self.num_branches, bias=True)
        nn.init.constant_(self.fusion_gate.bias, 1 / self.num_branches)

        # Dropout
        self.dropout = nn.Dropout(config.dropout)

        # 是否在此层使用因果调制（训练时可以按需关闭节省计算）
        self.use_causal_modulation = config.use_causal_modulation

    def forward(self, x: torch.Tensor,
                state: Optional[Dict] = None,
                xlstm_state=None,
                inference_mode: bool = False) -> Tuple[torch.Tensor, dict]:
        """
        x: (B, L, d_model)
        返回: (output, block_state)
        """
        if state is None:
            state = {}
        if xlstm_state is not None and "xlstm" not in state:
            state["xlstm"] = xlstm_state

        # ① SSM 路
        ssm_out, new_ssm_state = self.ssm(
            self.norm_ssm(x),
            inference_mode=inference_mode,
            states=state.get("ssm"),
            return_states=True,
        )

        # ② xLSTM 路（带矩阵记忆）
        xlstm_out, new_xlstm_state = self.xlstm(
            self.norm_xlstm(x), state=state.get("xlstm")
        )

        # ③ 因果注意力路
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

        # ④ 门控融合
        gates = F.softmax(self.fusion_gate(x.mean(dim=1, keepdim=True)), dim=-1)
        branch_outputs = [ssm_out, xlstm_out]
        if attn_out is not None:
            branch_outputs.append(attn_out)

        fused = 0
        for idx, branch in enumerate(branch_outputs):
            fused = fused + gates[..., idx:idx+1] * branch

        x = x + self.dropout(fused)

        # ⑤ FFN
        x = x + self.dropout(self.ffn(self.norm_ffn(x)))

        return x, {
            "ssm": new_ssm_state,
            "xlstm": new_xlstm_state,
            "attn": new_attn_state,
        }


class NovaMind(nn.Module):
    """
    NovaMind 完整语言模型
    
    特性:
    - 混合 SSM + xLSTM + 因果注意力
    - 每 titans_memory_layers 层插入一个 TITANS 记忆模块
    - 可选逻辑约束层
    - 支持增量推理（传入 kv cache 替代品 = SSM hidden state + TITANS fifo）
    """

    def __init__(self, config: NovaMindConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        nn.init.normal_(self.embed.weight, std=0.02)

        # 主干层
        self.layers = nn.ModuleList([
            HybridBlock(config, i) for i in range(config.num_layers)
        ])

        # TITANS 记忆模块（每 N 层一个）
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

        # 可选逻辑约束层
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

        # 输出
        self.final_norm = RMSNorm(config.hidden_dim, config.norm_eps)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # 权重绑定（节省参数）
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
        labels:    (B, L) — 用于计算语言模型损失（next token prediction）
        
        返回: dict with keys:
            - logits: (B, L, vocab_size)
            - loss: 语言模型损失（如果提供 labels）
            - logic_loss: 逻辑约束损失（如果 return_logic_loss=True）
            - xlstm_states: 更新后的 xLSTM 状态（推理用）
            - titans_fifos: 更新后的 TITANS FIFO 缓冲（推理用）
        """
        B, L = input_ids.shape
        device = input_ids.device

        # 初始化状态
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

        # 逐层处理
        for i, layer in enumerate(self.layers):
            # TITANS 记忆（在每 N 层前执行）
            if str(i) in self.titans_modules:
                fifo_key = str(i)
                fifo_buf = titans_fifos.get(fifo_key, None)
                x, new_fifo = self.titans_modules[fifo_key](x, fifo_buf)
                new_titans_fifos[fifo_key] = new_fifo

            # 混合计算块
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

        # 逻辑约束层
        if self.logic_layer is not None:
            x, satisfaction = self.logic_layer(x)
            logic_satisfactions.append(satisfaction)
            if return_logic_loss:
                layer_logic_loss = (1 - satisfaction).mean()
                logic_loss = layer_logic_loss if logic_loss is None else logic_loss + layer_logic_loss

        # 输出
        x = self.final_norm(x)
        hidden_states = x
        logits = self.lm_head(x)  # (B, L, vocab_size)

        # 语言模型损失
        loss = None
        if labels is not None:
            # next token prediction: 移位对齐
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
            if logic_loss is not None:
                loss = loss + 0.01 * logic_loss  # 逻辑约束作为辅助损失

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
        自回归生成
        推理时使用 SSM 递归模式（O(1) 内存，每步恒定）
        """
        self.eval()
        B = input_ids.shape[0]
        device = input_ids.device

        layer_states = None
        titans_fifos = {}
        wsra_state = None

        # 先处理 prompt
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
            # 采样
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Top-K 过滤
            if top_k > 0:
                top_k_vals = torch.topk(next_token_logits, top_k)[0]
                next_token_logits = next_token_logits.masked_fill(
                    next_token_logits < top_k_vals[:, -1:], float("-inf")
                )

            # Top-P 过滤
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_token_logits, descending=True)
                cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove_mask = cumprobs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove_mask] = float("-inf")
                next_token_logits.scatter_(1, sorted_idx, sorted_logits)

            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            generated.append(next_token)

            # 推进一步
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
