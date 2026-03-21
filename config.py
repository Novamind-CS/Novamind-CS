"""
NovaMind — 核心配置
所有超参数集中在这里，不要在各模块里散落 magic number
"""
import math
from dataclasses import dataclass, field, replace
from typing import Optional


@dataclass
class NovaMindConfig:
    # ── 模型规模 ──────────────────────────────────────────
    vocab_size: int = 32000
    hidden_dim: int = 4096          # 7B 用 4096；3B 用 2560；14B 用 5120
    num_layers: int = 32
    max_seq_len: int = 131072       # 128K 默认，TITANS 可扩展到 2M+

    # ── 混合头比例 ────────────────────────────────────────
    # 每个 block 内 SSM 头与注意力头的数量
    num_ssm_heads: int = 24         # 主力：负责全局上下文压缩
    num_attn_heads: int = 8         # 辅助：负责高价值局部精确提取
    head_dim: int = 64
    attention_window: int = 256     # 局部注意力保留的历史跨度
    use_attention_branch: bool = True
    ssm_backend: str = "auto"       # "auto" | "torch" | "mamba2"

    # ── SSM (Mamba-2 风格) ────────────────────────────────
    ssm_state_dim: int = 64         # 隐藏状态维度 d_state
    ssm_conv_kernel: int = 4        # 局部卷积核大小
    ssm_expand: int = 2             # 内部展开倍率

    # ── xLSTM 矩阵记忆 ───────────────────────────────────
    xlstm_matrix_dim: int = 256     # 矩阵记忆的 key/value 维度
    xlstm_num_heads: int = 4

    # ── TITANS 联想记忆 ──────────────────────────────────
    titans_memory_layers: int = 4   # 每隔几层插入一个 TITANS 记忆模块
    titans_surprise_threshold: float = 0.1
    titans_memory_lr: float = 0.01  # 推断期记忆更新学习率
    titans_summary_slots: int = 32
    titans_summary_momentum: float = 0.95
    titans_summary_temperature: float = 0.35

    # ── 因果注意力层 (简化版 ECAM) ───────────────────────
    causal_graph_dim: int = 32      # 局部因果图嵌入维度
    use_causal_modulation: bool = True

    # ── 可微逻辑层 ───────────────────────────────────────
    use_logic_layer: bool = True
    logic_entity_dim: int = 128
    logic_tnorm: str = "product"    # "product" | "lukasiewicz" | "minmax"

    # ── WSRA 推理栈 ──────────────────────────────────────
    use_wsra: bool = True
    wsra_world_slots: int = 8
    wsra_jit_rank: int = 32
    wsra_frustum_tokens: int = 256
    wsra_proof_branches: int = 4
    wsra_compile_threshold: float = 0.55

    # ── 训练 ─────────────────────────────────────────────
    dropout: float = 0.0
    tie_embeddings: bool = True
    norm_eps: float = 1e-5

    # ── 16GB 训练优化 ────────────────────────────────────
    use_activation_checkpointing: bool = True
    offload_activations: bool = True         # 激活卸载到 CPU RAM
    offload_threshold_mb: float = 50.0      # 超过此大小的激活才卸载
    use_unified_offload: bool = False
    unified_offload_device: str = "cpu"
    unified_offload_granularity: str = "layer"

    # ── LoRA ─────────────────────────────────────────────
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list = field(
        default_factory=lambda: ["ssm_B", "ssm_C", "q_proj", "k_proj", "v_proj", "o_proj"]
    )

    # ── WSC 持续学习 ──────────────────────────────────────
    wsc_reset_freq: int = 500       # 每 N 步执行一次参数重置
    wsc_reset_ratio: float = 0.05   # 重置最低贡献的 5% 参数
    wsc_ema_decay: float = 0.999    # 权重平均衰减系数

    # ── CATTS 推理调度 ───────────────────────────────────
    catts_fast_threshold: float = 0.85
    catts_deep_threshold: float = 0.40
    catts_num_deep_samples: int = 4
    tree_search_depth: int = 3
    tree_search_branching: int = 4
    tree_search_beam_width: int = 4

    @classmethod
    def from_size(cls, size: str) -> "NovaMindConfig":
        """预设规格快捷方式"""
        presets = {
            "3b":  dict(hidden_dim=2560, num_layers=32, num_ssm_heads=16, num_attn_heads=4, head_dim=64, attention_window=128),
            "7b":  dict(hidden_dim=4096, num_layers=32, num_ssm_heads=24, num_attn_heads=8, head_dim=128, attention_window=256),
            "14b": dict(hidden_dim=5120, num_layers=40, num_ssm_heads=32, num_attn_heads=8, head_dim=128, attention_window=384),
        }
        assert size in presets, f"size 必须是 {list(presets.keys())} 之一"
        return cls(**presets[size])

    def with_width_multiplier(self, multiplier: float) -> "NovaMindConfig":
        """按宽度扩展模型，同时保持各子模块的整除约束。"""
        if multiplier <= 1.0:
            return self

        multiple = math.lcm(self.num_ssm_heads, self.num_attn_heads, self.xlstm_num_heads, 64)
        new_hidden = int(math.ceil(self.hidden_dim * multiplier / multiple) * multiple)
        new_xlstm_dim = int(math.ceil(self.xlstm_matrix_dim * multiplier / 64) * 64)
        new_jit_rank = int(math.ceil(self.wsra_jit_rank * multiplier / 8) * 8)

        return replace(
            self,
            hidden_dim=new_hidden,
            xlstm_matrix_dim=new_xlstm_dim,
            wsra_jit_rank=new_jit_rank,
            wsra_frustum_tokens=max(self.wsra_frustum_tokens, self.attention_window),
        )
