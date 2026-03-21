"""
NovaMind core configuration.

Keep all primary hyperparameters here rather than scattering magic numbers
across modules.
"""
import math
from dataclasses import dataclass, field, replace
from typing import Optional


@dataclass
class NovaMindConfig:

    vocab_size: int = 32000
    hidden_dim: int = 4096
    num_layers: int = 32
    max_seq_len: int = 131072



    num_ssm_heads: int = 24
    num_attn_heads: int = 8
    head_dim: int = 64
    attention_window: int = 256
    use_attention_branch: bool = True
    ssm_backend: str = "auto"       # "auto" | "torch" | "mamba2"


    ssm_state_dim: int = 64
    ssm_conv_kernel: int = 4
    ssm_expand: int = 2


    xlstm_matrix_dim: int = 256
    xlstm_num_heads: int = 4


    titans_memory_layers: int = 4
    titans_surprise_threshold: float = 0.1
    titans_memory_lr: float = 0.01
    titans_summary_slots: int = 32
    titans_summary_momentum: float = 0.95
    titans_summary_temperature: float = 0.35


    causal_graph_dim: int = 32
    use_causal_modulation: bool = True


    use_logic_layer: bool = True
    logic_entity_dim: int = 128
    logic_tnorm: str = "product"    # "product" | "lukasiewicz" | "minmax"


    use_wsra: bool = True
    wsra_world_slots: int = 8
    wsra_jit_rank: int = 32
    wsra_frustum_tokens: int = 256
    wsra_proof_branches: int = 4
    wsra_compile_threshold: float = 0.55


    dropout: float = 0.0
    tie_embeddings: bool = True
    norm_eps: float = 1e-5


    use_activation_checkpointing: bool = True
    offload_activations: bool = True
    offload_threshold_mb: float = 50.0
    use_unified_offload: bool = False
    unified_offload_device: str = "cpu"
    unified_offload_granularity: str = "layer"


    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list = field(
        default_factory=lambda: ["ssm_B", "ssm_C", "q_proj", "k_proj", "v_proj", "o_proj"]
    )


    wsc_reset_freq: int = 500
    wsc_reset_ratio: float = 0.05
    wsc_ema_decay: float = 0.999


    catts_fast_threshold: float = 0.85
    catts_deep_threshold: float = 0.40
    catts_num_deep_samples: int = 4
    tree_search_depth: int = 3
    tree_search_branching: int = 4
    tree_search_beam_width: int = 4

    @classmethod
    def from_size(cls, size: str) -> "NovaMindConfig":
        """Convenience constructor for preset model sizes."""
        presets = {
            "3b":  dict(hidden_dim=2560, num_layers=32, num_ssm_heads=16, num_attn_heads=4, head_dim=64, attention_window=128),
            "7b":  dict(hidden_dim=4096, num_layers=32, num_ssm_heads=24, num_attn_heads=8, head_dim=128, attention_window=256),
            "14b": dict(hidden_dim=5120, num_layers=40, num_ssm_heads=32, num_attn_heads=8, head_dim=128, attention_window=384),
        }
        assert size in presets, f"size must be one of {list(presets.keys())}"
        return cls(**presets[size])

    def with_width_multiplier(self, multiplier: float) -> "NovaMindConfig":
        """Scale model width while preserving divisibility constraints."""
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
