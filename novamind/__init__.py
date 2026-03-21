"""NovaMind package compatibility layer."""

import importlib
import sys


def _alias(module_name: str, target: str):
    module = importlib.import_module(target)
    sys.modules[module_name] = module
    return module


_alias(__name__ + ".config", "config")
_alias(__name__ + ".core", "core")
_alias(__name__ + ".core.ssm", "core.ssm")
_alias(__name__ + ".core.device_manager", "core.device_manager")
_alias(__name__ + ".core.lod_router", "core.lod_router")
_alias(__name__ + ".core.met_controller", "core.met_controller")
_alias(__name__ + ".core.mamba_backbone", "core.mamba_backbone")
_alias(__name__ + ".core.xlstm", "core.xlstm")
_alias(__name__ + ".core.causal", "core.causal")
_alias(__name__ + ".core.catts", "core.catts")
_alias(__name__ + ".core.wsra", "core.wsra")
_alias(__name__ + ".core.tree_search", "core.tree_search")
_alias(__name__ + ".core.code_mcts", "core.code_mcts")
_alias(__name__ + ".core.ast_rollback", "core.ast_rollback")
_alias(__name__ + ".core.symbol_sentinel", "core.symbol_sentinel")
_alias(__name__ + ".memory", "memory")
_alias(__name__ + ".memory.titans", "memory.titans")
_alias(__name__ + ".learning", "learning")
_alias(__name__ + ".learning.wsc", "learning.wsc")
_alias(__name__ + ".training", "training")
_alias(__name__ + ".training.lora", "training.lora")
_alias(__name__ + ".training.bitlinear", "training.bitlinear")
_alias(__name__ + ".training.quantization", "training.quantization")
_alias(__name__ + ".training.unified_offload", "training.unified_offload")
_alias(__name__ + ".training.gradient_surgery", "training.gradient_surgery")
_alias(__name__ + ".training.rl", "training.rl")

from config import NovaMindConfig
from model import NovaMind

sys.modules[__name__ + ".model"] = importlib.import_module("model")

__version__ = "0.1.0"
__all__ = ["NovaMind", "NovaMindConfig"]
