"""
NovaMind quick smoke test.

Verifies that all modules import cleanly and complete a forward pass without
runtime errors.

Run: python test_novamind.py
"""
import sys
import traceback
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

print(f"Device: {DEVICE}, dtype: {DTYPE}\n")

results = {}


try:
    from novamind.config import NovaMindConfig
    cfg_3b = NovaMindConfig.from_size("3b")
    cfg_7b = NovaMindConfig.from_size("7b")
    assert cfg_3b.hidden_dim == 2560
    assert cfg_7b.hidden_dim == 4096
    results["Config"] = "✓"
except Exception as e:
    results["Config"] = f"✗ {e}"


try:
    from novamind.core.ssm import SelectiveSSM, MultiHeadSSM
    ssm = SelectiveSSM(d_model=128, d_state=16).to(DEVICE).to(DTYPE)
    x = torch.randn(2, 64, 128, device=DEVICE, dtype=DTYPE)
    y = ssm(x)
    assert y.shape == (2, 64, 128), f"shape mismatch: {y.shape}"
    results["SelectiveSSM"] = "✓"

    mssm = MultiHeadSSM(d_model=256, num_heads=4, d_state=16).to(DEVICE).to(DTYPE)
    x2 = torch.randn(2, 64, 256, device=DEVICE, dtype=DTYPE)
    y2 = mssm(x2)
    assert y2.shape == (2, 64, 256)
    results["MultiHeadSSM"] = "✓"
except Exception as e:
    results["SSM"] = f"✗ {e}"
    traceback.print_exc()


try:
    from novamind.core.xlstm import xLSTMBlock
    xlstm = xLSTMBlock(d_model=128, d_head=32, num_heads=2).to(DEVICE).to(DTYPE)
    x = torch.randn(2, 32, 128, device=DEVICE, dtype=DTYPE)
    y, state = xlstm(x)
    assert y.shape == (2, 32, 128)

    y2, state2 = xlstm(x, state=state)
    assert y2.shape == (2, 32, 128)
    results["xLSTMBlock"] = "✓"
except Exception as e:
    results["xLSTMBlock"] = f"✗ {e}"
    traceback.print_exc()


try:
    from novamind.core.causal import TNorm, LogicConstraintLayer
    tnorm = TNorm("product")
    a = torch.tensor([0.8, 0.6, 0.3])
    b = torch.tensor([0.5, 0.7, 0.9])
    and_result = tnorm.and_(a, b)
    assert and_result.shape == (3,)
    assert (and_result <= torch.min(a, b) + 0.01).all()

    logic = LogicConstraintLayer(d_model=64, num_predicates=8).to(DEVICE).to(DTYPE)
    x = torch.randn(2, 16, 64, device=DEVICE, dtype=DTYPE)
    x_out, sat = logic(x)
    assert x_out.shape == x.shape
    assert (sat >= 0).all() and (sat <= 1).all()
    results["TNorm + LogicLayer"] = "✓"
except Exception as e:
    results["TNorm + LogicLayer"] = f"✗ {e}"
    traceback.print_exc()


try:
    from novamind.core.causal import CausalModulatedAttention
    attn = CausalModulatedAttention(d_model=128, num_heads=4, causal_dim=16).to(DEVICE).to(DTYPE)
    x = torch.randn(2, 32, 128, device=DEVICE, dtype=DTYPE)
    y = attn(x, use_causal_modulation=True)
    assert y.shape == (2, 32, 128)
    results["CausalModulatedAttn"] = "✓"
except Exception as e:
    results["CausalModulatedAttn"] = f"✗ {e}"
    traceback.print_exc()


try:
    from novamind.memory.titans import TITANSMemoryModule
    titans = TITANSMemoryModule(
        d_model=128, fifo_size=64, memory_hidden=64, memory_lr=0.01
    ).to(DEVICE).to(DTYPE)
    x = torch.randn(2, 32, 128, device=DEVICE, dtype=DTYPE)
    y, fifo = titans(x)
    assert y.shape == (2, 32, 128)
    assert fifo is not None

    y2, fifo2 = titans(x, fifo_buffer=fifo)
    assert y2.shape == (2, 32, 128)
    results["TITANSMemory"] = "✓"
except Exception as e:
    results["TITANSMemory"] = f"✗ {e}"
    traceback.print_exc()


try:
    from novamind.learning.wsc import WSCOptimizer
    simple_model = torch.nn.Linear(64, 64).to(DEVICE)
    base_opt = torch.optim.AdamW(simple_model.parameters(), lr=1e-3)
    wsc = WSCOptimizer(simple_model, base_opt, reset_freq=5, reset_ratio=0.1)
    for _ in range(6):
        x_tmp = torch.randn(4, 64, device=DEVICE)
        loss = simple_model(x_tmp).sum()
        loss.backward()
        wsc.step()
        wsc.zero_grad()
    results["WSCOptimizer"] = "✓"
except Exception as e:
    results["WSCOptimizer"] = f"✗ {e}"
    traceback.print_exc()


try:
    from novamind.training.lora import inject_lora, freeze_base_model, estimate_vram
    test_model = torch.nn.Sequential(
        torch.nn.Linear(64, 128),
        torch.nn.Linear(128, 64)
    )
    test_model = inject_lora(test_model, target_modules=["0", "1"], rank=8, alpha=16)
    test_model = freeze_base_model(test_model)
    trainable = sum(p.numel() for p in test_model.parameters() if p.requires_grad)
    assert trainable > 0

    est = estimate_vram(7_000_000_000, hidden_dim=4096, num_layers=32)
    assert "total_estimated_gb" in est
    results["LoRA + VRAM estimate"] = "✓"
except Exception as e:
    results["LoRA + VRAM estimate"] = f"✗ {e}"
    traceback.print_exc()


try:
    from novamind.config import NovaMindConfig
    from novamind.model import NovaMind


    tiny_cfg = NovaMindConfig(
        vocab_size=1000,
        hidden_dim=128,
        num_layers=2,
        num_ssm_heads=4,
        num_attn_heads=2,
        head_dim=32,
        ssm_state_dim=16,
        xlstm_matrix_dim=32,
        xlstm_num_heads=2,
        causal_graph_dim=16,
        logic_entity_dim=32,
        titans_memory_layers=1,
        use_logic_layer=True,
        use_wsra=True,
        wsra_world_slots=4,
        wsra_jit_rank=8,
        wsra_frustum_tokens=8,
        wsra_proof_branches=3,
        tie_embeddings=True,
    )
    tiny_model = NovaMind(tiny_cfg).to(DEVICE).to(DTYPE)
    input_ids = torch.randint(0, 1000, (2, 16), device=DEVICE)
    labels = input_ids.clone()

    out = tiny_model(input_ids=input_ids, labels=labels)
    assert "loss" in out and out["loss"] is not None
    assert out["logits"].shape == (2, 16, 1000)
    assert out["loss"].item() > 0
    assert out["wsra"] is not None

    n_params = tiny_model.num_parameters()
    print(f"\n[Full Model] Parameter count: {n_params:,}")
    results["Full NovaMind Model"] = "✓"
except Exception as e:
    results["Full NovaMind Model"] = f"✗ {e}"
    traceback.print_exc()


try:
    input_ids = torch.randint(0, 1000, (1, 8), device=DEVICE)
    with torch.no_grad():
        generated = tiny_model.generate(input_ids, max_new_tokens=10, temperature=0.8)
    assert generated.shape[1] == 8 + 10
    results["Generation"] = "✓"
except Exception as e:
    results["Generation"] = f"✗ {e}"
    traceback.print_exc()


try:
    from novamind.core.causal import CausalModulatedAttention, CounterfactualIntervention
    attn = CausalModulatedAttention(
        d_model=64, num_heads=4, causal_dim=8, attention_window=8
    ).to(DEVICE).to(DTYPE)
    x = torch.randn(1, 6, 64, device=DEVICE, dtype=DTYPE)
    cf = attn.simulate_counterfactual(
        x,
        [CounterfactualIntervention(source_index=0, target_index=2, strength=1.0, mode="set")]
    )
    assert cf["counterfactual"].shape == x.shape
    assert cf["delta_norm"].shape == (1,)
    results["Counterfactual ECAM"] = "✓"
except Exception as e:
    results["Counterfactual ECAM"] = f"✗ {e}"
    traceback.print_exc()


try:
    from novamind.training.quantization import apply_qat_quantization
    qat_model = torch.nn.Sequential(
        torch.nn.Linear(64, 128),
        torch.nn.GELU(),
        torch.nn.Linear(128, 64),
    )
    qat_model, replaced = apply_qat_quantization(
        qat_model,
        clip_value=4.0,
        smooth_alpha=0.5,
        target_keywords=["0", "2"],
    )
    x = torch.randn(2, 16, 64)
    y = qat_model(x)
    assert y.shape == (2, 16, 64)
    assert replaced == 2
    results["QAT Quantization"] = "✓"
except Exception as e:
    results["QAT Quantization"] = f"✗ {e}"
    traceback.print_exc()


try:
    from novamind.training.bitlinear import BitLinear158
    bitlinear = BitLinear158(32, 16, bias=True)
    x = torch.randn(2, 5, 32)
    y_train = bitlinear(x)
    bitlinear.eval()
    bitlinear.compile_circuit()
    y_add = bitlinear(x, additive_inference=True)
    assert y_train.shape == y_add.shape == (2, 5, 16)
    results["BitLinear158"] = "✓"
except Exception as e:
    results["BitLinear158"] = f"✗ {e}"
    traceback.print_exc()


try:
    from novamind.training.unified_offload import UnifiedMemoryOffloader
    layers = [torch.nn.Linear(16, 16), torch.nn.Linear(16, 16)]
    offloader = UnifiedMemoryOffloader(execution_device=DEVICE, offload_device="cpu")
    offloader.register_layers(layers)
    assert offloader.model_resident_bytes() > 0
    out = offloader.stream_step(layers[0], torch.randn(2, 16))
    assert out.shape == (2, 16)
    results["Unified Offload"] = "✓"
except Exception as e:
    results["Unified Offload"] = f"✗ {e}"
    traceback.print_exc()


try:
    from novamind.core.code_mcts import CodeMCTSReasoner, PythonSandbox

    def proposal_fn(prompt, partial_code, block_index, num_options):
        if block_index == 0:
            return [{"block": "def solve(n):\n    pass  # TODO: next_block", "is_terminal": False}]
        return [
            {"block": "return n +", "is_terminal": True},
            {"block": "while True:\n        pass", "is_terminal": True},
            {"block": "return n * 2", "is_terminal": True},
        ][:num_options]

    reasoner = CodeMCTSReasoner(
        proposal_fn=proposal_fn,
        sandbox=PythonSandbox(timeout_s=0.2),
        num_options=3,
        max_rollouts=2,
    )
    result = reasoner.run(
        "Write solve(n) that doubles n",
        tests=["assert solve(3) == 6", "assert solve(0) == 0"]
    )
    assert "return n * 2" in result["code"]
    assert result["result"].status.startswith("passed")
    assert result["block_depth"] >= 2
    results["Code MCTS"] = "✓"
except Exception as e:
    results["Code MCTS"] = f"✗ {e}"
    traceback.print_exc()


try:
    from novamind.core.code_mcts import PythonSandbox
    import time

    sandbox = PythonSandbox(timeout_s=0.5)
    t0 = time.perf_counter()
    result = sandbox.run("def solve(:\n    pass", tests=["assert True"])
    elapsed = time.perf_counter() - t0
    assert result.stage == "lint"
    assert result.reward == -10.0
    assert result.early_exit is True
    assert elapsed < 0.1
    results["Sandbox Early Exit Lint"] = "✓"
except Exception as e:
    results["Sandbox Early Exit Lint"] = f"✗ {e}"
    traceback.print_exc()


try:
    from novamind.core.code_mcts import PythonSandbox

    sandbox = PythonSandbox(timeout_s=0.5)
    result = sandbox.run("import definitely_missing_module\n", tests=[])
    assert result.stage == "compile_import"
    assert result.reward == -5.0
    assert result.early_exit is True
    results["Sandbox Early Exit Import"] = "✓"
except Exception as e:
    results["Sandbox Early Exit Import"] = f"✗ {e}"
    traceback.print_exc()


try:
    from data.synthetic_curriculum import get_level_1_dataset

    dataset = get_level_1_dataset()
    assert len(dataset) >= 5
    for sample in dataset:
        assert "prompt" in sample and sample["prompt"]
        assert "tests" in sample and sample["tests"]
        assert "reference_code" in sample and sample["reference_code"]
    results["Synthetic Curriculum"] = "✓"
except Exception as e:
    results["Synthetic Curriculum"] = f"✗ {e}"
    traceback.print_exc()


try:
    import ast
    from data.synthetic_curriculum import get_level_1_dataset

    dataset = get_level_1_dataset()
    for sample in dataset:
        ast.parse(sample["reference_code"])
        for line in sample["tests"].splitlines():
            if line.strip():
                ast.parse(line)
    results["Synthetic Curriculum AST"] = "✓"
except Exception as e:
    results["Synthetic Curriculum AST"] = f"✗ {e}"
    traceback.print_exc()


try:
    from novamind.core.lod_router import get_lod_tier, lod_compute

    assert get_lod_tier() == "high_fi"
    with lod_compute("low_fi"):
        assert get_lod_tier() == "low_fi"
        with lod_compute("high_fi"):
            assert get_lod_tier() == "high_fi"
        assert get_lod_tier() == "low_fi"
    assert get_lod_tier() == "high_fi"
    results["LOD Router"] = "✓"
except Exception as e:
    results["LOD Router"] = f"✗ {e}"
    traceback.print_exc()


try:
    from novamind.core.code_mcts import CodeMCTSReasoner, PythonSandbox
    from novamind.core.lod_router import get_lod_tier

    observed = {}

    def proposal_fn(prompt, partial_code, block_index, num_options):
        if block_index == 0:
            return [{"block": "def solve(n):\n    return n * 2", "is_terminal": True}]
        return []

    def high_fi_backprop_fn(prompt, code):
        observed["tier"] = get_lod_tier()
        observed["code"] = code
        return {"loss": 0.123}

    reasoner = CodeMCTSReasoner(
        proposal_fn=proposal_fn,
        sandbox=PythonSandbox(timeout_s=0.2),
        high_fi_backprop_fn=high_fi_backprop_fn,
        num_options=1,
        max_rollouts=1,
    )
    result = reasoner.run(
        "Write solve(n) that doubles n",
        tests=["assert solve(3) == 6"],
    )
    assert observed["tier"] == "high_fi"
    assert result["high_fi_backprop"]["loss"] == 0.123
    assert "return n * 2" in observed["code"]
    results["MCTS High-Fi Backprop"] = "✓"
except Exception as e:
    results["MCTS High-Fi Backprop"] = f"✗ {e}"
    traceback.print_exc()


try:
    from data.synthetic_curriculum import get_secure_vault_dataset

    dataset = get_secure_vault_dataset(3)
    assert len(dataset) == 3
    for sample in dataset:
        assert sample["task_name"].startswith("vault_")
        assert "prompt" in sample and sample["prompt"]
        assert "tests" in sample and sample["tests"]
        assert "correct_block" in sample and sample["correct_block"]
        assert "poisoned_block" in sample and sample["poisoned_block"]
    results["Vault Curriculum"] = "✓"
except Exception as e:
    results["Vault Curriculum"] = f"✗ {e}"
    traceback.print_exc()


try:
    from novamind.core.code_mcts import PythonSandbox

    sandbox = PythonSandbox(timeout_s=0.5)
    code = (
        "def solve():\n"
        "    values = [1]\n"
        "    for idx in range(len(values) + 1):\n"
        "        item = values[idx]\n"
        "    return 1\n"
    )
    result = sandbox.run(code, tests=["assert solve() == 1"])
    assert result.failing_line_number is not None
    assert result.ast_status != "clean"
    assert result.rollback_reward is not None
    results["Sandbox AST Rollback"] = "✓"
except Exception as e:
    results["Sandbox AST Rollback"] = f"✗ {e}"
    traceback.print_exc()


try:
    from novamind.core.symbol_sentinel import SymbolSentinel

    sentinel = SymbolSentinel()
    undefined = sentinel.get_undefined_references("x = 1\nz = x + y\n")
    assert "y" in undefined
    assert "x" not in undefined
    results["Symbol Sentinel"] = "✓"
except Exception as e:
    results["Symbol Sentinel"] = f"✗ {e}"
    traceback.print_exc()


try:
    from novamind.core.symbol_sentinel import SymbolSentinel

    sentinel = SymbolSentinel()
    assert sentinel.get_undefined_references("x = 1\n") == set()
    undefined = sentinel.get_undefined_references("x = 1\nz = x + y\n")
    assert undefined == {"y"}
    results["Symbol Sentinel Incremental"] = "✓"
except Exception as e:
    results["Symbol Sentinel Incremental"] = f"✗ {e}"
    traceback.print_exc()


try:
    from novamind.training.gradient_surgery import apply_surgical_mask

    probe_model = torch.nn.Linear(4, 4)
    probe_model._sga_target_tensor = torch.randn(1, 4, 4, requires_grad=True)
    dummy_loss = probe_model._sga_target_tensor.sum()
    surgery = apply_surgical_mask(dummy_loss, probe_model, failing_line_index=2, total_tokens=4)
    if DEVICE == "cuda":
        assert "applied" in surgery
    else:
        assert surgery["applied"] is False
    results["Gradient Surgery"] = "✓"
except Exception as e:
    results["Gradient Surgery"] = f"✗ {e}"
    traceback.print_exc()


try:
    from novamind.core.met_controller import METGate, calculate_entropy

    low_entropy_logits = torch.tensor([[20.0, -20.0, -20.0]])
    high_entropy_logits = torch.zeros(1, 8)
    gate = METGate(entropy_threshold=1.0)
    assert calculate_entropy(high_entropy_logits) > calculate_entropy(low_entropy_logits)
    assert gate.should_trigger_system2(high_entropy_logits) is True
    assert gate.should_trigger_system2(low_entropy_logits) is False
    results["MET Gate"] = "✓"
except Exception as e:
    results["MET Gate"] = f"✗ {e}"
    traceback.print_exc()


try:
    from novamind.core.met_controller import MetStateTracker

    tracker = MetStateTracker(entropy_threshold=1.0, caution_window=2)
    high_entropy_logits = torch.zeros(1, 8)
    low_entropy_logits = torch.tensor([[20.0, -20.0, -20.0]])

    first = tracker.observe(high_entropy_logits)
    second = tracker.observe(low_entropy_logits)
    third = tracker.observe(low_entropy_logits)

    assert first["trigger_system2"] is True
    assert first["forced_by_inertia"] is False
    assert second["trigger_system2"] is True
    assert second["forced_by_inertia"] is True
    assert third["trigger_system2"] is False
    ratio = tracker.compression_ratio()
    assert ratio["system1_tokens"] == 1
    assert ratio["system2_tokens"] == 2
    results["MET Inertial Gate"] = "✓"
except Exception as e:
    results["MET Inertial Gate"] = f"✗ {e}"
    traceback.print_exc()



print("\n" + "="*50)
print("NovaMind Test Results")
print("="*50)
passed = sum(1 for v in results.values() if v.startswith("✓"))
total = len(results)
for k, v in results.items():
    print(f"  {v}  {k}")
print(f"\nPassed: {passed}/{total}")
if passed == total:
    print("🟢 All tests passed!")
else:
    print("🔴 Some tests failed. See the traceback above.")
