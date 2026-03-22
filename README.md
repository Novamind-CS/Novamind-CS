


# NovaMind

**v1.2: The Hardcore Curriculum.**

**Surgical Reasoning on Consumer Silicon.**  
*Breaking the memory wall with metacognitive gating, causal modulation, and surgical gradients.*

> The next leap in AI will not come solely from bigger models. It will come from systems that know **when to think harder**, **where to spend precision**, and **how to learn only from the part that truly failed**.

---

<p align="center">
  <img
    width="2710"
    height="800"
    alt="Novamind Banner"
    src="https://github.com/user-attachments/assets/09dc25e6-529b-4b00-9872-ceb1f0258d5f"
  />
</p>

## What This Is

NovaMind is a research-grade neuro-symbolic reasoning engine built for one specific, usually ignored hardware regime: **consumer GPUs with strict VRAM ceilings**.

Most LLM architecture assumes datacenters. NovaMind assumes a single RTX 4090, 3090, 5070 Ti, or a Mac M-series machine, and treats that constraint as a design principle rather than an inconvenience. The goal is simple: maximize **verified reasoning per watt**, not leaderboard score under unlimited compute.

This is not a lightly patched Transformer. It is a ground-up attempt to answer a more practical question:

**What does a reasoning-first model look like when memory is finite and every compute cycle matters?**

The current release track is centered on a dual-engine workflow:

- **The Sniper**: a compact reasoning specialist trained on execution-verified Python tasks
- **The Heavyweight**: a 7B LoRA adaptation path for larger base models
- **The Hardcore Curriculum**: a nightmare-grade dataset upgrade covering Tarjan SCC, bitmask DP, tree DP, trie logic, segment-style query workloads, N-Queens, and hard maze pathfinding

---

## Why Current Architectures Feel Wasteful

On constrained hardware, standard Transformer stacks are expensive in all the wrong places:

- **Quadratic attention**: sequence cost explodes with context length
- **KV cache growth**: memory usage expands with every token
- **Dense decoding**: easy tokens and hard tokens are treated with the same budget
- **Flat credit assignment**: one bad line can penalize an entire correct prefix
- **Static precision**: every branch pays FP16/BF16 cost even when it is obviously speculative

NovaMind replaces each of those failure modes with a more selective mechanism.

---

## The Five Pillars

### 1. MET: Metacognitive Entropy Throttling

NovaMind runs two reasoning modes:

- **System 1**: fast continuation
- **System 2**: explicit branching, validation, rollback, and replay

The switch is controlled by Shannon entropy over the next-token distribution:

```text
H(p) = -∑ pᵢ log pᵢ

Trigger System 2 if H(p) > τ
```

Once entropy spikes, the engine stays in System 2 for a short **caution window**. That prevents constant switching during recursion, nested loops, or other unstable local regions.

Most systems overspend uniformly. MET spends compute only where uncertainty actually justifies it.

### 2. CATTS: Confidence-Aware Test-Time Scaling

Inside the model, CATTS provides **layer-level early exit** based on confidence and representation drift.

At intermediate layers, NovaMind estimates:

- output entropy
- internal representation drift

Then it decides whether to:

- **exit early**,
- **continue normally**,
- or **escalate into deeper reasoning**.

```text
adjusted_confidence = base_confidence × (1 - 0.3 × drift_score)

confidence > 0.85  -> early exit
confidence < 0.45  -> tree search
otherwise          -> full forward
```

This is not prompt-level routing. It is **inference-time compute allocation inside the network itself**.

### 3. SGA: Surgical Gradient Attribution

If a program fails at line `ℓ`, NovaMind does not treat every previous token as equally guilty.

Instead it scales gradients around the failure boundary:

```text
g'ᵢ = 0.1 × gᵢ      if i < i_fail
g'ᵢ = 10.0 × gᵢ     if i ≥ i_fail
```

That means:

- correct prefixes are preserved,
- the failing region is corrected aggressively.

This is gradient routing with structural locality, not blunt backprop.

### 4. WSC: Weight Space Consolidation

NovaMind supports **online continual learning** without catastrophic forgetting.

It does that through two mechanisms:

- **SVD-based low-contribution resets** to recover plasticity
- **EMA shadow averaging** to keep the model close to a stable operating point

```text
Reset frequency: every 500 steps
Reset ratio:     lowest 5% singular directions
EMA decay:       0.999
```

The intuition is simple: free unhelpful directions, preserve stable ones.

### 5. TITANS: Inference-Time Associative Memory

Instead of growing a KV cache forever, NovaMind uses a compact associative memory whose **weights act as memory**.

During inference:

- high-surprise tokens are written into memory,
- low-surprise tokens pass through cheaply.

```text
O(1) memory usage with respect to sequence length
2M+ context targets without proportional VRAM growth
```

This is not passive caching. It is compressed, learned, associative recall.

---

## Architecture

```text
Input Tokens
    │
    ▼
Embedding Layer
    │
    ▼  ×N layers
┌─────────────────────────────────────────────────────┐
│  HybridBlock                                        │
│                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │
│  │  Mamba-2    │  │   xLSTM     │  │  Causal    │  │
│  │  SSM Head   │  │  Matrix Mem │  │  Attn Head │  │
│  │  O(1) mem   │  │  High-rank  │  │  ECAM+SCM  │  │
│  └──────┬──────┘  └──────┬──────┘  └─────┬──────┘  │
│         └───────────┬────┘               │         │
│               Gated Fusion ◄─────────────┘         │
│                     │                              │
│              FFN (SwiGLU)                          │
└─────────────────────┬───────────────────────────────┘
                      │
             TITANS Memory          ← every N layers
                      │
             CATTS Dispatcher       ← early exit check
                      │
          Logic Constraint Layer    ← optional
                      │
                 LM Head
```

Every `HybridBlock` runs three parallel paths and merges them with learned gating:

- **SSM path**: global context compression with constant-memory inference
- **xLSTM path**: precise, high-rank matrix recall
- **Causal attention path**: local, high-value dependencies with causal modulation

---

## Project Structure

```text
novamind/
├── config.py           all hyperparameters in one place
├── model.py            full model assembly
├── train.py            training loop with memory-aware optimizations
├── inference.py        inference with CATTS adaptive routing
│
├── core/
│   ├── ssm.py          selective SSM (Mamba-2 style)
│   ├── xlstm.py        matrix-memory recurrent block
│   ├── causal.py       causal attention + differentiable logic
│   ├── catts.py        confidence-aware test-time scaling
│   ├── code_mcts.py    code-level MCTS reasoning engine
│   ├── ast_rollback.py traceback-to-AST rollback analysis
│   └── symbol_sentinel.py incremental symbol-flow guard
│
├── memory/
│   └── titans.py       inference-time associative memory
│
├── learning/
│   └── wsc.py          weight-space consolidation
│
└── training/
    ├── lora.py         LoRA + activation offload + VRAM tools
    ├── bitlinear.py    BitLinear / 1.58-bit routing
    ├── gradient_surgery.py
    └── unified_offload.py
```

---

## 16 GB VRAM Budget

Estimated budget for the **NovaMind 7B configuration** under full LoRA fine-tuning on a single consumer GPU:

| Component | Memory |
|---|---:|
| Model weights (bf16) | ~7.0 GB |
| LoRA adapters (rank=64) | ~0.4 GB |
| Activations (checkpointing on) | ~2.8 GB |
| SSM hidden states | ~0.6 GB |
| Causal graph buffers | ~0.3 GB |
| CUDA misc | ~1.5 GB |
| **Total peak** | **~12.6 - 13.6 GB** |

This section describes the **target 7B architecture budget**, not a bundled pretrained checkpoint.
The important part is not the exact decimal place. It is the design direction: NovaMind spends memory on verified reasoning, not unbounded cache growth.

---

## Quick Comparison

| Dimension | Standard Transformer | NovaMind |
|---|---|---|
| Inference memory | O(N), cache grows with context | **O(1), fixed-state reasoning core** |
| Long-context behavior | memory-bound quickly | **designed for compressed long-context reasoning** |
| Compute allocation | uniform | **entropy-gated and confidence-routed** |
| Credit assignment | flat | **surgical** |
| Continual learning | retrain-heavy | **online-friendly with WSC** |
| 16 GB fine-tuning | usually painful | **first-class target** |
| Reasoning depth | fixed | **dynamic System 1 / System 2** |

---

## Install

```bash
pip install -r requirements.txt
```

Recommended extras:

```bash
pip install mamba-ssm causal-conv1d
pip install bitsandbytes accelerate
```

Or use the setup helper:

```bash
./setup_env.sh
```

---

## Quick Start

### One-click 16GB launch

For the paper-launch workflow on an RTX 5070 Ti class 16 GB GPU:

```bash
./launch_novamind.sh
```

That script will:

- generate `data/sniper_train.jsonl`
- run `pretrain_tiny` with a 16 GB-safe recipe
- optionally continue into `finetune_lora_7b`

The locked 16 GB recipes are:

- `pretrain_tiny`: `batch_size=1`, `grad_accum_steps=8`, `max_seq_len=512`
- `finetune_lora_7b`: `batch_size=1`, `grad_accum_steps=16`, `max_seq_len=384`, `bf16`, `LoRA rank=64`, `LoRA alpha=128`

To generate the full nightmare curriculum directly:

```bash
python3 data/sniper_dataset_gen.py --output data/sniper_train.jsonl --samples 10000
```

### Build a model

```python
from novamind import NovaMind, NovaMindConfig

# This instantiates the 7B configuration spec.
# It does not load pretrained weights by itself.
cfg = NovaMindConfig.from_size("7b")
model = NovaMind(cfg)
print(model.num_parameters() / 1e9)
```

### Fine-tune on 16 GB

```bash
python3 train.py \
    --run_mode finetune_lora_7b \
    --hf_model_name meta-llama/Llama-2-7b-hf \
    --tokenizer meta-llama/Llama-2-7b-hf \
    --data ./data/sniper_train.jsonl \
    --output ./checkpoints/finetune_lora_7b \
    --batch_size 1 \
    --grad_accum_steps 16 \
    --max_seq_len 384 \
    --lora_rank 64 \
    --lora_alpha 128 \
    --bf16 \
    --use_qat \
    --use_unified_offload \
    --use_met
```

### Tiny pre-train on 16 GB

```bash
python3 train.py \
    --run_mode pretrain_tiny \
    --data ./data/sniper_train.jsonl \
    --output ./checkpoints/pretrain_tiny \
    --batch_size 1 \
    --grad_accum_steps 8 \
    --max_seq_len 512 \
    --bf16 \
    --use_met \
    --use_rl_objective
```

### Run inference

```bash
# `--checkpoint` points to your own trained or exported weights.
python inference.py --checkpoint ./ckpt.pt --size 7b \
    --prompt "Explain the P vs NP problem"
```

### Interactive mode

```bash
# `--size 7b` selects the 7B config preset.
python inference.py --checkpoint ./ckpt.pt --size 7b --interactive
```

### Smoke test

```bash
python test_novamind.py
```

---

## The Reasoning Stress Test

If you want to see the "thinking" behavior rather than just text generation, run:

```bash
python train.py \
  --vault_stress_test \
  --vault_tasks 1 \
  --vault_candidate_paths 4 \
  --vault_min_paths 3 \
  --mcts_max_new_tokens 8 \
  --use_met \
  --met_entropy_threshold 3.5 \
  --met_caution_window 5
```

What you should see:

- MET deciding when to escalate into System 2
- MCTS exploring multiple candidate paths
- rollback traces catching poisoned logic
- high-fi replay only on winning branches
- final reasoning-efficiency summary

If you want a more guided walkthrough, see [Docs/QuickStart.md](/Users/felix/Desktop/Novamind/Docs/QuickStart.md).

For the `v1.2` curriculum-hardening release notes, see [Release_v1.2.md](/Users/felix/Desktop/Novamind/Release_v1.2.md).

---

## Useful Snippets

### CATTS in code

```python
from novamind.core.catts import AdaptiveNovaMindWrapper

wrapper = AdaptiveNovaMindWrapper(model, cfg)
result = wrapper.adaptive_forward(input_ids)

decision = result["decision"]
print(decision.mode)
print(decision.confidence)
print(decision.exit_layer)
print(result["catts_stats"])
```

### Continual learning with WSC

```python
from novamind.learning.wsc import WSCOptimizer

base_opt = torch.optim.AdamW(trainable_params, lr=2e-4)
optimizer = WSCOptimizer(model, base_opt, reset_freq=500)

loss.backward()
optimizer.step()
optimizer.zero_grad()
```

### Estimate VRAM before committing

```python
from novamind.training.lora import estimate_vram

est = estimate_vram(
    num_params=7_000_000_000,
    hidden_dim=4096,
    num_layers=32,
    use_activation_checkpointing=True,
)
print(est)
```

---

## Roadmap

### Implemented

- [x] Selective SSM reasoning core
- [x] xLSTM matrix memory
- [x] TITANS inference-time associative memory
- [x] Causal attention with differentiable logic
- [x] WSC continual learning
- [x] CATTS adaptive compute dispatcher
- [x] LoRA + activation offload for 16 GB training
- [x] Code MCTS sandbox, AST rollback, symbol sentinel, MET, SGA

### Next

- [ ] Triton-fused kernels for faster low-fi branching
- [ ] stronger dense baselines for cleaner apples-to-apples comparison
- [ ] tokenizer-trie symbol masking for more precise ISFS blocking
- [ ] richer code benchmarks beyond the current vault stress test
- [ ] multimodal symbolic sentinels

---

## Honest Status

Every major module in this repository is real PyTorch, not placeholder pseudocode.

What is already true:

- the code runs,
- the test suite passes,
- the memory-aware design is reflected in the implementation,
- the reasoning dashboard is real.

What still needs more empirical work:

- serious GPU-side throughput benchmarking,
- stronger training convergence evidence,
- large-scale comparative evaluation,
- long-context production stress beyond prototype benchmarks.

The claim is not “this beats every frontier model on one card.”  
The claim is simpler and more interesting:

**NovaMind makes structurally better decisions about where to spend compute.**

On constrained hardware, that is the difference between a demo and a system.

---

## License

Apache License 2.0.
