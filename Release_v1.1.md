# Novamind-CS v1.1: The Sniper and The Heavyweight

`Novamind-CS v1.1` is the release where the project stops being only a reasoning architecture prototype and becomes a practical dual-engine training workflow for consumer hardware.

The core value of this update is straightforward:

> **Bring reasoning-aware training to consumer GPUs without pretending datacenter assumptions still apply.**

This release formalizes a workflow for a 16 GB RTX 5070 Ti-class machine:

- generate verifiable logic data,
- pre-train a compact reasoning model from scratch,
- fine-tune a larger 7B backbone with LoRA,
- and keep the full MCTS + AST + MET + SGA stack in the loop.

---

## The Three Pillars of v1.1

### 1. Procedural Sniper Dataset Generation

`v1.1` introduces `data/sniper_dataset_gen.py`, a pure-Python dataset engine built for the **Sniper** path.

It generates **10,000+ verifiable algorithmic tasks** using only the Python standard library. Every sample is emitted as JSONL with:

- `prompt`
- `solution`
- `tests`

The generated tasks cover core reasoning domains:

- Binary Search
- Dynamic Programming
- BFS / graph traversal
- Sorting with explicit comparator logic
- Matrix manipulation

Every item includes AST-compatible `assert` statements, so the same records can be used for:

- supervised next-token learning,
- MCTS sandbox evaluation,
- AST-aware rollback,
- reward shaping,
- and reasoning-aware fine-tuning.

This is executable logic data, not generic text filler.

---

### 2. Dual-Engine Workflow

`train.py` now exposes a clear split between two strategic paths:

#### `--run_mode pretrain_tiny`

The **Sniper** path.

This mode trains a compact model from scratch on pure Python algorithmic logic. It is designed for:

- fast iteration,
- syntax and control-flow acquisition,
- system validation under low-resource constraints,
- and aggressive reasoning feedback through the full symbolic stack.

#### `--run_mode finetune_lora_7b`

The **Heavyweight** path.

This mode loads a 7B causal LM backbone, injects LoRA adapters, and fine-tunes only the adapter weights while preserving the reasoning controls introduced by Novamind-CS.

This path is intended to combine:

- base-model fluency,
- LoRA memory efficiency,
- low-fi speculative reasoning,
- high-fi replay,
- and surgical gradient correction.

The result is a unified CLI that can move cleanly between from-scratch small-model training and large-model adaptation.

---

### 3. One-Click 16GB VRAM Automation

`v1.1` ships with:

- `launch_novamind.sh`
- `launch_novamind.bat`

These scripts automate the full paper-launch workflow:

1. generate the Sniper dataset,
2. run tiny-model sanity pretraining,
3. optionally continue into 7B LoRA fine-tuning.

The recipes are tuned specifically for a **16 GB VRAM ceiling** and rely on:

- micro-batching,
- gradient accumulation,
- activation checkpointing,
- bounded MCTS token expansion,
- LoRA-only trainable adapters,
- optional unified offloading,
- and low-fi / high-fi routing through LOD-Compute.

The target is not raw maximal throughput. The target is **stable execution without OOM** while preserving the reasoning stack.

---

## Technical Spec

### SGA: Surgical Gradient Attribution

When a generated trace fails, `Novamind-CS` does not update every token equally.

`SGA` scales gradients around the fault boundary so that:

- earlier, already-correct logic is preserved,
- later failing logic is corrected aggressively.

That gives the project a much more selective learning signal than flat backprop over the whole sequence.

### MET: Metacognitive Entropy Throttling

`MET` decides when a token can stay in **System 1** and when the model must escalate into **System 2** reasoning.

Low-entropy states take the cheap path. High-entropy states trigger deeper reasoning through MCTS and remain there briefly through inertial gating.

This converts uncertainty into a real compute-allocation policy.

### MCTS Sandbox Integration

The code-generation loop now relies on:

- MCTS branch exploration,
- `ISFS` symbol blocking before bad identifiers are sampled,
- Python sandbox execution,
- AST-aware rollback on precise failure locations,
- and high-fidelity replay only on winning paths.

This is the defining systems idea of the project:

> speculative reasoning should be cheap, symbolic verification should be explicit, and expensive gradients should only be spent on validated trajectories.

---

## Recommended 16GB Launch Path

```bash
./launch_novamind.sh
```

Or manually:

```bash
python3 data/sniper_dataset_gen.py --output data/sniper_train.jsonl --samples 10000

python3 train.py \
  --run_mode pretrain_tiny \
  --data data/sniper_train.jsonl
```

Then:

```bash
python3 train.py \
  --run_mode finetune_lora_7b \
  --hf_model_name meta-llama/Llama-2-7b-hf \
  --tokenizer meta-llama/Llama-2-7b-hf \
  --data data/sniper_train.jsonl
```

---

## Closing

`Novamind-CS v1.1` is not just a feature increment. It is the release that formalizes the project’s training identity:

- **The Sniper** for tight, logic-first pretraining
- **The Heavyweight** for practical 7B adaptation

If `v1.0` proved the architecture could reason, `v1.1` proves the workflow can now be launched, repeated, and pushed on real consumer hardware.
