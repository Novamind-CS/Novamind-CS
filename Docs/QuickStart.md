# Quick Start: Your First Reasoning Stress Test

## Problem

You have cloned `Novamind-CS`, but the repository is not a conventional “run one script and watch tokens stream” project. The interesting behavior emerges when the engine is forced to decide:

- when to stay in fast-path inference,
- when to escalate into deliberate reasoning,
- how to localize a logic fault,
- and how to update only the part of the model that actually failed.

That is exactly what the `vault_stress_test` is for.

## Solution

This walkthrough gets you from a clean checkout to your first visible reasoning cycle in three stages:

1. prepare the environment,
2. run the stress test,
3. read the MET and SGA diagnostics like an engineer.

---

## Step 1: Prepare the Environment

From the repository root:

```bash
bash setup_env.sh
```

What this does:

- detects your active Python or Conda environment,
- installs `requirements.txt`,
- runs the hardware probe from `device_manager`,
- executes `test_novamind.py`,
- prints `System Ready` if the local stack is healthy.

Expected hardware outcomes:

- `CUDA_EXTREME`: Windows/Linux NVIDIA path
- `MAC_DEBUG`: Apple Silicon debug path
- `CPU_FALLBACK`: universal safe path

If you are on a Mac, this is normal: the project intentionally routes away from unstable CUDA-only kernels.

---

## Step 2: Run the First Stress Test

The fastest meaningful benchmark is:

```bash
python3 train.py \
  --vault_stress_test \
  --vault_tasks 1 \
  --vault_candidate_paths 4 \
  --vault_min_paths 3 \
  --mcts_max_new_tokens 8 \
  --use_met \
  --met_entropy_threshold 3.5 \
  --met_caution_window 5
```

This command forces the full reasoning stack to activate:

- `MET` decides whether uncertainty is high enough to enter System 2
- `MCTS` expands multiple candidate code paths
- `ISFS` blocks impossible symbol usage before execution
- `AST-Rollback` finds the exact failure region when a branch crashes
- `LOD-Compute` replays winning paths in high fidelity
- `SGA` targets gradient updates at the failing boundary

If everything is wired correctly, you should see:

- a proof-of-intelligence dashboard,
- path-by-path reward lines,
- MET activation logs,
- rollback traces for poisoned code,
- and a final reasoning summary table.

---

## Step 3: Make the Model “Think More” or “Think Less”

The most important knob is:

```bash
--met_entropy_threshold
```

This threshold controls how easily the engine escalates from System 1 into System 2.

### Lower Threshold: More Deliberate Reasoning

Example:

```bash
python3 train.py \
  --vault_stress_test \
  --vault_tasks 1 \
  --vault_candidate_paths 4 \
  --vault_min_paths 3 \
  --mcts_max_new_tokens 8 \
  --use_met \
  --met_entropy_threshold 2.5 \
  --met_caution_window 5
```

What happens:

- MET triggers earlier
- more tokens enter the MCTS path
- the dashboard will show lower cognitive compression and heavier System 2 usage
- the model looks more “paranoid,” but also more visibly deliberate

### Higher Threshold: More Fast-Path Decoding

Example:

```bash
python3 train.py \
  --vault_stress_test \
  --vault_tasks 1 \
  --vault_candidate_paths 4 \
  --vault_min_paths 3 \
  --mcts_max_new_tokens 8 \
  --use_met \
  --met_entropy_threshold 5.5 \
  --met_caution_window 5
```

What happens:

- MET waits longer before escalating
- more tokens stay in System 1
- the engine is faster, but may defer deep reasoning until later

### How to Observe the “Thinking” Process

When MET triggers, you will see logs like:

```text
[MET] High Entropy Detected (H=6.191). Engaging MCTS Reasoning...
```

Interpretation:

- `H=...` is the measured Shannon entropy of the token distribution
- a higher value means the model is less certain
- that uncertainty causes the engine to switch into deliberate search

If you lower `--met_entropy_threshold`, these logs appear more often.

---

## Reading the SGA Gradient Logs

The most important SGA log looks like this:

```text
[SGA] Surgical Gradient scaling applied. Amplifying error at line X, preserving previous logic.
```

What it means:

- the sandbox found a failure,
- rollback mapped it to a concrete line,
- the high-fidelity replay is no longer treating every token equally.

Conceptually:

- tokens **before** the failure get a reduced gradient multiplier
- tokens **at or after** the failure get an amplified gradient multiplier

This is the opposite of blunt backprop. It is targeted correction.

### How to Read It in Practice

If you see an SGA line after a successful reasoning pass:

1. The engine found a branch worth preserving.
2. It identified a localized failure region.
3. It chose not to rewrite the whole trace.
4. It pushed learning pressure exactly where the logic broke.

That is the core design philosophy of `Novamind-CS`.

---

## Understanding the Final Dashboard

At the end of the stress test you will see a summary containing metrics such as:

- `Solved tasks`
- `AST poison caught`
- `Total high-fi tokens`
- `High-fi token efficiency gain`
- `Reasoning Efficiency`
- `Cognitive Compression Ratio`
- `VRAM Efficiency`
- `Reasoning Accuracy`

Three lines matter most on your first run:

### 1. Cognitive Compression Ratio

This tells you how much work stayed in System 1 versus System 2.

- higher System 1 percentage: more efficient, less deliberative
- higher System 2 percentage: more explicit reasoning, more expensive

### 2. High-Fi Token Efficiency Gain

This is the concrete payoff of LOD-Compute.

If the system can search broadly in low fidelity and only replay winning branches in FP16/BF16, then this ratio should improve relative to a dense high-fidelity baseline.

### 3. Reasoning Efficiency

This compresses solved-task yield against compute and memory cost. It is the closest thing in the current dashboard to the project’s core thesis: **information gain per watt**.

---

## Recommended First Experiments

### Experiment A: See More Thinking

Lower the threshold:

```bash
--met_entropy_threshold 2.5
```

Expected result:

- more MET logs
- more deliberate branch expansion
- lower cognitive compression ratio

### Experiment B: Force a More Efficient Personality

Raise the threshold:

```bash
--met_entropy_threshold 5.5
```

Expected result:

- fewer MET triggers
- more System 1 behavior
- faster but shallower reasoning

### Experiment C: Increase Search Breadth

Raise candidate paths:

```bash
--vault_candidate_paths 8
```

Expected result:

- more low-fi exploration
- richer rollback behavior
- stronger contrast between cheap search and expensive replay

---

## If Something Looks Wrong

### No MET logs appear

Check:

- `--use_met` is present
- `--met_entropy_threshold` is not set too high

### No SGA logs appear

This usually means either:

- no rollback-worthy fault was encountered,
- or you are on a non-CUDA path where high-fi update is simulated rather than applied.

### The run says `MAC_DEBUG`

That is expected on Apple Silicon. The debug path is designed for correctness and iteration, not peak fused-kernel performance.

---

## One-Line Recap

If you remember only one command, use this:

```bash
python3 train.py --vault_stress_test --vault_tasks 1 --vault_candidate_paths 4 --vault_min_paths 3 --mcts_max_new_tokens 8 --use_met --met_entropy_threshold 3.5 --met_caution_window 5
```

It is the shortest path from “I cloned the repo” to “I watched a neuro-symbolic reasoning engine decide when to think.”
