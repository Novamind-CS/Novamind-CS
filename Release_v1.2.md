# Novamind-CS v1.2: The Hardcore Curriculum

`Novamind-CS v1.2` is the data release that hardens the entire project.

The dual-engine training stack from `v1.1` remains intact, but the curriculum beneath it has been upgraded from introductory algorithmic logic to a true nightmare-grade Python reasoning workload.

This release is built around one premise:

> A "Python Sniper" cannot be trained on polite toy tasks. It must learn on adversarial, state-heavy, execution-verified problems that punish fragile logic.

---

## Highlight: Nightmare Difficulty Data Upgrade

`data/sniper_dataset_gen.py` has been completely rewritten.

The new generator now procedurally emits **10,000+ extreme-difficulty pure-Python tasks** across advanced domains:

- Dijkstra with state constraints
- lexicographically stable Topological Sorting
- Tarjan's Strongly Connected Components
- Bitmask Dynamic Programming / state compression
- DP on Trees
- Trie prefix-count workloads
- Segment-tree style range-minimum logic
- N-Queens backtracking
- hard maze pathfinding with adversarial obstacles

Every sample still follows the same execution-first structure:

- `prompt`
- `solution`
- `tests`

The difference is that the hidden tests are now much harsher:

- disconnected states
- degenerate inputs
- cyclic graphs
- empty structures
- boundary-case integer behavior
- backtracking correctness checks

This is a curriculum designed to stress:

- AST-aware rollback
- MCTS branch repair
- MET escalation logic
- SGA gradient locality
- and the symbolic stability of the full Novamind-CS stack

---

## Why v1.2 Matters

`v1.1` proved the workflow could launch.

`v1.2` raises the bar on what the workflow is allowed to learn from.

The practical effect is simple:

- fewer shallow memorization wins,
- more stateful reasoning pressure,
- stronger failure signals,
- and better separation between fragile code generation and truly robust symbolic execution.

For a small 100M-200M "Sniper" model, that matters more than raw sample count.

---

## Intended Outcome

This curriculum upgrade is specifically aimed at training a compact model that can punch above its size by combining:

- difficult execution-verified data,
- explicit search through MCTS,
- symbolic rollback on concrete failure sites,
- and selective high-fidelity correction.

The point is not to imitate a large dense model.

The point is to train a smaller model that learns to survive harder reasoning environments.

---

## Recommended Upgrade Flow

```bash
python3 data/sniper_dataset_gen.py --output data/sniper_train.jsonl --samples 10000
./launch_novamind.sh
```

---

## Closing

`Novamind-CS v1.2` is the curriculum-hardening release.

If `v1.1` introduced **The Sniper and The Heavyweight**, then `v1.2` gives The Sniper a battlefield worthy of the name.
