# Novamind-CS v1.0-RC

**The first 1.58-bit Reasoning Engine designed for 16GB Consumer GPUs.**

## Highlights

- **MET: Metacognitive Entropy Throttling**  
  Entropy-driven dual-process routing that escalates uncertain regions into System 2 reasoning.

- **AST-Aware Rollback**  
  Compiler-style traceback localization that preserves credit for valid prefixes and penalizes only the failing node.

- **SGA: Surgical Gradient Attribution**  
  Non-uniform gradient scaling that protects stable logic while aggressively correcting the defective span.

- **ISFS: Incremental Symbol Flow Sentinel**  
  Symbol-table-aware logit blocking that suppresses undefined identifiers before they hit the sandbox.

- **LOD-Compute**  
  Graphics-inspired precision routing: low-fi 1.58-bit exploration, high-fi FP16/BF16 replay on verified paths.

## Release Focus

- Consumer-GPU-first reasoning stack
- Cross-platform CUDA / MPS / CPU fallback routing
- Public stress-test dashboard for reasoning efficiency
- GitHub-ready documentation and setup workflow

## Recommended Demo

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
