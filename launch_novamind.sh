#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
SNIPER_SAMPLES="${SNIPER_SAMPLES:-10000}"
SNIPER_DATA_PATH="${SNIPER_DATA_PATH:-data/sniper_train.jsonl}"
HF_MODEL_NAME="${HF_MODEL_NAME:-meta-llama/Llama-2-7b-hf}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./checkpoints}"

echo "============================================================"
echo "NovaMind-CS One-Click Launcher"
echo "Target GPU: RTX 5070 Ti / 16GB VRAM"
echo "Python: ${PYTHON_BIN}"
echo "Output root: ${OUTPUT_ROOT}"
echo "============================================================"

echo "Step 1: Generating Sniper Dataset..."
"${PYTHON_BIN}" data/sniper_dataset_gen.py \
  --output "${SNIPER_DATA_PATH}" \
  --samples "${SNIPER_SAMPLES}" \
  --seed 42

echo "Step 2: Commencing Path 1 - Tiny Pre-training (Sanity Check)..."
"${PYTHON_BIN}" train.py \
  --run_mode pretrain_tiny \
  --data "${SNIPER_DATA_PATH}" \
  --output "${OUTPUT_ROOT}/pretrain_tiny" \
  --tokenizer "${HF_MODEL_NAME}" \
  --max_steps 1000 \
  --max_epochs 1 \
  --batch_size 1 \
  --grad_accum_steps 8 \
  --max_seq_len 512 \
  --lr 2e-4 \
  --warmup_steps 50 \
  --log_interval 10 \
  --save_interval 250 \
  --bf16 \
  --use_met \
  --met_entropy_threshold 3.4 \
  --met_caution_window 5 \
  --use_rl_objective \
  --rl_weight 0.2 \
  --mcts_options 4 \
  --mcts_rollouts 6 \
  --mcts_max_new_tokens 48 \
  --mcts_temperature 0.8 \
  --mcts_top_p 0.95 \
  --sandbox_timeout 0.5

read -r -p "Tiny pre-training complete. Do you want to proceed to Path 2 (7B LoRA Fine-tuning)? (y/n) " PROCEED
if [[ ! "${PROCEED}" =~ ^[Yy]$ ]]; then
  echo "Stopping after Path 1."
  exit 0
fi

echo "Step 3: Commencing Path 2 - 7B LoRA Fine-tuning..."
"${PYTHON_BIN}" train.py \
  --run_mode finetune_lora_7b \
  --hf_model_name "${HF_MODEL_NAME}" \
  --tokenizer "${HF_MODEL_NAME}" \
  --data "${SNIPER_DATA_PATH}" \
  --output "${OUTPUT_ROOT}/finetune_lora_7b" \
  --max_steps 600 \
  --max_epochs 1 \
  --batch_size 1 \
  --grad_accum_steps 16 \
  --max_seq_len 384 \
  --lr 1e-4 \
  --warmup_steps 40 \
  --log_interval 10 \
  --save_interval 100 \
  --bf16 \
  --use_qat \
  --use_met \
  --met_entropy_threshold 3.3 \
  --met_caution_window 6 \
  --use_rl_objective \
  --rl_weight 0.15 \
  --use_unified_offload \
  --lora_rank 64 \
  --lora_alpha 128 \
  --mcts_options 3 \
  --mcts_rollouts 4 \
  --mcts_max_new_tokens 40 \
  --mcts_temperature 0.75 \
  --mcts_top_p 0.92 \
  --sandbox_timeout 0.5

echo "NovaMind-CS dual-engine run complete."
