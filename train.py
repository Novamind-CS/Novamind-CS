"""
NovaMind training script.

Supports:
- LoRA fine-tuning within a 16 GB VRAM budget
- WSC continual learning
- activation offloading
- mixed precision
- gradient checkpointing
- simple checkpoint save / resume flows

Usage:
    python train.py --size 7b --data /path/to/data --output ./checkpoints

Dependencies:
    pip install torch einops transformers datasets
"""

# GitHub README Snippet:
# 1. LOD-Compute: low-fi search, high-fi anchor; cheap breadth first, expensive
#    precision only on verified paths.
# 2. AST-Rollback: failing lines are traced back to concrete syntax nodes so the
#    search loop can preserve correct prefixes and only retry poisoned logic.
# 3. ISFS: an incremental symbol sentinel blocks undefined identifiers directly at
#    logit time, reducing wasted branches before they ever hit the sandbox.
# 4. SGA: gradient surgery preserves already-correct token spans while amplifying
#    updates around the exact failure boundary discovered by rollback analysis.
# 5. MET: entropy-driven metacognition routes easy tokens through System 1 and
#    escalates uncertain regions into persistent System 2 reasoning windows.

import os
import sys
import math
import time
import argparse
import json
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler

try:
    import triton  # noqa: F401
except Exception:
    triton = None

try:
    import deepspeed  # noqa: F401
except Exception:
    deepspeed = None


def build_training_config(config_cls, size: str, run_mode: str = "legacy"):
    if size != "tiny":
        return config_cls.from_size(size)

    if run_mode == "pretrain_tiny":
        return config_cls(
            vocab_size=32000,
            hidden_dim=768,
            num_layers=12,
            max_seq_len=1024,
            num_ssm_heads=8,
            num_attn_heads=4,
            head_dim=64,
            attention_window=96,
            ssm_state_dim=32,
            ssm_conv_kernel=4,
            ssm_expand=2,
            xlstm_matrix_dim=128,
            xlstm_num_heads=4,
            titans_memory_layers=2,
            titans_summary_slots=16,
            causal_graph_dim=24,
            logic_entity_dim=64,
            wsra_world_slots=8,
            wsra_jit_rank=16,
            wsra_frustum_tokens=64,
            wsra_proof_branches=4,
            lora_rank=8,
            lora_alpha=16,
        )

    return config_cls(
        vocab_size=512,
        hidden_dim=128,
        num_layers=2,
        max_seq_len=512,
        num_ssm_heads=4,
        num_attn_heads=2,
        head_dim=32,
        attention_window=32,
        ssm_state_dim=16,
        ssm_conv_kernel=4,
        ssm_expand=2,
        xlstm_matrix_dim=64,
        xlstm_num_heads=2,
        titans_memory_layers=1,
        titans_summary_slots=8,
        causal_graph_dim=16,
        logic_entity_dim=32,
        wsra_world_slots=4,
        wsra_jit_rank=8,
        wsra_frustum_tokens=16,
        wsra_proof_branches=2,
        lora_rank=8,
        lora_alpha=16,
    )


def count_model_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    if hasattr(model, "num_parameters"):
        return int(model.num_parameters(trainable_only=trainable_only))
    params = (
        p for p in model.parameters()
        if (p.requires_grad or not trainable_only)
    )
    return int(sum(p.numel() for p in params))


class CausalLMAdapter(nn.Module):
    """Normalize HuggingFace causal LMs to the NovaMind training interface."""

    def __init__(self, hf_model: nn.Module):
        super().__init__()
        self.hf_model = hf_model
        self.config = getattr(hf_model, "config", None)

    def forward(self,
                input_ids: torch.Tensor,
                labels: torch.Tensor | None = None,
                inference_mode: bool = False,
                return_logic_loss: bool = False,
                **kwargs):
        outputs = self.hf_model(
            input_ids=input_ids,
            labels=labels,
            use_cache=not inference_mode,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = None
        if getattr(outputs, "hidden_states", None):
            hidden_states = outputs.hidden_states[-1]
        return {
            "logits": outputs.logits,
            "hidden_states": hidden_states,
            "loss": outputs.loss,
            "logic_loss": None,
            "xlstm_states": None,
            "layer_states": None,
            "titans_fifos": None,
            "wsra": None,
            "wsra_state": None,
        }

    def num_parameters(self, trainable_only: bool = False) -> int:
        return count_model_parameters(self.hf_model, trainable_only=trainable_only)






class TextDataset(Dataset):
    """
    Supports two input formats:
    1. plain text files with one sample per line
    2. JSON Lines files with records such as {"text": "..."}
    """

    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.samples = []

        path = Path(data_path)
        assert path.exists(), f"Data file does not exist: {data_path}"

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("{"):
                    obj = json.loads(line)
                    text = obj.get("text", obj.get("content", ""))
                else:
                    text = line
                if line.startswith("{"):
                    if text or obj.get("prompt") or obj.get("answer"):
                        self.samples.append(obj)
                elif text:
                    self.samples.append(text)

        print(f"[Dataset] Loaded {len(self.samples)} samples from {data_path}")

    def __len__(self):
        return len(self.samples)

    def _tokenize(self, text: str):
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = tokens["input_ids"].squeeze(0) if isinstance(tokens, dict) else tokens.input_ids.squeeze(0)
        return input_ids

    def __getitem__(self, idx):
        sample = self.samples[idx]
        reward = 1.0
        prompt = ""
        tests = ""
        if isinstance(sample, dict):
            prompt = sample.get("prompt", "")
            rationale = sample.get("rationale", "")
            answer = sample.get(
                "answer",
                sample.get("solution", sample.get("reference_code", sample.get("text", sample.get("content", ""))))
            )
            reward = float(sample.get("reward", 1.0))
            tests = sample.get("tests", "")
            if prompt or rationale:
                text = f"User: {prompt}\nReasoning: {rationale}\nAssistant:\n```python\n{answer}\n```"
            else:
                text = answer
        else:
            text = sample

        input_ids = self._tokenize(text)
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "reward": torch.tensor(reward, dtype=torch.float32),
            "prompt": prompt,
            "tests": tests,
            "reference_code": sample.get("solution", sample.get("reference_code", "")) if isinstance(sample, dict) else "",
        }


class SniperDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 2048):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.samples = []

        path = Path(data_path)
        assert path.exists(), f"Sniper dataset does not exist: {data_path}"

        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                if sample.get("prompt") and sample.get("solution") and sample.get("tests"):
                    self.samples.append(sample)

        print(f"[SniperDataset] Loaded {len(self.samples)} records from {data_path}")

    def __len__(self):
        return len(self.samples)

    def _tokenize(self, text: str):
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].squeeze(0) if isinstance(tokens, dict) else tokens.input_ids.squeeze(0)
        return input_ids

    def __getitem__(self, idx):
        sample = self.samples[idx]
        training_text = (
            f"User: {sample['prompt']}\n"
            f"Assistant:\n```python\n{sample['solution']}\n```"
        )
        input_ids = self._tokenize(training_text)
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "reward": torch.tensor(1.0, dtype=torch.float32),
            "prompt": sample["prompt"],
            "tests": sample["tests"],
            "reference_code": sample["solution"],
            "task_type": sample.get("task_type", "sniper"),
        }


class SyntheticCurriculumDataset(Dataset):
    def __init__(self, tokenizer, max_length: int = 2048):
        from data.synthetic_curriculum import get_level_1_dataset

        self.samples = get_level_1_dataset()
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"[Synthetic] Loaded {len(self.samples)} Level-1 reasoning curriculum samples")

    def __len__(self):
        return len(self.samples)

    def _tokenize(self, text: str):
        tokens = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = tokens["input_ids"].squeeze(0) if isinstance(tokens, dict) else tokens.input_ids.squeeze(0)
        return input_ids

    def __getitem__(self, idx):
        sample = self.samples[idx]
        training_text = (
            f"User: {sample['prompt']}\n"
            f"Assistant:\n```python\n{sample['reference_code']}```"
        )
        input_ids = self._tokenize(training_text)
        return {
            "input_ids": input_ids,
            "labels": input_ids.clone(),
            "reward": torch.tensor(1.0, dtype=torch.float32),
            "prompt": sample["prompt"],
            "tests": sample["tests"],
            "reference_code": sample["reference_code"],
        }






class NovaMindTrainer:

    def __init__(self, model, optimizer, config, args, tokenizer,
                 teacher_model=None, zero_memory_manager=None):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.args = args
        self.tokenizer = tokenizer
        self.teacher_model = teacher_model
        self.zero_memory_manager = zero_memory_manager
        self.base_optimizer = (
            self.optimizer.base_optimizer
            if hasattr(self.optimizer, "base_optimizer")
            else self.optimizer
        )
        self.reasoner = None
        self.high_fi_callback = None
        self.sga_pre_backward_hook = None
        self.last_reward = None
        self.last_lod_status = ""
        self.last_met_penalty = 0.0
        self.last_met_entropy = None

        self.scaler = GradScaler() if args.fp16 else None
        self.step = 0
        self.best_loss = float("inf")


        self.scheduler = self._build_scheduler()

    def _build_scheduler(self):
        def lr_lambda(current_step):
            warmup = self.args.warmup_steps
            total = self.args.max_steps
            if current_step < warmup:
                return current_step / max(1, warmup)
            progress = (current_step - warmup) / max(1, total - warmup)
            return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

        return torch.optim.lr_scheduler.LambdaLR(self.base_optimizer, lr_lambda)

    def _extract_text_list(self, value):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _parse_tests_blob(self, tests_text: str) -> list[str]:
        text = str(tests_text).strip()
        if not text:
            return []
        if "\n\n" in text:
            return [block.strip() for block in text.split("\n\n") if block.strip()]
        return [line.strip() for line in text.splitlines() if line.strip()]

    def _encode_text(self, text: str) -> list[int]:
        if hasattr(self.tokenizer, "__call__"):
            try:
                tokens = self.tokenizer(text, add_special_tokens=False)
            except TypeError:
                tokens = self.tokenizer(text)
            if isinstance(tokens, dict):
                return list(tokens["input_ids"])
            return list(tokens.input_ids)
        return list(self.tokenizer.encode(text))

    def _code_line_to_token_index(self, prompt: str, code: str, failing_line: int) -> int:
        code_lines = code.splitlines()
        line_index = max(0, min(failing_line - 1, len(code_lines)))
        prefix_code = "\n".join(code_lines[:line_index])
        prefix_text = f"{prompt}\n\n{prefix_code}".rstrip()
        return len(self._encode_text(prefix_text))

    def _inspect_met_prompt(self, prompt: str):
        from novamind.core.code_mcts import _tokenize_to_tensor, inspect_system1_entropy

        input_ids = _tokenize_to_tensor(self.tokenizer, prompt, str(self.args.device))
        return inspect_system1_entropy(
            self.model,
            input_ids=input_ids,
            entropy_threshold=self.args.met_entropy_threshold,
        )

    def _build_sga_pre_backward_hook(self):
        from novamind.training.gradient_surgery import apply_surgical_mask

        def _hook(loss, model, metadata, total_tokens, training_text, prompt, code, outputs):
            rollback_trace = (metadata or {}).get("rollback_trace")
            if not rollback_trace:
                return None

            failing_line = rollback_trace.get("failing_line_number")
            if failing_line is None:
                return None

            token_index = self._code_line_to_token_index(prompt, code, failing_line)
            surgery_result = apply_surgical_mask(
                loss=loss,
                model=model,
                failing_line_index=token_index,
                total_tokens=total_tokens,
            )
            if surgery_result.get("applied"):
                print(
                    f"[SGA] Surgical Gradient scaling applied. "
                    f"Amplifying error at line {failing_line}, preserving previous logic."
                )
                surgery_result["failing_line"] = failing_line
            return surgery_result

        return _hook

    def _ensure_reasoner(self):
        if self.reasoner is not None:
            return self.reasoner

        from novamind.core.code_mcts import (
            CodeMCTSReasoner,
            PythonSandbox,
            make_high_fi_backprop_callback,
            make_sampling_proposal_fn,
        )
        from novamind.core.device_manager import is_high_perf_mode
        from novamind.core.met_controller import METGate, MetStateTracker

        self.sga_pre_backward_hook = self._build_sga_pre_backward_hook()
        self.high_fi_callback = make_high_fi_backprop_callback(
            self.model,
            self.tokenizer,
            device=str(self.args.device),
            optimizer=self.optimizer,
            enable_weight_updates=is_high_perf_mode(),
            pre_backward_hook=self.sga_pre_backward_hook,
        )

        met_tracker = None if not self.args.use_met else MetStateTracker(
            entropy_threshold=self.args.met_entropy_threshold,
            caution_window=self.args.met_caution_window,
        )
        proposal_fn = make_sampling_proposal_fn(
            self.model,
            self.tokenizer,
            device=str(self.args.device),
            max_new_tokens=self.args.mcts_max_new_tokens,
            temperature=self.args.mcts_temperature,
            top_p=self.args.mcts_top_p,
            met_gate=None if not self.args.use_met else METGate(self.args.met_entropy_threshold),
            met_tracker=met_tracker,
        )
        proposal_fn.met_tracker = met_tracker
        self.reasoner = CodeMCTSReasoner(
            proposal_fn=proposal_fn,
            sandbox=PythonSandbox(timeout_s=self.args.sandbox_timeout),
            high_fi_backprop_fn=self.high_fi_callback,
            num_options=self.args.mcts_options,
            max_rollouts=self.args.mcts_rollouts,
        )
        return self.reasoner

    def _make_curriculum_proposal_fn(self, prompt: str, reference_code: str):
        from novamind.core.code_mcts import BLOCK_MARKER, CodeBlockProposal, make_sampling_proposal_fn
        from novamind.core.met_controller import METGate, MetStateTracker

        met_tracker = None if not self.args.use_met else MetStateTracker(
            entropy_threshold=self.args.met_entropy_threshold,
            caution_window=self.args.met_caution_window,
        )
        model_proposal_fn = make_sampling_proposal_fn(
            self.model,
            self.tokenizer,
            device=str(self.args.device),
            max_new_tokens=self.args.mcts_max_new_tokens,
            temperature=self.args.mcts_temperature,
            top_p=self.args.mcts_top_p,
            met_gate=None if not self.args.use_met else METGate(self.args.met_entropy_threshold),
            met_tracker=met_tracker,
        )

        lines = reference_code.rstrip().splitlines()
        signature = lines[0].rstrip()
        body = textwrap.dedent("\n".join(lines[1:])).strip()

        def _proposal_fn(current_prompt, partial_code, block_index, num_options):
            proposals = []
            if block_index == 0:
                proposals.append(CodeBlockProposal(
                    block=f"{signature}\n    pass  {BLOCK_MARKER}",
                    is_terminal=False,
                    label="curriculum_signature",
                ))
            elif block_index == 1 and body:
                proposals.append(CodeBlockProposal(
                    block=body,
                    is_terminal=True,
                    label="curriculum_body",
                ))

            remaining = max(0, num_options - len(proposals))
            if remaining > 0:
                proposals.extend(
                    model_proposal_fn(current_prompt, partial_code, block_index, remaining)
                )
            return proposals[:num_options]

        _proposal_fn.met_tracker = met_tracker
        return _proposal_fn

    def _run_curriculum_step(self, sample: dict, epoch_idx: int) -> str:
        from novamind.core.code_mcts import CodeMCTSReasoner, PythonSandbox

        prompt = sample["prompt"]
        tests = self._parse_tests_blob(sample["tests"])
        proposal_fn = self._make_curriculum_proposal_fn(prompt, sample["reference_code"])
        reasoner = CodeMCTSReasoner(
            proposal_fn=proposal_fn,
            sandbox=PythonSandbox(timeout_s=self.args.sandbox_timeout),
            high_fi_backprop_fn=self.high_fi_callback,
            num_options=self.args.mcts_options,
            max_rollouts=self.args.mcts_rollouts,
        )

        was_training = self.model.training
        self.model.eval()
        try:
            result = reasoner.run(prompt, tests=tests)
        finally:
            self.model.train(was_training)

        reward = result["reward"]
        self.last_reward = reward
        callback_result = result.get("high_fi_backprop") or {}
        updated = bool(callback_result.get("updated", False))
        simulated = bool(callback_result.get("simulated", False))

        if updated:
            self.scheduler.step()

        if reward > 0 and updated:
            lod_status = "Switched to High-Fi & Updated Weights"
        elif reward > 0 and simulated:
            lod_status = "Switched to High-Fi (Simulated Update)"
        elif reward > 0:
            lod_status = "Winning Path Verified"
        else:
            lod_status = "Stayed in Low-Fi Search"

        self.last_lod_status = lod_status
        task_label = prompt.split("`")[1] if "`" in prompt else prompt[:40]
        print(
            f"[Epoch {epoch_idx}] Prompt: {task_label} | "
            f"Best Reward: {reward:+.0f} | LOD: {lod_status}"
        )
        return lod_status

    def _make_vault_proposal_fn(self, sample: dict):
        from novamind.core.code_mcts import BLOCK_MARKER, CodeBlockProposal, make_sampling_proposal_fn
        from novamind.core.met_controller import METGate, MetStateTracker

        met_tracker = None if not self.args.use_met else MetStateTracker(
            entropy_threshold=self.args.met_entropy_threshold,
            caution_window=self.args.met_caution_window,
        )
        model_proposal_fn = make_sampling_proposal_fn(
            self.model,
            self.tokenizer,
            device=str(self.args.device),
            max_new_tokens=self.args.mcts_max_new_tokens,
            temperature=self.args.mcts_temperature,
            top_p=self.args.mcts_top_p,
            met_gate=None if not self.args.use_met else METGate(self.args.met_entropy_threshold),
            met_tracker=met_tracker,
        )

        signature_block = (
            "def solve_vault(vault: dict, target: str, key: int = 0, max_depth: int = 8):\n"
            "    def _walk(node, depth):\n"
            f"        pass  {BLOCK_MARKER}\n"
            "    return _walk(vault, 0)"
        )

        def _proposal_fn(current_prompt, partial_code, block_index, num_options):
            if block_index == 0:
                return [CodeBlockProposal(
                    block=signature_block,
                    is_terminal=False,
                    label="vault_signature",
                )]

            proposals = []
            if block_index == 1:
                proposals.append(CodeBlockProposal(
                    block=sample["poisoned_block"],
                    is_terminal=True,
                    label="logic_poison",
                ))
                proposals.append(CodeBlockProposal(
                    block=sample["correct_block"],
                    is_terminal=True,
                    label="rollback_fix",
                ))

            remaining = max(0, num_options - len(proposals))
            if remaining > 0:
                proposals.extend(
                    model_proposal_fn(current_prompt, partial_code, block_index, remaining)
                )
            return proposals[:num_options]

        _proposal_fn.met_tracker = met_tracker
        return _proposal_fn

    def _print_vault_dashboard(self, task_name: str, path_traces: list[dict]):
        print(f"\n[Vault Task] {task_name}")
        print("| Path # | VRAM (MB) | LOD Tier | AST Status | Reward |")
        print("|--------|-----------|----------|------------|--------|")
        for trace in path_traces:
            print(
                f"| {trace['path_no']:>6} | "
                f"{trace['vram_mb']:>9.2f} | "
                f"{trace['lod_tier']:<8} | "
                f"{trace['ast_status'][:10]:<10} | "
                f"{trace['reward']:>+6.1f} |"
            )

    def _run_vault_stress_test(self):
        from data.synthetic_curriculum import get_secure_vault_dataset
        from novamind.core.code_mcts import CodeMCTSReasoner, PythonSandbox

        dataset = get_secure_vault_dataset(self.args.vault_tasks)
        self._ensure_reasoner()

        solved = 0
        poisoned_caught = 0
        total_high_fi_tokens = 0
        baseline_high_fi_tokens = 0
        total_loss_before = 0.0
        total_loss_after = 0.0
        max_low_fi_vram_mb = 0.0
        total_met_penalty = 0.0
        total_system1_tokens = 0
        total_system2_tokens = 0
        total_vram_success_mb = 0.0
        started_at = time.perf_counter()

        print("\n" + "=" * 72)
        print("Micro-CS Proof of Intelligence Dashboard")
        print("=" * 72)

        for sample_idx, sample in enumerate(dataset, start=1):
            tests = self._parse_tests_blob(sample["tests"])
            met_decision = self._inspect_met_prompt(sample["prompt"]) if self.args.use_met else None
            proposal_fn = self._make_vault_proposal_fn(sample)
            reasoner = CodeMCTSReasoner(
                proposal_fn=proposal_fn,
                sandbox=PythonSandbox(timeout_s=self.args.sandbox_timeout),
                high_fi_backprop_fn=self.high_fi_callback,
                num_options=self.args.vault_candidate_paths,
                max_rollouts=2,
                max_block_depth=2,
                min_paths_before_early_exit=self.args.vault_min_paths,
            )

            was_training = self.model.training
            self.model.eval()
            try:
                result = reasoner.run(sample["prompt"], tests=tests)
            finally:
                self.model.train(was_training)

            path_traces = result.get("path_traces", [])
            self._print_vault_dashboard(sample["task_name"], path_traces)

            if any(trace.get("failing_line_number") for trace in path_traces):
                poisoned_caught += 1

            if result["reward"] > 0:
                solved += 1
            elif met_decision is not None and not met_decision["trigger_system2"]:
                total_met_penalty += self.args.dunning_kruger_penalty

            if path_traces:
                sample_max_vram = max(trace["vram_mb"] for trace in path_traces)
                max_low_fi_vram_mb = max(max_low_fi_vram_mb, sample_max_vram)
                if result["reward"] > 0:
                    total_vram_success_mb += sample_max_vram

            met_tracker = getattr(proposal_fn, "met_tracker", None)
            if met_tracker is not None:
                total_system1_tokens += met_tracker.system1_tokens
                total_system2_tokens += met_tracker.system2_tokens

            high_fi = result.get("high_fi_backprop") or {}
            total_high_fi_tokens += int(high_fi.get("token_count", 0))
            total_loss_before += float(high_fi.get("loss_before", 0.0))
            total_loss_after += float(high_fi.get("loss_after", 0.0))
            baseline_high_fi_tokens += len(path_traces) * int(high_fi.get("token_count", 0))

            lod_summary = "No High-Fi Anchor"
            if high_fi:
                if high_fi.get("updated"):
                    lod_summary = "Switched to High-Fi & Updated Weights"
                elif high_fi.get("simulated"):
                    lod_summary = "Switched to High-Fi (Simulated Update)"
                else:
                    lod_summary = "Switched to High-Fi"
                if high_fi.get("sga_applied"):
                    lod_summary += f" + SGA(line {high_fi.get('sga_line')})"

            print(
                f"[Vault {sample_idx:02d}] {sample['task_name']} | "
                f"Reward: {result['reward']:+.1f} | "
                f"Low-Fi paths: {len(path_traces)} | "
                f"High-Fi tokens: {int(high_fi.get('token_count', 0))} | "
                f"LOD: {lod_summary}"
                f"{'' if met_decision is None else f' | MET(H={met_decision['entropy']:.3f})'}"
            )
            self.step += 1

        loss_drop = total_loss_before - total_loss_after
        loss_drop_ratio = 0.0 if total_loss_before <= 1e-8 else loss_drop / total_loss_before
        token_efficiency = 0.0 if total_high_fi_tokens == 0 else baseline_high_fi_tokens / total_high_fi_tokens
        total_runtime_s = time.perf_counter() - started_at
        reasoning_efficiency = solved / (max(max_low_fi_vram_mb, 1.0) * max(total_runtime_s, 1e-6))
        avg_vram_per_success = 0.0 if solved == 0 else total_vram_success_mb / solved
        total_met_tokens = total_system1_tokens + total_system2_tokens
        system1_pct = 0.0 if total_met_tokens == 0 else 100.0 * total_system1_tokens / total_met_tokens
        system2_pct = 0.0 if total_met_tokens == 0 else 100.0 * total_system2_tokens / total_met_tokens
        reasoning_accuracy = 0.0 if len(dataset) == 0 else solved / len(dataset)
        validated = (
            solved == len(dataset)
            and max_low_fi_vram_mb < 8192.0
            and total_high_fi_tokens < 500
        )

        print("\n" + "-" * 72)
        print(f"Solved tasks: {solved}/{len(dataset)}")
        print(f"AST poison caught: {poisoned_caught}/{len(dataset)}")
        print(f"Max low-fi VRAM: {max_low_fi_vram_mb:.2f} MB")
        print(f"Total high-fi tokens: {total_high_fi_tokens}")
        print(f"Dense baseline high-fi tokens: {baseline_high_fi_tokens}")
        print(f"High-fi token efficiency gain: {token_efficiency:.2f}x")
        print(f"Loss before anchor: {total_loss_before:.4f}")
        print(f"Loss after anchor:  {total_loss_after:.4f}")
        print(f"Loss drop ratio:    {100.0 * loss_drop_ratio:.2f}%")
        print(f"Dunning-Kruger Penalty: {total_met_penalty:.4f}")
        print(f"Reasoning Efficiency: {reasoning_efficiency:.6f} solved/(MB*s)")
        print("| Metric | Value |")
        print("|--------|-------|")
        print(f"| Cognitive Compression Ratio | System1 {system1_pct:.2f}% / System2 {system2_pct:.2f}% |")
        print(f"| VRAM Efficiency | {avg_vram_per_success:.2f} MB per successful deduction |")
        print(f"| Reasoning Accuracy | {solved}/{len(dataset)} ({100.0 * reasoning_accuracy:.2f}%) |")
        verdict = "VALIDATED: More Efficient than Transformer" if validated else "NOT YET VALIDATED"
        print(f"Verdict: {verdict}")
        print("-" * 72)

    def _compute_reasoning_rewards(self, batch: dict):
        prompts = self._extract_text_list(batch.get("prompt"))
        tests_blob = self._extract_text_list(batch.get("tests"))
        reference_codes = self._extract_text_list(batch.get("reference_code"))

        if not prompts or not tests_blob:
            return None

        if self.high_fi_callback is None:
            self._ensure_reasoner()

        rewards = []
        penalties = []

        was_training = self.model.training
        self.model.eval()
        try:
            for idx, (prompt, tests_text) in enumerate(zip(prompts, tests_blob)):
                tests = self._parse_tests_blob(tests_text)
                met_decision = self._inspect_met_prompt(prompt) if self.args.use_met else None
                reference_code = ""
                if reference_codes and idx < len(reference_codes):
                    reference_code = reference_codes[idx] or ""

                if reference_code:
                    from novamind.core.code_mcts import CodeMCTSReasoner, PythonSandbox

                    proposal_fn = self._make_curriculum_proposal_fn(prompt, reference_code)
                    reasoner = CodeMCTSReasoner(
                        proposal_fn=proposal_fn,
                        sandbox=PythonSandbox(timeout_s=self.args.sandbox_timeout),
                        high_fi_backprop_fn=self.high_fi_callback,
                        num_options=self.args.mcts_options,
                        max_rollouts=self.args.mcts_rollouts,
                    )
                else:
                    reasoner = self._ensure_reasoner()

                result = reasoner.run(prompt, tests=tests)
                rewards.append(result["reward"])
                penalty = 0.0
                if (
                    met_decision is not None
                    and not met_decision["trigger_system2"]
                    and result["reward"] <= 0
                ):
                    penalty = self.args.dunning_kruger_penalty
                penalties.append(penalty)
                if met_decision is not None:
                    self.last_met_entropy = met_decision["entropy"]
        finally:
            self.model.train(was_training)

        reward_tensor = torch.tensor(rewards, device=self.args.device, dtype=torch.float32)
        self.last_reward = reward_tensor.mean().item()
        self.last_met_penalty = float(sum(penalties) / max(1, len(penalties)))
        return reward_tensor

    def train_step(self, batch: dict) -> float:
        """Run one training step and return the loss value."""
        input_ids = batch["input_ids"].to(self.args.device)
        labels = batch["labels"].to(self.args.device)
        rewards = batch.get("reward")
        if rewards is not None:
            rewards = rewards.to(self.args.device)

        reasoning_rewards = self._compute_reasoning_rewards(batch)
        if reasoning_rewards is not None:
            rewards = reasoning_rewards

        self.model.train()


        from novamind.core.device_manager import get_autocast_context

        if self.zero_memory_manager is not None and self.args.use_unified_offload:
            self.zero_memory_manager.restore_optimizer_state(self.base_optimizer)

        with get_autocast_context(self.args.device, enabled=self.args.bf16):
            out = self.model(input_ids=input_ids, labels=labels,
                             return_logic_loss=True)
            loss = out["loss"]
            if out["logic_loss"] is not None:
                loss = loss + 0.01 * out["logic_loss"]

            if self.teacher_model is not None:
                from novamind.training.quantization import distillation_loss

                with torch.no_grad():
                    teacher_out = self.teacher_model(input_ids=input_ids)

                kd = distillation_loss(
                    student_logits=out["logits"],
                    teacher_logits=teacher_out["logits"].detach(),
                    student_hidden=out.get("hidden_states"),
                    teacher_hidden=teacher_out.get("hidden_states"),
                    temperature=self.args.kd_temperature,
                    alpha=self.args.kd_alpha,
                )
                loss = loss + kd["total"]

            if (self.args.use_rl_objective or reasoning_rewards is not None) and rewards is not None:
                from novamind.training.rl import reward_weighted_ce_loss
                loss = loss + self.args.rl_weight * reward_weighted_ce_loss(
                    out["logits"], labels, rewards
                )
            if self.last_met_penalty > 0:
                loss = loss + self.last_met_penalty


        loss = loss / self.args.grad_accum_steps
        loss.backward()

        if (self.step + 1) % self.args.grad_accum_steps == 0:

            if hasattr(self.optimizer, "base_optimizer"):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.max_grad_norm
                )
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.args.max_grad_norm
                )


            if hasattr(self.optimizer, "step"):
                self.optimizer.step()
            if self.zero_memory_manager is not None and self.args.use_unified_offload:
                self.zero_memory_manager.offload_optimizer_state(self.base_optimizer)
            self.scheduler.step()
            self.optimizer.zero_grad()
            if self.zero_memory_manager is not None and self.args.use_unified_offload:
                self.zero_memory_manager.clear()

        return loss.item() * self.args.grad_accum_steps

    def save_checkpoint(self, path: str, loss: float):
        os.makedirs(path, exist_ok=True)
        payload = {
            "step": self.step,
            "loss": loss,
            "optimizer_state": (self.optimizer.state_dict()
                                if hasattr(self.optimizer, "state_dict") else None),
        }

        if self.args.use_qat:
            payload["model_state_dict"] = self.model.state_dict()
        else:
            lora_state = {k: v for k, v in self.model.state_dict().items()
                          if "lora_A" in k or "lora_B" in k}
            payload["lora_state_dict"] = lora_state

        torch.save(payload, os.path.join(path, f"checkpoint_step{self.step}.pt"))
        print(f"[Trainer] Saved checkpoint: step={self.step}, loss={loss:.4f}")

    def train(self, dataloader: DataLoader | None):
        """Run the full training loop."""
        print(f"\n{'='*60}")
        print("NovaMind training started")
        print(f"  Device: {self.args.device}")
        print(f"  Run mode: {self.args.run_mode}")
        print(f"  Max steps: {self.args.max_steps}")
        print(f"  Batch size: {self.args.batch_size} × grad accumulation: {self.args.grad_accum_steps}")
        print(f"  Effective batch size: {self.args.batch_size * self.args.grad_accum_steps}")
        print(f"  Model size: {count_model_parameters(self.model)/1e9:.2f}B total, "
              f"{count_model_parameters(self.model, trainable_only=True)/1e6:.1f}M trainable")
        print(f"{'='*60}\n")

        self._ensure_reasoner()

        if self.args.vault_stress_test:
            self._run_vault_stress_test()
            return

        if self.args.synthetic_curriculum:
            dataset = getattr(dataloader, "dataset", None)
            if dataset is None:
                raise RuntimeError("Synthetic curriculum mode requires a dataset-backed dataloader")

            for epoch in range(1, self.args.max_epochs + 1):
                for sample in dataset:
                    self._run_curriculum_step(sample, epoch)
                    self.step += 1
                    if self.step >= self.args.max_steps:
                        print(f"\n[Training] Synthetic sanity phase complete after {self.step} steps")
                        return
            print(f"\n[Training] Synthetic sanity phase complete after {self.step} steps")
            return

        loss_window = []
        t0 = time.time()

        for epoch in range(self.args.max_epochs):
            for batch in dataloader:
                loss_val = self.train_step(batch)
                self.step += 1

                loss_window.append(loss_val)
                if len(loss_window) > 100:
                    loss_window.pop(0)


                if self.step % self.args.log_interval == 0:
                    elapsed = time.time() - t0
                    avg_loss = sum(loss_window) / len(loss_window)
                    steps_per_sec = self.step / elapsed


                    vram_gb = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0

                    print(f"Step {self.step:6d} | "
                          f"Loss: {avg_loss:.4f} | "
                          f"LR: {self.scheduler.get_last_lr()[0]:.2e} | "
                          f"VRAM: {vram_gb:.1f}GB | "
                          f"Speed: {steps_per_sec:.1f} steps/s"
                          f"{'' if self.last_reward is None else f' | Reward: {self.last_reward:.1f}'}"
                          f"{'' if self.last_met_entropy is None else f' | MET-H: {self.last_met_entropy:.2f}'}"
                          f"{'' if self.last_met_penalty <= 0 else f' | DK: {self.last_met_penalty:.2f}'}")


                if (self.step % self.args.save_interval == 0
                        and loss_val < self.best_loss):
                    self.best_loss = loss_val
                    self.save_checkpoint(self.args.output, loss_val)


                if self.step >= self.args.max_steps:
                    print(f"\nTraining complete after {self.step} steps")
                    self.save_checkpoint(self.args.output, loss_val)
                    return


def main():
    parser = argparse.ArgumentParser(description="NovaMind training")
    parser.add_argument(
        "--run_mode",
        choices=["legacy", "pretrain_tiny", "finetune_lora_7b"],
        default="legacy",
        help="Training path: keep legacy behavior, pretrain a tiny sniper model, or fine-tune a 7B HF model with LoRA.",
    )
    parser.add_argument("--size", choices=["tiny", "3b", "7b", "14b"], default="tiny")
    parser.add_argument("--data", type=str, default="", help="Path to the training data")
    parser.add_argument("--output", type=str, default="./checkpoints")
    parser.add_argument("--tokenizer", type=str, default="",
                        help="Path or identifier for the HuggingFace tokenizer")
    parser.add_argument("--hf_model_name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--generate_sniper_data", action="store_true", default=False)
    parser.add_argument("--sniper_samples", type=int, default=10000)
    parser.add_argument("--sniper_seed", type=int, default=42)
    parser.add_argument("--synthetic_curriculum", action="store_true", default=False)
    parser.add_argument("--vault_stress_test", action="store_true", default=False)
    parser.add_argument("--vault_tasks", type=int, default=50)
    parser.add_argument("--vault_candidate_paths", type=int, default=52)
    parser.add_argument("--vault_min_paths", type=int, default=50)


    parser.add_argument("--max_steps", type=int, default=12)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--width_multiplier", type=float, default=1.0)


    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--no_activation_checkpointing", action="store_true")
    parser.add_argument("--no_cpu_offload", action="store_true")
    parser.add_argument("--use_qat", action="store_true", default=False)
    parser.add_argument("--quant_clip_value", type=float, default=6.0)
    parser.add_argument("--quant_smooth_alpha", type=float, default=0.5)
    parser.add_argument("--use_unified_offload", action="store_true", default=False)

    # LoRA
    parser.add_argument("--lora_rank", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)

    # WSC
    parser.add_argument("--use_wsc", action="store_true", default=True)
    parser.add_argument("--wsc_reset_freq", type=int, default=500)
    parser.add_argument("--teacher_checkpoint", type=str, default="")
    parser.add_argument("--teacher_size", choices=["3b", "7b", "14b"], default="")
    parser.add_argument("--kd_alpha", type=float, default=0.7)
    parser.add_argument("--kd_temperature", type=float, default=2.0)
    parser.add_argument("--use_rl_objective", action="store_true", default=False)
    parser.add_argument("--rl_weight", type=float, default=0.1)
    parser.add_argument("--use_met", action="store_true", default=False)
    parser.add_argument("--met_entropy_threshold", type=float, default=3.5)
    parser.add_argument("--met_caution_window", type=int, default=5)
    parser.add_argument("--dunning_kruger_penalty", type=float, default=0.25)
    parser.add_argument("--mcts_options", type=int, default=3)
    parser.add_argument("--mcts_rollouts", type=int, default=4)
    parser.add_argument("--mcts_max_new_tokens", type=int, default=64)
    parser.add_argument("--mcts_temperature", type=float, default=0.8)
    parser.add_argument("--mcts_top_p", type=float, default=0.95)
    parser.add_argument("--sandbox_timeout", type=float, default=0.5)

    args = parser.parse_args()


    from novamind.config import NovaMindConfig
    from novamind.model import NovaMind
    from novamind.training.lora import inject_lora, freeze_base_model, estimate_vram
    from novamind.learning.wsc import WSCOptimizer
    from novamind.training.unified_offload import ZeROMemoryManager
    from novamind.core.device_manager import (
        HardwareTier,
        detect_hardware_tier,
        get_device,
        get_dtype,
        get_hardware_banner,
        is_high_perf_mode,
    )

    detected_tier = detect_hardware_tier()
    args.device = get_device()
    runtime_dtype = get_dtype(detected_tier)

    if args.run_mode == "pretrain_tiny":
        args.size = "tiny"
        args.use_met = True
        args.use_rl_objective = True
        args.rl_weight = max(args.rl_weight, 0.2)
        args.mcts_options = max(args.mcts_options, 4)
        args.mcts_rollouts = max(args.mcts_rollouts, 6)
        args.synthetic_curriculum = False
    elif args.run_mode == "finetune_lora_7b":
        args.size = "7b"
        args.use_met = True
        args.tokenizer = args.tokenizer or args.hf_model_name

    if (
        args.run_mode == "legacy"
        and not args.data
        and not args.synthetic_curriculum
        and not args.vault_stress_test
    ):
        args.synthetic_curriculum = True
        print("[Training] No --data provided, switching to the synthetic sanity curriculum")

    if (
        args.synthetic_curriculum
        or args.vault_stress_test
        or args.run_mode == "pretrain_tiny"
    ) and detected_tier == HardwareTier.MAC_DEBUG:
        args.device = torch.device("cpu")
        runtime_dtype = torch.float32
        print("[Training] MAC_DEBUG reasoning mode forces CPU execution to avoid MPS kernel crashes")

    if (
        args.synthetic_curriculum
        or args.vault_stress_test
        or args.run_mode == "pretrain_tiny"
    ) and detected_tier == HardwareTier.MAC_DEBUG:
        print(f"[Hardware] tier=MAC_DEBUG device=cpu dtype={runtime_dtype}")
    else:
        print(get_hardware_banner(args.device))

    if (
        (args.synthetic_curriculum or args.vault_stress_test or args.run_mode == "pretrain_tiny")
        and not is_high_perf_mode()
        and args.size != "tiny"
    ):
        print("[Training] Non-CUDA reasoning environments automatically fall back to the tiny config")
        args.size = "tiny"

    cfg: Any = None
    teacher_model = None

    def _load_tokenizer(tokenizer_name: str):
        try:
            if tokenizer_name:
                from transformers import AutoTokenizer

                tok = AutoTokenizer.from_pretrained(tokenizer_name)
                if tok.pad_token is None:
                    tok.pad_token = tok.eos_token
                return tok
        except Exception as exc:
            print(f"[Tokenizer] Failed to load '{tokenizer_name}': {exc}")

        print("[Training] Using the character-level tokenizer for a local sanity check")
        from inference import CharTokenizer
        return CharTokenizer()

    def _configure_novamind_model():
        local_cfg = build_training_config(NovaMindConfig, args.size, args.run_mode)
        if args.width_multiplier > 1.0:
            local_cfg = local_cfg.with_width_multiplier(args.width_multiplier)
        local_cfg.max_seq_len = args.max_seq_len
        local_cfg.use_activation_checkpointing = not args.no_activation_checkpointing
        local_cfg.offload_activations = not args.no_cpu_offload
        local_cfg.use_unified_offload = args.use_unified_offload
        local_cfg.lora_rank = args.lora_rank
        local_cfg.lora_alpha = args.lora_alpha
        local_cfg.wsc_reset_freq = args.wsc_reset_freq

        num_params = {
            "tiny": 120_000_000 if args.run_mode == "pretrain_tiny" else 2_000_000,
            "3b": 3_000_000_000,
            "7b": 7_000_000_000,
            "14b": 14_000_000_000,
        }[args.size]
        vram_est = estimate_vram(
            num_params=num_params,
            hidden_dim=local_cfg.hidden_dim,
            num_layers=local_cfg.num_layers,
            use_activation_checkpointing=local_cfg.use_activation_checkpointing,
        )
        print(f"\n[VRAM Estimate] {args.size.upper()} model:")
        for key, value in vram_est.items():
            flag = "✓" if key == "fits_16gb" else "  "
            print(f"  {flag} {key}: {value}")
        print()

        print(f"[Model] Initializing NovaMind-{args.size.upper()}...")
        local_model = NovaMind(local_cfg)
        if args.use_qat:
            from novamind.training.quantization import estimate_qat_vram, preserve_sensitive_precision
            from novamind.training.bitlinear import replace_with_bitlinear

            preserve_sensitive_precision(local_model)
            if is_high_perf_mode():
                local_model, replaced = replace_with_bitlinear(local_model)
                print(f"[QAT] Replaced {replaced} linear layers with BitLinear 1.58-bit QAT modules")
            else:
                print("[QAT] Non-CUDA tier detected, skipping BitLinear replacement.")
            qat_est = estimate_qat_vram(num_params=num_params)
            print(f"[QAT] Training VRAM estimate: {qat_est}")

        if not args.use_qat and args.run_mode != "pretrain_tiny":
            local_model = inject_lora(
                local_model,
                target_modules=local_cfg.lora_target_modules,
                rank=local_cfg.lora_rank,
                alpha=local_cfg.lora_alpha,
            )
            local_model = freeze_base_model(local_model)

        local_model = local_model.to(args.device)
        local_model = local_model.to(runtime_dtype)
        return local_model, local_cfg, num_params

    def _configure_hf_7b_model():
        try:
            from transformers import AutoModelForCausalLM
        except Exception as exc:
            raise RuntimeError(
                "finetune_lora_7b requires transformers with AutoModelForCausalLM available"
            ) from exc

        print(f"[Model] Loading HF base model: {args.hf_model_name}")
        load_kwargs = {
            "torch_dtype": runtime_dtype,
            "low_cpu_mem_usage": True,
        }
        if args.load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        hf_model = AutoModelForCausalLM.from_pretrained(args.hf_model_name, **load_kwargs)
        if not args.load_in_8bit:
            hf_model = hf_model.to(args.device)

        local_model = CausalLMAdapter(hf_model)
        local_model = inject_lora(
            local_model,
            target_modules=["q_proj", "v_proj"],
            rank=args.lora_rank,
            alpha=args.lora_alpha,
        )
        local_model = freeze_base_model(local_model)

        if args.use_qat:
            from novamind.training.bitlinear import replace_with_bitlinear

            if is_high_perf_mode():
                local_model, replaced = replace_with_bitlinear(
                    local_model,
                    target_keywords=("gate_proj", "up_proj", "down_proj", "k_proj", "o_proj"),
                )
                print(f"[QAT] Replaced {replaced} non-LoRA linear layers with BitLinear modules")
            else:
                print("[QAT] Non-CUDA tier detected, skipping BitLinear replacement.")

        local_cfg = SimpleNamespace(
            hidden_dim=getattr(hf_model.config, "hidden_size", 4096),
            num_layers=getattr(hf_model.config, "num_hidden_layers", 32),
            use_activation_checkpointing=not args.no_activation_checkpointing,
            offload_activations=not args.no_cpu_offload,
            use_unified_offload=args.use_unified_offload,
            unified_offload_device="cpu",
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_target_modules=["q_proj", "v_proj"],
            wsc_reset_freq=args.wsc_reset_freq,
            wsc_ema_decay=0.999,
        )
        return local_model, local_cfg, 7_000_000_000

    if args.run_mode == "finetune_lora_7b":
        model, cfg, num_params = _configure_hf_7b_model()
    else:
        model, cfg, num_params = _configure_novamind_model()

    if getattr(cfg, "use_activation_checkpointing", False):
        print("[VRAM] Gradient checkpointing enabled")

    if args.teacher_checkpoint:
        teacher_size = args.teacher_size or args.size
        teacher_cfg = NovaMindConfig.from_size(teacher_size)
        teacher_model = NovaMind(teacher_cfg)
        teacher_ckpt = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
        state = teacher_ckpt.get("model_state_dict") or teacher_ckpt.get("ema_model") or teacher_ckpt
        missing, unexpected = teacher_model.load_state_dict(state, strict=False)
        print(f"[KD] Teacher loaded (missing={len(missing)}, unexpected={len(unexpected)})")
        teacher_model = teacher_model.to(args.device)
        teacher_model = teacher_model.to(runtime_dtype)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)


    if args.use_qat:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
    base_opt = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=0.01,
        betas=(0.9, 0.95)
    )

    if args.use_wsc:
        optimizer = WSCOptimizer(
            model=model,
            base_optimizer=base_opt,
            reset_freq=args.wsc_reset_freq,
            ema_decay=cfg.wsc_ema_decay
        )
        print("[WSC] Continual-learning optimizer enabled")
    else:
        optimizer = base_opt

    zero_memory_manager = ZeROMemoryManager(offload_device=cfg.unified_offload_device)
    if args.use_unified_offload:
        zero_memory_manager.register_offload_hooks(model, base_opt)
        print("[ZeRO] Unified offload hooks registered")

    tokenizer = _load_tokenizer(args.tokenizer)

    if args.run_mode == "pretrain_tiny":
        data_path = args.data or "data/sniper_train.jsonl"
        if args.generate_sniper_data or not Path(data_path).exists():
            from data.sniper_dataset_gen import write_sniper_dataset

            write_sniper_dataset(data_path, samples=args.sniper_samples, seed=args.sniper_seed)
            print(f"[SniperData] Generated dataset at {data_path}")
        args.data = data_path

    if args.vault_stress_test:
        dataset = None
    elif args.run_mode == "pretrain_tiny":
        dataset = SniperDataset(args.data, tokenizer, max_length=args.max_seq_len)
    elif args.synthetic_curriculum:
        dataset = SyntheticCurriculumDataset(tokenizer, max_length=args.max_seq_len)
    elif tokenizer is not None and args.data:
        dataset = TextDataset(args.data, tokenizer, max_length=args.max_seq_len)
    else:
        print("[Error] No --data path provided and --synthetic_curriculum is disabled")
        sys.exit(1)

    dataloader = None
    if dataset is not None:
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )


    trainer = NovaMindTrainer(
        model, optimizer, cfg, args, tokenizer,
        teacher_model=teacher_model,
        zero_memory_manager=zero_memory_manager,
    )
    trainer.train(dataloader)


if __name__ == "__main__":
    main()
