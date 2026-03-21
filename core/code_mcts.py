"""
Industrial code-oriented MCTS reasoning engine.

Key upgrades:
1. Node state is represented as partial_code + block_index.
2. Expansion is incremental at the code-block level.
3. The sandbox uses a four-stage early-exit reward funnel.
4. macOS / Python 3.14 compatibility is preserved via the fork-first path.
"""

from __future__ import annotations

import ast
import copy
import math
import multiprocessing as mp
import queue
import time
import traceback
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch

from novamind.core.ast_rollback import (
    calculate_rollback_reward,
    execute_and_trace,
    find_failing_ast_node,
)
from novamind.core.lod_router import lod_compute
from novamind.core.met_controller import METGate, MetStateTracker, calculate_entropy
from novamind.core.symbol_sentinel import SymbolSentinel


SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "RecursionError": RecursionError,
    "set": set,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}

BLOCK_MARKER = "# TODO: next_block"


@dataclass
class SandboxResult:
    status: str
    reward: float
    stage: str
    stdout: str = ""
    error: str = ""
    passed_tests: int = 0
    total_tests: int = 0
    execution_time_s: float = 0.0
    peak_memory_kb: float = 0.0
    early_exit: bool = False
    failing_line_number: Optional[int] = None
    ast_node_type: str = ""
    ast_source_segment: str = ""
    rollback_reward: Optional[float] = None
    ast_status: str = "clean"


@dataclass
class CodeBlockProposal:
    block: str
    is_terminal: Optional[bool] = None
    label: str = ""


@dataclass
class CodeSearchNode:
    prompt: str
    partial_code: str
    block_index: int = 0
    is_terminal: bool = False
    parent: Optional["CodeSearchNode"] = None
    children: List["CodeSearchNode"] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    sandbox_result: Optional[SandboxResult] = None
    expanded: bool = False

    @property
    def mean_value(self) -> float:
        return 0.0 if self.visits == 0 else self.value_sum / self.visits

    def ucb_score(self, total_parent_visits: int, c_puct: float = 1.4) -> float:
        if self.visits == 0:
            return float("inf")
        explore = c_puct * math.sqrt(math.log(max(total_parent_visits, 1) + 1) / self.visits)
        return self.mean_value + explore


def _contains_marker(code: str) -> bool:
    return BLOCK_MARKER in code


def compose_partial_code(partial_code: str, next_block: str, is_terminal: bool = False) -> str:
    next_block = next_block.rstrip()
    if not partial_code.strip():
        return next_block.strip() + ("\n" if next_block.strip() else "")

    lines = partial_code.splitlines()
    marker_line_idx = None
    for idx, line in enumerate(lines):
        if BLOCK_MARKER in line:
            marker_line_idx = idx
            break

    if marker_line_idx is None:
        joined = partial_code.rstrip() + "\n" + next_block
        return joined.rstrip() + "\n"

    indent = lines[marker_line_idx][:len(lines[marker_line_idx]) - len(lines[marker_line_idx].lstrip())]
    replacement_lines = next_block.splitlines()
    replacement_lines = [
        line if not line.strip() else indent + line
        for line in replacement_lines
    ]
    if not is_terminal and not any(BLOCK_MARKER in line for line in replacement_lines):
        replacement_lines.append(lines[marker_line_idx])

    lines[marker_line_idx:marker_line_idx + 1] = replacement_lines
    return "\n".join(lines).rstrip() + "\n"


def _normalize_proposals(raw: Sequence[Any]) -> List[CodeBlockProposal]:
    proposals: List[CodeBlockProposal] = []
    for item in raw:
        if isinstance(item, CodeBlockProposal):
            proposals.append(item)
        elif isinstance(item, dict):
            proposals.append(CodeBlockProposal(
                block=item.get("block", ""),
                is_terminal=item.get("is_terminal"),
                label=item.get("label", ""),
            ))
        else:
            proposals.append(CodeBlockProposal(block=str(item)))
    return proposals


def _build_ast_payload(code: str, failing_line_number: Optional[int]) -> Dict[str, Any]:
    if failing_line_number is None:
        return {
            "failing_line_number": None,
            "ast_node_type": "",
            "ast_source_segment": "",
            "rollback_reward": None,
            "ast_status": "unknown",
        }

    match = find_failing_ast_node(code, failing_line_number)
    rollback_reward = calculate_rollback_reward(
        total_lines=len(code.strip().splitlines()),
        failing_line=failing_line_number,
    )
    node_label = match.node_type or "Unknown"
    return {
        "failing_line_number": failing_line_number,
        "ast_node_type": node_label,
        "ast_source_segment": match.source_segment or "",
        "rollback_reward": rollback_reward,
        "ast_status": f"{node_label}@L{failing_line_number}",
    }


def _extract_script_lineno(exc: BaseException, filename: str) -> Optional[int]:
    tb_exception = traceback.TracebackException.from_exception(exc)
    for frame in reversed(tb_exception.stack):
        if frame.filename == filename:
            return frame.lineno
    return None


def _sandbox_worker(code: str,
                    tests: List[str],
                    optimal_time_s: float,
                    optimal_memory_kb: float,
                    result_queue):
    namespace = {"__builtins__": SAFE_BUILTINS.copy()}
    stdout_chunks: List[str] = []

    def _capture_print(*args, **kwargs):
        sep = kwargs.get("sep", " ")
        end = kwargs.get("end", "\n")
        stdout_chunks.append(sep.join(str(a) for a in args) + end)

    namespace["__builtins__"]["print"] = _capture_print

    filename = "<novamind_mcts>"
    try:
        compiled = compile(code, filename, "exec")
        tracemalloc.start()
        started = time.perf_counter()
        exec(compiled, namespace, namespace)
    except (ImportError, NameError) as exc:
        failing_line = _extract_script_lineno(exc, filename)
        ast_payload = _build_ast_payload(code, failing_line)
        result_queue.put({
            "status": "compile_import_error",
            "reward": -5.0,
            "stage": "compile_import",
            "error": f"{exc.__class__.__name__}: {exc}",
            "stdout": "".join(stdout_chunks),
            "passed_tests": 0,
            "total_tests": len(tests),
            "execution_time_s": 0.0,
            "peak_memory_kb": 0.0,
            "early_exit": True,
            **ast_payload,
        })
        return
    except Exception as exc:
        failing_line = _extract_script_lineno(exc, filename)
        if failing_line is None:
            execution = execute_and_trace(code)
            failing_line = execution.failing_line_number
        ast_payload = _build_ast_payload(code, failing_line)
        result_queue.put({
            "status": "compile_runtime_error",
            "reward": ast_payload["rollback_reward"] if ast_payload["rollback_reward"] is not None else -5.0,
            "stage": "compile_import",
            "error": f"{exc.__class__.__name__}: {exc}",
            "stdout": "".join(stdout_chunks),
            "passed_tests": 0,
            "total_tests": len(tests),
            "execution_time_s": 0.0,
            "peak_memory_kb": 0.0,
            "early_exit": True,
            **ast_payload,
        })
        return

    passed = 0
    for test in tests:
        try:
            compiled_test = compile(test, "<novamind_test>", "exec")
            exec(compiled_test, namespace, namespace)
            passed += 1
        except (ImportError, NameError) as exc:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            failing_line = _extract_script_lineno(exc, filename)
            ast_payload = _build_ast_payload(code, failing_line)
            result_queue.put({
                "status": "compile_import_error",
                "reward": -5.0,
                "stage": "compile_import",
                "error": f"{exc.__class__.__name__}: {exc}",
                "stdout": "".join(stdout_chunks),
                "passed_tests": passed,
                "total_tests": len(tests),
                "execution_time_s": time.perf_counter() - started,
                "peak_memory_kb": peak / 1024.0,
                "early_exit": True,
                **ast_payload,
            })
            return
        except AssertionError as exc:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            result_queue.put({
                "status": "unit_test_failed",
                "reward": -2.0,
                "stage": "unit_test",
                "error": f"AssertionError: {exc}",
                "stdout": "".join(stdout_chunks),
                "passed_tests": passed,
                "total_tests": len(tests),
                "execution_time_s": time.perf_counter() - started,
                "peak_memory_kb": peak / 1024.0,
                "early_exit": True,
                "failing_line_number": None,
                "ast_node_type": "",
                "ast_source_segment": "",
                "rollback_reward": None,
                "ast_status": "assertion_failed",
            })
            return
        except Exception as exc:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            failing_line = _extract_script_lineno(exc, filename)
            ast_payload = _build_ast_payload(code, failing_line)
            result_queue.put({
                "status": "unit_test_error",
                "reward": ast_payload["rollback_reward"] if ast_payload["rollback_reward"] is not None else -2.0,
                "stage": "unit_test",
                "error": f"{exc.__class__.__name__}: {exc}",
                "stdout": "".join(stdout_chunks),
                "passed_tests": passed,
                "total_tests": len(tests),
                "execution_time_s": time.perf_counter() - started,
                "peak_memory_kb": peak / 1024.0,
                "early_exit": True,
                **ast_payload,
            })
            return

    elapsed = time.perf_counter() - started
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    efficient = elapsed <= optimal_time_s and (peak / 1024.0) <= optimal_memory_kb
    result_queue.put({
        "status": "passed_optimal" if efficient else "passed_suboptimal",
        "reward": 100.0 if efficient else 20.0,
        "stage": "performance",
        "error": "",
        "stdout": "".join(stdout_chunks),
        "passed_tests": passed,
        "total_tests": len(tests),
        "execution_time_s": elapsed,
        "peak_memory_kb": peak / 1024.0,
        "early_exit": False,
        "failing_line_number": None,
        "ast_node_type": "",
        "ast_source_segment": "",
        "rollback_reward": None,
        "ast_status": "verified",
    })


class PythonSandbox:
    def __init__(self, timeout_s: float = 2.0,
                 optimal_time_s: float = 0.02,
                 optimal_memory_kb: float = 256.0):
        self.timeout_s = timeout_s
        self.optimal_time_s = optimal_time_s
        self.optimal_memory_kb = optimal_memory_kb

    def run(self, code: str, tests: Optional[List[str]] = None) -> SandboxResult:
        tests = tests or []

        try:
            ast.parse(code)
        except SyntaxError as exc:
            return SandboxResult(
                status="lint_error",
                reward=-10.0,
                stage="lint",
                error=f"{exc.__class__.__name__}: {exc}",
                total_tests=len(tests),
                early_exit=True,
            )

        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context("spawn")
        result_queue = ctx.Queue()
        process = ctx.Process(
            target=_sandbox_worker,
            args=(code, tests, self.optimal_time_s, self.optimal_memory_kb, result_queue),
        )
        process.start()
        process.join(self.timeout_s)

        if process.is_alive():
            process.kill()
            process.join()
            return SandboxResult(
                status="timeout",
                reward=-50.0,
                stage="timeout",
                error=f"Execution exceeded {self.timeout_s:.2f}s",
                total_tests=len(tests),
                early_exit=True,
            )

        try:
            payload = result_queue.get_nowait()
        except queue.Empty:
            return SandboxResult(
                status="crash",
                reward=-75.0,
                stage="sandbox",
                error="Sandbox exited without returning a result",
                total_tests=len(tests),
                early_exit=True,
            )

        return SandboxResult(**payload)


class CodeMCTSReasoner:
    """
    """

    def __init__(self,
                 proposal_fn: Callable[..., Sequence[Any]],
                 sandbox: Optional[PythonSandbox] = None,
                 high_fi_backprop_fn: Optional[Callable[[str, str], Any]] = None,
                 num_options: int = 3,
                 max_rollouts: int = 24,
                 max_block_depth: int = 6,
                 c_puct: float = 1.4,
                 min_paths_before_early_exit: int = 0):
        self.proposal_fn = proposal_fn
        self.sandbox = sandbox or PythonSandbox()
        self.high_fi_backprop_fn = high_fi_backprop_fn
        self.num_options = num_options
        self.max_rollouts = max_rollouts
        self.max_block_depth = max_block_depth
        self.c_puct = c_puct
        self.min_paths_before_early_exit = min_paths_before_early_exit

    def _select(self, root: CodeSearchNode) -> CodeSearchNode:
        node = root
        while node.children and node.expanded and not node.is_terminal:
            node = max(node.children, key=lambda child: child.ucb_score(node.visits, self.c_puct))
        return node

    def _call_proposal_fn(self, node: CodeSearchNode) -> List[CodeBlockProposal]:
        try:
            raw = self.proposal_fn(node.prompt, node.partial_code, node.block_index, self.num_options)
        except TypeError:
            raw = self.proposal_fn(node.prompt, node.partial_code, self.num_options)
        return _normalize_proposals(raw)

    def _expand(self, node: CodeSearchNode) -> List[CodeSearchNode]:
        proposals = self._call_proposal_fn(node)
        children = []
        for proposal in proposals:
            tentative_terminal = proposal.is_terminal if proposal.is_terminal is not None else False
            new_partial = compose_partial_code(node.partial_code, proposal.block, is_terminal=tentative_terminal)
            child_terminal = tentative_terminal or (not _contains_marker(new_partial)) or (
                node.block_index + 1 >= self.max_block_depth
            )
            child = CodeSearchNode(
                prompt=node.prompt,
                partial_code=new_partial,
                block_index=node.block_index + 1,
                is_terminal=child_terminal,
                parent=node,
            )
            node.children.append(child)
            children.append(child)
        node.expanded = True
        return children

    def _backprop(self, node: CodeSearchNode, reward: float):
        current = node
        while current is not None:
            current.visits += 1
            current.value_sum += reward
            current = current.parent

    def _run_high_fi_backprop(self, prompt: str, code: str, metadata: Optional[Dict[str, Any]] = None):
        if self.high_fi_backprop_fn is None:
            return None
        with lod_compute(tier="high_fi"):
            try:
                return self.high_fi_backprop_fn(prompt, code, metadata=metadata)
            except TypeError:
                return self.high_fi_backprop_fn(prompt, code)

    def _current_vram_mb(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024.0 * 1024.0)
        return 0.0

    def run(self, prompt: str, tests: Optional[List[str]] = None) -> Dict[str, Any]:
        root = CodeSearchNode(prompt=prompt, partial_code="", block_index=0, is_terminal=False)
        best_node = root
        best_reward = float("-inf")
        path_traces: List[Dict[str, Any]] = []
        path_counter = 0

        for _ in range(self.max_rollouts):
            target = self._select(root)
            leaves = self._expand(target) if not target.expanded and not target.is_terminal else [target]

            for leaf in leaves:
                result = self.sandbox.run(leaf.partial_code, tests)
                leaf.sandbox_result = result
                self._backprop(leaf, result.reward)
                path_counter += 1
                path_traces.append({
                    "path_no": path_counter,
                    "vram_mb": round(self._current_vram_mb(), 2),
                    "lod_tier": "low_fi",
                    "ast_status": result.ast_status,
                    "reward": result.reward,
                    "stage": result.stage,
                    "failing_line_number": result.failing_line_number,
                    "ast_node_type": result.ast_node_type,
                    "ast_source_segment": result.ast_source_segment,
                })

                if result.reward > best_reward:
                    best_reward = result.reward
                    best_node = leaf

                rollback_trace = next(
                    (trace for trace in reversed(path_traces) if trace.get("failing_line_number") is not None),
                    None,
                )

                if (
                    result.stage == "performance"
                    and result.reward >= 100.0
                    and path_counter >= self.min_paths_before_early_exit
                ):
                    high_fi_result = self._run_high_fi_backprop(
                        prompt,
                        leaf.partial_code,
                        metadata={
                            "rollback_trace": rollback_trace,
                            "path_traces": path_traces,
                        },
                    )
                    return {
                        "code": leaf.partial_code,
                        "reward": result.reward,
                        "result": result,
                        "high_fi_backprop": high_fi_result,
                        "rollouts": root.visits,
                        "block_depth": leaf.block_index,
                        "path_traces": path_traces,
                    }

        high_fi_result = None
        if best_node.sandbox_result is not None and best_node.sandbox_result.reward > 0:
            rollback_trace = next(
                (trace for trace in reversed(path_traces) if trace.get("failing_line_number") is not None),
                None,
            )
            high_fi_result = self._run_high_fi_backprop(
                prompt,
                best_node.partial_code,
                metadata={
                    "rollback_trace": rollback_trace,
                    "path_traces": path_traces,
                },
            )

        return {
            "code": best_node.partial_code,
            "reward": best_reward,
            "result": best_node.sandbox_result,
            "high_fi_backprop": high_fi_result,
            "rollouts": root.visits,
            "block_depth": best_node.block_index,
            "path_traces": path_traces,
        }


def _extract_block_fragment(text: str, fallback_terminal: bool = False) -> CodeBlockProposal:
    cleaned = text.strip("\n")
    if not cleaned:
        return CodeBlockProposal(block=f"pass  {BLOCK_MARKER}", is_terminal=False)

    lines = cleaned.splitlines()
    fragment: List[str] = []
    seen_code = False
    for line in lines:
        if line.strip():
            seen_code = True
        if seen_code:
            fragment.append(line.rstrip())
        if seen_code and line.strip() == "":
            break

    block = "\n".join(fragment).strip()
    is_terminal = fallback_terminal or (BLOCK_MARKER not in block and "pass" not in block.lower())
    return CodeBlockProposal(block=block, is_terminal=is_terminal)


def _fallback_generated_block(block_index: int, proposal_index: int) -> CodeBlockProposal:
    if block_index == 0:
        return CodeBlockProposal(
            block=f"def solve(x):\n    pass  {BLOCK_MARKER}",
            is_terminal=False,
            label=f"fallback_signature_{proposal_index}",
        )

    fallback_blocks = [
        "return None",
        "if node is None:\n    return None\nreturn None",
        "for _ in range(1):\n    return None\nreturn None",
        "items = []\nfor item in items:\n    return item\nreturn None",
    ]
    block = fallback_blocks[proposal_index % len(fallback_blocks)]
    return CodeBlockProposal(
        block=block,
        is_terminal=True,
        label=f"fallback_block_{proposal_index}",
    )


def _tokenize_to_tensor(tokenizer, text: str, device: str) -> torch.Tensor:
    if hasattr(tokenizer, "__call__"):
        tokens = tokenizer(text, return_tensors="pt")
        if isinstance(tokens, dict):
            input_ids = tokens["input_ids"]
        else:
            input_ids = tokens.input_ids
        return input_ids.to(device)
    return torch.tensor([tokenizer.encode(text)], device=device)


@torch.no_grad()
def inspect_system1_entropy(model, input_ids: torch.Tensor,
                            entropy_threshold: float = 3.5) -> Dict[str, Any]:
    outputs = model(input_ids=input_ids, inference_mode=True)
    logits = outputs["logits"][:, -1, :]
    gate = METGate(entropy_threshold=entropy_threshold)
    decision = gate.inspect(logits)
    decision["logits"] = logits
    return decision


def _decode_ids(tokenizer, token_ids: list[int]) -> str:
    if hasattr(tokenizer, "decode"):
        try:
            return tokenizer.decode(token_ids, skip_special_tokens=True)
        except TypeError:
            return tokenizer.decode(token_ids)
    return tokenizer.decode(token_ids)


def _symbol_token_ids(tokenizer, symbol: str) -> list[int]:
    try:
        if hasattr(tokenizer, "encode"):
            try:
                token_ids = tokenizer.encode(symbol, add_special_tokens=False)
            except TypeError:
                token_ids = tokenizer.encode(symbol)
            if isinstance(token_ids, torch.Tensor):
                token_ids = token_ids.tolist()
            return [int(tok) for tok in token_ids if int(tok) >= 0]
    except Exception:
        return []
    return []


def _sample_next_token(logits: torch.Tensor,
                       temperature: float,
                       top_p: float,
                       top_k: int) -> torch.Tensor:
    next_token_logits = logits
    if temperature != 1.0:
        next_token_logits = next_token_logits / max(temperature, 1e-5)

    if top_k > 0 and top_k < next_token_logits.shape[-1]:
        top_k_vals = torch.topk(next_token_logits, top_k, dim=-1)[0]
        next_token_logits = next_token_logits.masked_fill(
            next_token_logits < top_k_vals[..., -1:], float("-inf")
        )

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(next_token_logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(sorted_probs, dim=-1)
        remove_mask = cumprobs - sorted_probs > top_p
        sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
        next_token_logits = torch.full_like(next_token_logits, float("-inf"))
        next_token_logits.scatter_(1, sorted_idx, sorted_logits)

    probs = torch.softmax(next_token_logits, dim=-1)
    if not torch.isfinite(probs).all() or (probs.sum(dim=-1) <= 0).any():
        return torch.argmax(logits, dim=-1, keepdim=True)
    return torch.multinomial(probs, num_samples=1)


def _apply_symbol_bias(tokenizer,
                       logits: torch.Tensor,
                       code_prefix: str,
                       blocked_cache: set[tuple[str, int]]) -> torch.Tensor:
    sentinel = SymbolSentinel()
    undefined_symbols = sentinel.get_undefined_references(code_prefix)
    if not undefined_symbols:
        return logits

    biased = logits.clone()
    for symbol in sorted(undefined_symbols):
        token_ids = _symbol_token_ids(tokenizer, symbol)
        if not token_ids:
            continue
        token_id = token_ids[0]
        if token_id >= biased.shape[-1]:
            continue
        biased[..., token_id] = float("-inf")
        cache_key = (symbol, token_id)
        if cache_key not in blocked_cache:
            print(f"[Sentinel] Blocked undefined symbol token '{symbol}' (Logit forced to -inf)")
            blocked_cache.add(cache_key)
    return biased


def make_sampling_proposal_fn(model, tokenizer, device: str,
                              max_new_tokens: int = 128,
                              temperature: float = 0.8,
                              top_p: float = 0.95,
                              met_gate: Optional[METGate] = None,
                              met_tracker: Optional[MetStateTracker] = None):
    if met_tracker is None and met_gate is not None:
        met_tracker = MetStateTracker(
            entropy_threshold=met_gate.entropy_threshold,
            caution_window=1,
        )

    def _proposal(prompt: str, partial_code: str, block_index: int, num_options: int) -> List[CodeBlockProposal]:
        if not partial_code.strip():
            stage_prompt = (
                f"{prompt}\n\n"
                "Return ONLY the first executable Python block. "
                f"End unfinished regions with `{BLOCK_MARKER}`. "
                "Example: function signature plus `pass  # TODO: next_block`."
            )
        else:
            stage_prompt = (
                f"{prompt}\n\nCurrent partial code:\n{partial_code}\n"
                "Return ONLY the next logical Python block that should replace the next TODO marker. "
                "Keep indentation correct. If the implementation is complete, do not emit a TODO marker."
            )

        proposals: List[CodeBlockProposal] = []
        blocked_cache: set[tuple[str, int]] = set()
        for proposal_index in range(num_options):
            input_ids = _tokenize_to_tensor(tokenizer, stage_prompt, device)
            generated_ids: list[int] = []
            generated_text = ""

            try:
                with lod_compute(tier="low_fi"):
                    for _ in range(max_new_tokens):
                        outputs = model(input_ids=input_ids, inference_mode=True)
                        logits = outputs["logits"][:, -1, :]
                        if met_tracker is not None:
                            met_decision = met_tracker.observe(logits)
                            if met_decision["trigger_system2"] and not met_decision["forced_by_inertia"]:
                                print(
                                    f"[MET] High Entropy Detected "
                                    f"(H={met_decision['entropy']:.3f}). Engaging MCTS Reasoning..."
                                )
                        elif met_gate is not None:
                            entropy = calculate_entropy(logits)
                            if met_gate.should_trigger_system2(logits):
                                print(f"[MET] High Entropy Detected (H={entropy:.3f}). Engaging MCTS Reasoning...")
                        current_code_prefix = (
                            partial_code.rstrip() + "\n" + generated_text
                            if partial_code.strip() else generated_text
                        )
                        logits = _apply_symbol_bias(
                            tokenizer=tokenizer,
                            logits=logits,
                            code_prefix=current_code_prefix,
                            blocked_cache=blocked_cache,
                        )
                        next_token = _sample_next_token(
                            logits=logits,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=50,
                        )
                        token_id = int(next_token.item())
                        generated_ids.append(token_id)
                        input_ids = torch.cat([input_ids, next_token], dim=1)
                        generated_text = _decode_ids(tokenizer, generated_ids)

                        if "\n\n" in generated_text or BLOCK_MARKER in generated_text:
                            break
            except Exception:
                proposals.append(_fallback_generated_block(block_index, proposal_index))
                continue

            decoded = _decode_ids(tokenizer, generated_ids)
            proposals.append(_extract_block_fragment(decoded, fallback_terminal=block_index >= 2))
        return proposals

    return _proposal


def make_high_fi_backprop_callback(model, tokenizer, device: str, optimizer=None,
                                   enable_weight_updates: bool = True,
                                   pre_backward_hook: Optional[Callable[..., Optional[Dict[str, Any]]]] = None):
    import torch
    from novamind.core.device_manager import is_high_perf_mode

    def _encode(text: str) -> torch.Tensor:
        if hasattr(tokenizer, "__call__"):
            tokens = tokenizer(text, return_tensors="pt")
            return tokens.input_ids.to(device)
        return torch.tensor([tokenizer.encode(text)], device=device)

    def _callback(prompt: str, code: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        def _zero_grad():
            if optimizer is None:
                return
            try:
                optimizer.zero_grad(set_to_none=True)
            except TypeError:
                optimizer.zero_grad()

        was_training = model.training
        model.train()
        try:
            training_text = f"{prompt}\n\n{code}".strip()
            input_ids = _encode(training_text)
            should_update = bool(enable_weight_updates and optimizer is not None and is_high_perf_mode())

            _zero_grad()

            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs["loss"]
            if loss is None:
                raise RuntimeError("High-Fi backprop requires model outputs to include loss")
            loss = torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=-1e4)
            loss_before = float(loss.detach().item())
            model._sga_target_tensor = outputs.get("hidden_states")
            surgery_result = None
            hook_handle = None
            if pre_backward_hook is not None:
                surgery_result = pre_backward_hook(
                    loss=loss,
                    model=model,
                    metadata=metadata or {},
                    total_tokens=int(input_ids.shape[1]),
                    training_text=training_text,
                    prompt=prompt,
                    code=code,
                    outputs=outputs,
                )
                if surgery_result is not None:
                    hook_handle = surgery_result.get("handle")

            if should_update:
                loss.backward()
                optimizer.step()
                _zero_grad()
                with torch.no_grad():
                    post_outputs = model(input_ids=input_ids, labels=input_ids)
                    post_loss = post_outputs["loss"]
                if post_loss is None:
                    loss_after = loss_before
                else:
                    post_loss = torch.nan_to_num(post_loss, nan=0.0, posinf=1e4, neginf=-1e4)
                    loss_after = float(post_loss.detach().item())
            else:
                loss_after = loss_before

            if hook_handle is not None:
                hook_handle.remove()
            if hasattr(model, "_sga_target_tensor"):
                delattr(model, "_sga_target_tensor")

            return {
                "loss_before": loss_before,
                "loss": loss_before,
                "token_count": int(input_ids.numel()),
                "updated": bool(should_update),
                "simulated": bool(not should_update),
                "loss_after": loss_after,
                "sga_applied": bool(surgery_result and surgery_result.get("applied")),
                "sga_line": None if not surgery_result else surgery_result.get("failing_line"),
            }
        finally:
            model.train(was_training)

    return _callback


def clone_search_state(obj):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: clone_search_state(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [clone_search_state(v) for v in obj]
    return copy.deepcopy(obj)
