"""

This module provides three core building blocks:
1. `execute_and_trace`: execute generated code and intercept the exact failing line.
2. `find_failing_ast_node`: map the failing line back to the most specific AST node.
3. `calculate_rollback_reward`: assign partial credit to code that ran before the crash.
"""

from __future__ import annotations

import ast
import traceback
from dataclasses import dataclass
from typing import Any, Dict, Optional


SCRIPT_FILENAME = "<generated_script>"
EXCLUDED_NODE_TYPES = {
    "Load", "Store", "Del", "Constant", "Name", "arg", "arguments",
}

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
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}


@dataclass
class ExecutionTrace:
    success: bool
    failing_line_number: Optional[int]
    exception_type: Optional[str]
    exception_message: Optional[str]
    traceback_text: str
    namespace: Dict[str, Any]


@dataclass
class ASTNodeMatch:
    node_type: Optional[str]
    lineno: Optional[int]
    end_lineno: Optional[int]
    source_segment: Optional[str]


def execute_and_trace(source_code: str) -> ExecutionTrace:
    """
    Execute `source_code` in an isolated namespace and recover the exact failing line.

    The code is compiled with a synthetic filename so traceback frames point directly
    to the generated script instead of the surrounding sandbox wrapper.
    """
    namespace: Dict[str, Any] = {"__builtins__": SAFE_BUILTINS.copy()}

    try:
        compiled = compile(source_code, SCRIPT_FILENAME, "exec")
        exec(compiled, namespace, namespace)
        return ExecutionTrace(
            success=True,
            failing_line_number=None,
            exception_type=None,
            exception_message=None,
            traceback_text="",
            namespace=namespace,
        )
    except Exception as exc:
        tb_exception = traceback.TracebackException.from_exception(exc)
        failing_line = None

        for frame in reversed(tb_exception.stack):
            if frame.filename == SCRIPT_FILENAME:
                failing_line = frame.lineno
                break

        return ExecutionTrace(
            success=False,
            failing_line_number=failing_line,
            exception_type=exc.__class__.__name__,
            exception_message=str(exc),
            traceback_text="".join(tb_exception.format()),
            namespace=namespace,
        )


def _node_span(node: ast.AST) -> Optional[tuple[int, int]]:
    lineno = getattr(node, "lineno", None)
    end_lineno = getattr(node, "end_lineno", lineno)
    if lineno is None:
        return None
    return lineno, end_lineno if end_lineno is not None else lineno


def find_failing_ast_node(source_code: str, failing_line_number: int) -> ASTNodeMatch:
    """
    Find the most specific AST node covering `failing_line_number`.

    "Most specific" is approximated as the smallest source span that still contains
    the failing line. This tends to prefer `Call`, `Assign`, or branch bodies over
    large enclosing function definitions.
    """
    tree = ast.parse(source_code)
    best_node = None
    best_priority = None

    for node in ast.walk(tree):
        span = _node_span(node)
        if span is None:
            continue
        start, end = span
        if start <= failing_line_number <= end:
            node_type = node.__class__.__name__
            if node_type in EXCLUDED_NODE_TYPES:
                continue

            width = end - start
            segment = ast.get_source_segment(source_code, node) or ""
            complexity_penalty = 0 if segment.strip() else 1
            priority = (width, complexity_penalty, -start)

            if best_priority is None or priority < best_priority:
                best_node = node
                best_priority = priority

    if best_node is None:
        return ASTNodeMatch(
            node_type=None,
            lineno=None,
            end_lineno=None,
            source_segment=None,
        )

    segment = ast.get_source_segment(source_code, best_node)
    if segment is None:
        lines = source_code.splitlines()
        lineno = getattr(best_node, "lineno", failing_line_number)
        end_lineno = getattr(best_node, "end_lineno", lineno)
        segment = "\n".join(lines[lineno - 1:end_lineno])

    return ASTNodeMatch(
        node_type=best_node.__class__.__name__,
        lineno=getattr(best_node, "lineno", None),
        end_lineno=getattr(best_node, "end_lineno", getattr(best_node, "lineno", None)),
        source_segment=segment,
    )


def calculate_rollback_reward(total_lines: int, failing_line: Optional[int]) -> float:
    """
    Assign partial credit to code that executed before the crash.

    Success:
        +100

    Failure:
        successful_prefix_reward = (failing_line - 1) * 0.5
        failing_line_penalty = -50
    """
    if failing_line is None:
        return 100.0

    bounded_line = max(1, min(failing_line, max(total_lines, 1)))
    return ((bounded_line - 1) * 0.5) - 50.0


if __name__ == "__main__":
    demo_script = """def safe_prefix(values):
    doubled = [value * 2 for value in values]
    total = sum(doubled)
    return total

result = safe_prefix([1, 2, 3])
crash = result / 0
"""

    print("== Demo Script ==")
    print(demo_script)

    execution = execute_and_trace(demo_script)
    print("== Execution Trace ==")
    print(f"success: {execution.success}")
    print(f"failing_line_number: {execution.failing_line_number}")
    print(f"exception_type: {execution.exception_type}")
    print(f"exception_message: {execution.exception_message}")

    if not execution.success and execution.failing_line_number is not None:
        match = find_failing_ast_node(demo_script, execution.failing_line_number)
        reward = calculate_rollback_reward(
            total_lines=len(demo_script.strip().splitlines()),
            failing_line=execution.failing_line_number,
        )

        print("== AST Match ==")
        print(f"node_type: {match.node_type}")
        print(f"lineno: {match.lineno}")
        print(f"end_lineno: {match.end_lineno}")
        print("source_segment:")
        print(match.source_segment)

        print("== Rollback Reward ==")
        print(reward)

        print("== Traceback ==")
        print(execution.traceback_text)
