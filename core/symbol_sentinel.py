"""
NovaMind — Incremental Symbol Flow Sentinel.

The sentinel statically inspects partial Python code and identifies references
to symbols that are accessed before being defined in the current snippet.
"""

from __future__ import annotations

import ast
import builtins
import textwrap
from dataclasses import dataclass, field
from typing import Iterable, Set


PYTHON_BUILTINS = set(dir(builtins))


def _trim_to_parseable_prefix(source: str) -> str:
    lines = source.splitlines()
    while lines:
        candidate = "\n".join(lines).rstrip()
        if not candidate.strip():
            return ""
        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError as exc:
            lineno = exc.lineno or len(lines)
            cut_index = max(0, lineno - 1)
            lines = lines[:cut_index]
    return ""


def _collect_target_names(target: ast.AST) -> Set[str]:
    names: Set[str] = set()
    for node in ast.walk(target):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
    return names


@dataclass
class _Scope:
    defined: Set[str] = field(default_factory=set)
    accessed: Set[str] = field(default_factory=set)


class SymbolSentinel(ast.NodeVisitor):
    def __init__(self):
        self._scopes = [_Scope()]
        self._cached_source = ""
        self._cached_parseable = ""
        self._cached_defined: Set[str] = set()
        self._cached_accessed: Set[str] = set()

    @property
    def defined_symbols(self) -> Set[str]:
        merged: Set[str] = set()
        for scope in self._scopes:
            merged |= scope.defined
        return merged

    @property
    def accessed_symbols(self) -> Set[str]:
        merged: Set[str] = set()
        for scope in self._scopes:
            merged |= scope.accessed
        return merged

    @classmethod
    def from_code(cls, partial_code: str) -> "SymbolSentinel":
        sentinel = cls()
        parseable = _trim_to_parseable_prefix(partial_code)
        if parseable.strip():
            tree = ast.parse(parseable)
            sentinel.visit(tree)
        return sentinel

    def get_undefined_references(self, partial_code: str) -> Set[str]:
        parseable = _trim_to_parseable_prefix(partial_code)
        if not parseable.strip():
            return set()

        if parseable.startswith(self._cached_parseable):
            incremental = self._incremental_update(parseable)
            if incremental is not None:
                return incremental

        self._full_refresh(parseable)
        return self._cached_accessed - self._cached_defined - PYTHON_BUILTINS

    def _full_refresh(self, parseable: str):
        fresh = self.from_code(parseable)
        self._cached_source = parseable
        self._cached_parseable = parseable
        self._cached_defined = set(fresh.defined_symbols)
        self._cached_accessed = set(fresh.accessed_symbols)

    def _incremental_update(self, parseable: str) -> Set[str] | None:
        new_segment = parseable[len(self._cached_parseable):]
        if not new_segment.strip():
            return self._cached_accessed - self._cached_defined - PYTHON_BUILTINS

        try:
            tree = ast.parse(textwrap.dedent(new_segment))
        except SyntaxError:
            return None

        incremental = SymbolSentinel()
        incremental._scopes[0].defined.update(self._cached_defined)
        incremental.visit(tree)
        self._cached_source = parseable
        self._cached_parseable = parseable
        self._cached_defined |= incremental.defined_symbols
        self._cached_accessed |= incremental.accessed_symbols
        return self._cached_accessed - self._cached_defined - PYTHON_BUILTINS

    def _current_scope(self) -> _Scope:
        return self._scopes[-1]

    def _mark_defined(self, names: Iterable[str]):
        self._current_scope().defined.update(name for name in names if name and name != "_")

    def _mark_accessed(self, name: str):
        if name and name != "_":
            self._current_scope().accessed.add(name)

    def visit_Assign(self, node: ast.Assign):
        self.visit(node.value)
        for target in node.targets:
            self._mark_defined(_collect_target_names(target))

    def visit_AnnAssign(self, node: ast.AnnAssign):
        if node.value is not None:
            self.visit(node.value)
        self._mark_defined(_collect_target_names(node.target))

    def visit_AugAssign(self, node: ast.AugAssign):
        self.visit(node.value)
        if isinstance(node.target, ast.Name):
            self._mark_accessed(node.target.id)
        else:
            self.visit(node.target)
        self._mark_defined(_collect_target_names(node.target))

    def visit_For(self, node: ast.For):
        self.visit(node.iter)
        self._mark_defined(_collect_target_names(node.target))
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)

    def visit_AsyncFor(self, node: ast.AsyncFor):
        self.visit_For(node)

    def visit_With(self, node: ast.With):
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                self._mark_defined(_collect_target_names(item.optional_vars))
        for stmt in node.body:
            self.visit(stmt)

    def visit_AsyncWith(self, node: ast.AsyncWith):
        self.visit_With(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        if node.type is not None:
            self.visit(node.type)
        if node.name:
            self._mark_defined([node.name])
        for stmt in node.body:
            self.visit(stmt)

    def visit_Import(self, node: ast.Import):
        aliases = [alias.asname or alias.name.split(".")[0] for alias in node.names]
        self._mark_defined(aliases)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        aliases = [alias.asname or alias.name for alias in node.names]
        self._mark_defined(aliases)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._mark_defined([node.name])
        self._visit_function_like(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._mark_defined([node.name])
        self._visit_function_like(node)

    def visit_Lambda(self, node: ast.Lambda):
        self._visit_function_like(node)

    def _visit_function_like(self, node: ast.AST):
        self._scopes.append(_Scope())
        args = getattr(node, "args", None)
        if args is not None:
            for arg in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs):
                self._mark_defined([arg.arg])
            if args.vararg is not None:
                self._mark_defined([args.vararg.arg])
            if args.kwarg is not None:
                self._mark_defined([args.kwarg.arg])

        body = getattr(node, "body", [])
        if isinstance(body, list):
            for stmt in body:
                self.visit(stmt)
        else:
            self.visit(body)

        inner_scope = self._scopes.pop()
        outer_scope = self._current_scope()
        outer_scope.accessed.update(
            name for name in inner_scope.accessed
            if name not in inner_scope.defined
        )

    def visit_ClassDef(self, node: ast.ClassDef):
        self._mark_defined([node.name])
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_Name(self, node: ast.Name):
        if isinstance(node.ctx, ast.Load):
            self._mark_accessed(node.id)
        elif isinstance(node.ctx, ast.Store):
            self._mark_defined([node.id])


if __name__ == "__main__":
    demo = "x = 1\nz = x + y\n"
    sentinel = SymbolSentinel()
    undefined = sentinel.get_undefined_references(demo)
    print("Defined symbols:", sorted(SymbolSentinel.from_code(demo).defined_symbols))
    print("Accessed symbols:", sorted(SymbolSentinel.from_code(demo).accessed_symbols))
    print("Undefined references:", sorted(undefined))
