"""
Generate pure-Python algorithmic training data for the NovaMind "Sniper" path.

The generator only uses the Python standard library and emits JSONL records:

    {"prompt": "...", "solution": "...", "tests": "..."}

Each record is fully verifiable through hidden assertions so it can drive both
supervised learning and the MCTS + sandbox reasoning loop.

Usage:
    python data/sniper_dataset_gen.py --output data/sniper_train.jsonl --samples 10000
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _json_line(record: dict) -> str:
    return json.dumps(record, ensure_ascii=True)


def _make_binary_search_case(rng: random.Random, idx: int) -> dict:
    fn_name = f"binary_search_{idx}"
    size = rng.randint(8, 24)
    values = sorted(rng.sample(range(-150, 151), size))
    target_present = rng.random() < 0.7
    if target_present:
        target = rng.choice(values)
        expected = values.index(target)
    else:
        pool = [x for x in range(-170, 171) if x not in values]
        target = rng.choice(pool)
        expected = -1

    prompt = (
        f"Write a Python function `def {fn_name}(nums, target):` that returns the index "
        f"of `target` in a sorted list using binary search, or `-1` if it is absent."
    )
    solution = "\n".join([
        f"def {fn_name}(nums, target):",
        "    left, right = 0, len(nums) - 1",
        "    while left <= right:",
        "        mid = (left + right) // 2",
        "        if nums[mid] == target:",
        "            return mid",
        "        if nums[mid] < target:",
        "            left = mid + 1",
        "        else:",
        "            right = mid - 1",
        "    return -1",
    ])
    tests = "\n".join([
        f"nums = {values!r}",
        f"assert {fn_name}(nums, {target}) == {expected}",
        f"assert {fn_name}([], {target}) == -1",
        f"assert {fn_name}({values!r}, {values[0]}) == 0",
        f"assert {fn_name}({values!r}, {values[-1]}) == {len(values) - 1}",
    ])
    return {"prompt": prompt, "solution": solution, "tests": tests, "task_type": "binary_search"}


def _make_dp_stairs_case(rng: random.Random, idx: int) -> dict:
    fn_name = f"count_ways_{idx}"
    n = rng.randint(3, 14)
    allow_three = rng.random() < 0.5

    def count_ways(steps: int) -> int:
        dp = [0] * (steps + 1)
        dp[0] = 1
        for i in range(1, steps + 1):
            dp[i] += dp[i - 1]
            if i >= 2:
                dp[i] += dp[i - 2]
            if allow_three and i >= 3:
                dp[i] += dp[i - 3]
        return dp[steps]

    expected = count_ways(n)
    mode_text = "1, 2, or 3" if allow_three else "1 or 2"
    solution_lines = [
        f"def {fn_name}(n):",
        "    dp = [0] * (n + 1)",
        "    dp[0] = 1",
        "    for i in range(1, n + 1):",
        "        dp[i] += dp[i - 1]",
        "        if i >= 2:",
        "            dp[i] += dp[i - 2]",
    ]
    if allow_three:
        solution_lines.extend([
            "        if i >= 3:",
            "            dp[i] += dp[i - 3]",
        ])
    solution_lines.append("    return dp[n]")

    prompt = (
        f"Write a Python function `def {fn_name}(n):` that returns how many distinct ways "
        f"there are to climb `n` steps when you may move {mode_text} steps at a time."
    )
    tests = "\n".join([
        f"assert {fn_name}({n}) == {expected}",
        f"assert {fn_name}(1) == 1",
        f"assert {fn_name}(2) == {count_ways(2)}",
        f"assert {fn_name}(3) == {count_ways(3)}",
    ])
    return {"prompt": prompt, "solution": "\n".join(solution_lines), "tests": tests, "task_type": "dynamic_programming"}


def _make_graph_bfs_case(rng: random.Random, idx: int) -> dict:
    fn_name = f"shortest_path_{idx}"
    rows = rng.randint(3, 5)
    cols = rng.randint(3, 5)
    grid = []
    for _ in range(rows):
        row = [0 if rng.random() > 0.28 else 1 for _ in range(cols)]
        grid.append(row)
    grid[0][0] = 0
    grid[-1][-1] = 0

    def shortest_path(board: list[list[int]]) -> int:
        queue = [(0, 0, 0)]
        seen = {(0, 0)}
        while queue:
            r, c, dist = queue.pop(0)
            if (r, c) == (rows - 1, cols - 1):
                return dist
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                    continue
                if board[nr][nc] == 1 or (nr, nc) in seen:
                    continue
                seen.add((nr, nc))
                queue.append((nr, nc, dist + 1))
        return -1

    expected = shortest_path(grid)
    prompt = (
        f"Write a Python function `def {fn_name}(grid):` that uses BFS to return the "
        f"shortest path length from the top-left cell to the bottom-right cell in a grid "
        f"where `0` means open and `1` means blocked. Return `-1` if no path exists."
    )
    solution = "\n".join([
        f"def {fn_name}(grid):",
        "    rows, cols = len(grid), len(grid[0])",
        "    queue = [(0, 0, 0)]",
        "    seen = {(0, 0)}",
        "    while queue:",
        "        r, c, dist = queue.pop(0)",
        "        if (r, c) == (rows - 1, cols - 1):",
        "            return dist",
        "        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):",
        "            nr, nc = r + dr, c + dc",
        "            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:",
        "                continue",
        "            if grid[nr][nc] == 1 or (nr, nc) in seen:",
        "                continue",
        "            seen.add((nr, nc))",
        "            queue.append((nr, nc, dist + 1))",
        "    return -1",
    ])
    tests = "\n".join([
        f"grid = {grid!r}",
        f"assert {fn_name}(grid) == {expected}",
        f"assert {fn_name}([[0]]) == 0",
        f"assert {fn_name}([[0, 1], [1, 0]]) == -1",
    ])
    return {"prompt": prompt, "solution": solution, "tests": tests, "task_type": "graph_bfs"}


def _make_sort_case(rng: random.Random, idx: int) -> dict:
    fn_name = f"sort_by_key_{idx}"
    values = [rng.randint(-50, 50) for _ in range(rng.randint(6, 12))]
    reverse_order = rng.random() < 0.5

    def sort_key(value: int) -> tuple[int, int]:
        return (abs(value), value)

    expected = sorted(values, key=sort_key, reverse=reverse_order)
    order_text = "descending" if reverse_order else "ascending"
    prompt = (
        f"Write a Python function `def {fn_name}(values):` that returns a new list sorted "
        f"by absolute value and then by the raw value in {order_text} order."
    )
    solution = "\n".join([
        f"def {fn_name}(values):",
        f"    return sorted(values, key=lambda value: (abs(value), value), reverse={reverse_order})",
    ])
    tests = "\n".join([
        f"assert {fn_name}({values!r}) == {expected!r}",
        f"assert {fn_name}([]) == []",
        f"assert {fn_name}([3, -3, 2]) == {sorted([3, -3, 2], key=sort_key, reverse=reverse_order)!r}",
    ])
    return {"prompt": prompt, "solution": solution, "tests": tests, "task_type": "sorting"}


def _make_matrix_case(rng: random.Random, idx: int) -> dict:
    fn_name = f"transpose_add_{idx}"
    rows = rng.randint(2, 4)
    cols = rng.randint(2, 4)
    matrix = [[rng.randint(0, 9) for _ in range(cols)] for _ in range(rows)]
    bias = rng.randint(-3, 4)
    expected = [[matrix[r][c] + bias for r in range(rows)] for c in range(cols)]
    prompt = (
        f"Write a Python function `def {fn_name}(matrix):` that returns the transpose of "
        f"`matrix`, then adds {bias} to every element in the transposed output."
    )
    solution = "\n".join([
        f"def {fn_name}(matrix):",
        "    rows = len(matrix)",
        "    cols = len(matrix[0])",
        "    output = []",
        "    for c in range(cols):",
        "        row = []",
        "        for r in range(rows):",
        f"            row.append(matrix[r][c] + ({bias}))",
        "        output.append(row)",
        "    return output",
    ])
    tests = "\n".join([
        f"assert {fn_name}({matrix!r}) == {expected!r}",
        f"assert {fn_name}([[1, 2], [3, 4]]) == {[[1 + bias, 3 + bias], [2 + bias, 4 + bias]]!r}",
    ])
    return {"prompt": prompt, "solution": solution, "tests": tests, "task_type": "matrix"}


GENERATORS = (
    _make_binary_search_case,
    _make_dp_stairs_case,
    _make_graph_bfs_case,
    _make_sort_case,
    _make_matrix_case,
)


def generate_sniper_records(samples: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    records = []
    for idx in range(samples):
        generator = GENERATORS[idx % len(GENERATORS)]
        records.append(generator(rng, idx))
    rng.shuffle(records)
    return records


def write_sniper_dataset(output_path: str | Path, samples: int, seed: int = 42) -> Path:
    records = generate_sniper_records(samples=samples, seed=seed)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(_json_line(record) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description="Generate NovaMind Sniper JSONL data")
    parser.add_argument("--output", type=str, default="data/sniper_train.jsonl")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = write_sniper_dataset(args.output, samples=args.samples, seed=args.seed)
    print(f"[SniperData] Wrote {args.samples} records to {output}")


if __name__ == "__main__":
    main()
