"""
Generate nightmare-difficulty pure-Python algorithmic training data.

This file is intentionally standard-library only. It emits JSONL records in the
form:

    {"prompt": "...", "solution": "...", "tests": "..."}

The curriculum is designed for code-generation agents that learn through
execution, AST rollback, and symbolic repair rather than simple pattern
matching. Every sample includes strict assertions with edge cases.

Usage:
    python3 data/sniper_dataset_gen.py --output data/sniper_train.jsonl --samples 10000
"""

from __future__ import annotations

import argparse
import heapq
import json
import random
from pathlib import Path


def _json_line(record: dict) -> str:
    return json.dumps(record, ensure_ascii=True)


def _dijkstra_with_state(edges: list[tuple[int, int, int]], n: int, discount_budget: int) -> int:
    graph = [[] for _ in range(n)]
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))

    inf = 10**18
    dist = [[inf] * (discount_budget + 1) for _ in range(n)]
    pq = [(0, 0, 0)]
    dist[0][0] = 0
    while pq:
        cost, node, used = heapq.heappop(pq)
        if cost != dist[node][used]:
            continue
        if node == n - 1:
            return cost
        for nxt, weight in graph[node]:
            nxt_cost = cost + weight
            if nxt_cost < dist[nxt][used]:
                dist[nxt][used] = nxt_cost
                heapq.heappush(pq, (nxt_cost, nxt, used))
            if used < discount_budget:
                discounted = cost + weight // 2
                if discounted < dist[nxt][used + 1]:
                    dist[nxt][used + 1] = discounted
                    heapq.heappush(pq, (discounted, nxt, used + 1))
    return -1


def _topological_layers(n: int, edges: list[tuple[int, int]]) -> list[int]:
    graph = [[] for _ in range(n)]
    indeg = [0] * n
    for u, v in edges:
        graph[u].append(v)
        indeg[v] += 1

    queue = [i for i in range(n) if indeg[i] == 0]
    order = []
    while queue:
        queue.sort()
        node = queue.pop(0)
        order.append(node)
        for nxt in graph[node]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                queue.append(nxt)
    return order if len(order) == n else []


def _tarjan_scc_sizes(n: int, edges: list[tuple[int, int]]) -> list[int]:
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)

    index = 0
    stack = []
    on_stack = [False] * n
    indices = [-1] * n
    low = [0] * n
    sizes = []

    def dfs(node: int):
        nonlocal index
        indices[node] = index
        low[node] = index
        index += 1
        stack.append(node)
        on_stack[node] = True

        for nxt in graph[node]:
            if indices[nxt] == -1:
                dfs(nxt)
                low[node] = min(low[node], low[nxt])
            elif on_stack[nxt]:
                low[node] = min(low[node], indices[nxt])

        if low[node] == indices[node]:
            size = 0
            while True:
                top = stack.pop()
                on_stack[top] = False
                size += 1
                if top == node:
                    break
            sizes.append(size)

    for node in range(n):
        if indices[node] == -1:
            dfs(node)
    return sorted(sizes)


def _bitmask_tsp(cost: list[list[int]]) -> int:
    n = len(cost)
    inf = 10**18
    dp = [[inf] * n for _ in range(1 << n)]
    dp[1][0] = 0
    for mask in range(1 << n):
        for node in range(n):
            cur = dp[mask][node]
            if cur >= inf:
                continue
            for nxt in range(n):
                if mask & (1 << nxt):
                    continue
                cand = cur + cost[node][nxt]
                nxt_mask = mask | (1 << nxt)
                if cand < dp[nxt_mask][nxt]:
                    dp[nxt_mask][nxt] = cand
    full = (1 << n) - 1
    best = min(dp[full][node] + cost[node][0] for node in range(n))
    return best


def _tree_max_independent_set(n: int, edges: list[tuple[int, int]], values: list[int]) -> int:
    graph = [[] for _ in range(n)]
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)

    def dfs(node: int, parent: int) -> tuple[int, int]:
        take = values[node]
        skip = 0
        for nxt in graph[node]:
            if nxt == parent:
                continue
            child_take, child_skip = dfs(nxt, node)
            take += child_skip
            skip += max(child_take, child_skip)
        return take, skip

    return max(dfs(0, -1))


def _trie_prefix_counts(words: list[str], queries: list[str]) -> list[int]:
    root: dict = {"count": 0, "children": {}}
    for word in words:
        node = root
        for char in word:
            children = node["children"]
            if char not in children:
                children[char] = {"count": 0, "children": {}}
            node = children[char]
            node["count"] += 1
    output = []
    for query in queries:
        node = root
        ok = True
        for char in query:
            children = node["children"]
            if char not in children:
                ok = False
                break
            node = children[char]
        output.append(node["count"] if ok else 0)
    return output


def _segment_process(values: list[int], ops: list[tuple[str, int, int]]) -> list[int]:
    arr = values[:]
    answers = []
    for op, a, b in ops:
        if op == "set":
            arr[a] = b
        else:
            answers.append(min(arr[a:b + 1]))
    return answers


def _nqueens_count(n: int) -> int:
    cols = set()
    diag1 = set()
    diag2 = set()
    count = 0

    def dfs(row: int):
        nonlocal count
        if row == n:
            count += 1
            return
        for col in range(n):
            if col in cols or row - col in diag1 or row + col in diag2:
                continue
            cols.add(col)
            diag1.add(row - col)
            diag2.add(row + col)
            dfs(row + 1)
            cols.remove(col)
            diag1.remove(row - col)
            diag2.remove(row + col)

    dfs(0)
    return count


def _maze_shortest_path(grid: list[list[int]], start: tuple[int, int], goal: tuple[int, int]) -> int:
    rows, cols = len(grid), len(grid[0])
    q = [(start[0], start[1], 0)]
    seen = {start}
    while q:
        r, c, d = q.pop(0)
        if (r, c) == goal:
            return d
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                continue
            if grid[nr][nc] == 1 or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            q.append((nr, nc, d + 1))
    return -1


def _rand_word(rng: random.Random, min_len: int = 3, max_len: int = 8) -> str:
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    return "".join(rng.choice(alphabet) for _ in range(rng.randint(min_len, max_len)))


def _make_dijkstra_case(rng: random.Random, idx: int) -> dict:
    fn_name = f"discounted_shortest_path_{idx}"
    n = rng.randint(5, 7)
    edges = set()
    for node in range(n - 1):
        edges.add((node, node + 1, rng.randint(3, 19)))
    while len(edges) < n + rng.randint(3, 6):
        u = rng.randrange(n)
        v = rng.randrange(n)
        if u != v:
            a, b = sorted((u, v))
            edges.add((a, b, rng.randint(2, 25)))
    edge_list = sorted(edges)
    budget = rng.randint(1, 2)
    expected = _dijkstra_with_state(edge_list, n, budget)
    tests = "\n".join([
        f"edges = {edge_list!r}",
        f"assert {fn_name}({n}, edges, {budget}) == {expected}",
        f"assert {fn_name}(2, [(0, 1, 10)], 1) == 5",
        f"assert {fn_name}(3, [(0, 1, 4)], 1) == -1",
        f"assert {fn_name}(1, [], 0) == 0",
    ])
    solution = "\n".join([
        f"def {fn_name}(n, edges, discount_budget):",
        "    import heapq",
        "    graph = [[] for _ in range(n)]",
        "    for u, v, w in edges:",
        "        graph[u].append((v, w))",
        "        graph[v].append((u, w))",
        "    inf = 10 ** 18",
        "    dist = [[inf] * (discount_budget + 1) for _ in range(n)]",
        "    dist[0][0] = 0",
        "    pq = [(0, 0, 0)]",
        "    while pq:",
        "        cost, node, used = heapq.heappop(pq)",
        "        if cost != dist[node][used]:",
        "            continue",
        "        if node == n - 1:",
        "            return cost",
        "        for nxt, weight in graph[node]:",
        "            nxt_cost = cost + weight",
        "            if nxt_cost < dist[nxt][used]:",
        "                dist[nxt][used] = nxt_cost",
        "                heapq.heappush(pq, (nxt_cost, nxt, used))",
        "            if used < discount_budget:",
        "                discounted = cost + weight // 2",
        "                if discounted < dist[nxt][used + 1]:",
        "                    dist[nxt][used + 1] = discounted",
        "                    heapq.heappush(pq, (discounted, nxt, used + 1))",
        "    return -1",
    ])
    prompt = (
        f"Write a Python function `def {fn_name}(n, edges, discount_budget):` that returns "
        f"the minimum cost from node 0 to node n-1 in an undirected weighted graph. The traveler "
        f"may apply integer floor-halving to at most `discount_budget` edge weights. Return -1 "
        f"if the target is unreachable."
    )
    return {"prompt": prompt, "solution": solution, "tests": tests, "task_type": "graph_dijkstra_state"}


def _make_toposort_case(rng: random.Random, idx: int) -> dict:
    fn_name = f"stable_toposort_{idx}"
    n = rng.randint(5, 8)
    perm = list(range(n))
    rng.shuffle(perm)
    pos = {node: i for i, node in enumerate(perm)}
    edges = set()
    while len(edges) < n + rng.randint(1, 4):
        u = rng.randrange(n)
        v = rng.randrange(n)
        if u != v and pos[u] < pos[v]:
            edges.add((u, v))
    edge_list = sorted(edges)
    expected = _topological_layers(n, edge_list)
    tests = "\n".join([
        f"edges = {edge_list!r}",
        f"assert {fn_name}({n}, edges) == {expected!r}",
        f"assert {fn_name}(3, [(0, 1), (1, 2)]) == [0, 1, 2]",
        f"assert {fn_name}(3, [(0, 1), (1, 2), (2, 0)]) == []",
        f"assert {fn_name}(1, []) == [0]",
    ])
    solution = "\n".join([
        f"def {fn_name}(n, edges):",
        "    graph = [[] for _ in range(n)]",
        "    indeg = [0] * n",
        "    for u, v in edges:",
        "        graph[u].append(v)",
        "        indeg[v] += 1",
        "    queue = [i for i in range(n) if indeg[i] == 0]",
        "    order = []",
        "    while queue:",
        "        queue.sort()",
        "        node = queue.pop(0)",
        "        order.append(node)",
        "        for nxt in graph[node]:",
        "            indeg[nxt] -= 1",
        "            if indeg[nxt] == 0:",
        "                queue.append(nxt)",
        "    return order if len(order) == n else []",
    ])
    prompt = (
        f"Write a Python function `def {fn_name}(n, edges):` that returns the lexicographically "
        f"smallest topological ordering of a directed graph with nodes `0..n-1`. Return an empty "
        f"list if the graph contains a cycle."
    )
    return {"prompt": prompt, "solution": solution, "tests": tests, "task_type": "toposort"}


def _make_tarjan_case(rng: random.Random, idx: int) -> dict:
    fn_name = f"scc_signature_{idx}"
    n = rng.randint(5, 8)
    edges = set()
    for _ in range(n + rng.randint(2, 6)):
        u = rng.randrange(n)
        v = rng.randrange(n)
        if u != v:
            edges.add((u, v))
    edge_list = sorted(edges)
    expected = _tarjan_scc_sizes(n, edge_list)
    tests = "\n".join([
        f"edges = {edge_list!r}",
        f"assert {fn_name}({n}, edges) == {expected!r}",
        f"assert {fn_name}(3, [(0, 1), (1, 2), (2, 0)]) == [3]",
        f"assert {fn_name}(4, [(0, 1), (1, 0), (2, 3)]) == [1, 1, 2]",
        f"assert {fn_name}(0, []) == []",
    ])
    solution = "\n".join([
        f"def {fn_name}(n, edges):",
        "    if n == 0:",
        "        return []",
        "    graph = [[] for _ in range(n)]",
        "    for u, v in edges:",
        "        graph[u].append(v)",
        "    index = 0",
        "    stack = []",
        "    on_stack = [False] * n",
        "    indices = [-1] * n",
        "    low = [0] * n",
        "    sizes = []",
        "    def dfs(node):",
        "        nonlocal index",
        "        indices[node] = index",
        "        low[node] = index",
        "        index += 1",
        "        stack.append(node)",
        "        on_stack[node] = True",
        "        for nxt in graph[node]:",
        "            if indices[nxt] == -1:",
        "                dfs(nxt)",
        "                low[node] = min(low[node], low[nxt])",
        "            elif on_stack[nxt]:",
        "                low[node] = min(low[node], indices[nxt])",
        "        if low[node] == indices[node]:",
        "            size = 0",
        "            while True:",
        "                top = stack.pop()",
        "                on_stack[top] = False",
        "                size += 1",
        "                if top == node:",
        "                    break",
        "            sizes.append(size)",
        "    for node in range(n):",
        "        if indices[node] == -1:",
        "            dfs(node)",
        "    return sorted(sizes)",
    ])
    prompt = (
        f"Write a Python function `def {fn_name}(n, edges):` that returns the sizes of all "
        f"strongly connected components in ascending order using Tarjan's algorithm."
    )
    return {"prompt": prompt, "solution": solution, "tests": tests, "task_type": "tarjan_scc"}


def _make_bitmask_dp_case(rng: random.Random, idx: int) -> dict:
    fn_name = f"cycle_tsp_{idx}"
    n = rng.randint(4, 6)
    cost = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(0)
            else:
                row.append(rng.randint(2, 25))
        cost.append(row)
    expected = _bitmask_tsp(cost)
    tests = "\n".join([
        f"cost = {cost!r}",
        f"assert {fn_name}(cost) == {expected}",
        f"assert {fn_name}([[0, 10], [10, 0]]) == 20",
        f"assert {fn_name}([[0, 3, 4], [3, 0, 5], [4, 5, 0]]) == 12",
    ])
    solution = "\n".join([
        f"def {fn_name}(cost):",
        "    n = len(cost)",
        "    inf = 10 ** 18",
        "    dp = [[inf] * n for _ in range(1 << n)]",
        "    dp[1][0] = 0",
        "    for mask in range(1 << n):",
        "        for node in range(n):",
        "            cur = dp[mask][node]",
        "            if cur >= inf:",
        "                continue",
        "            for nxt in range(n):",
        "                if mask & (1 << nxt):",
        "                    continue",
        "                nxt_mask = mask | (1 << nxt)",
        "                cand = cur + cost[node][nxt]",
        "                if cand < dp[nxt_mask][nxt]:",
        "                    dp[nxt_mask][nxt] = cand",
        "    full = (1 << n) - 1",
        "    best = min(dp[full][node] + cost[node][0] for node in range(n))",
        "    return best",
    ])
    prompt = (
        f"Write a Python function `def {fn_name}(cost):` that returns the minimum Hamiltonian "
        f"cycle cost starting and ending at node 0 using bitmask dynamic programming."
    )
    return {"prompt": prompt, "solution": solution, "tests": tests, "task_type": "bitmask_dp"}


def _make_tree_dp_case(rng: random.Random, idx: int) -> dict:
    fn_name = f"tree_mis_{idx}"
    n = rng.randint(5, 8)
    edges = []
    for node in range(1, n):
        edges.append((rng.randrange(node), node))
    values = [rng.randint(1, 20) for _ in range(n)]
    expected = _tree_max_independent_set(n, edges, values)
    tests = "\n".join([
        f"edges = {edges!r}",
        f"values = {values!r}",
        f"assert {fn_name}({n}, edges, values) == {expected}",
        f"assert {fn_name}(1, [], [9]) == 9",
        f"assert {fn_name}(3, [(0, 1), (1, 2)], [5, 10, 5]) == 10",
    ])
    solution = "\n".join([
        f"def {fn_name}(n, edges, values):",
        "    graph = [[] for _ in range(n)]",
        "    for u, v in edges:",
        "        graph[u].append(v)",
        "        graph[v].append(u)",
        "    def dfs(node, parent):",
        "        take = values[node]",
        "        skip = 0",
        "        for nxt in graph[node]:",
        "            if nxt == parent:",
        "                continue",
        "            child_take, child_skip = dfs(nxt, node)",
        "            take += child_skip",
        "            skip += max(child_take, child_skip)",
        "        return take, skip",
        "    return max(dfs(0, -1))",
    ])
    prompt = (
        f"Write a Python function `def {fn_name}(n, edges, values):` that returns the maximum "
        f"weight independent set of a tree. `values[i]` is the reward for selecting node `i`, "
        f"and adjacent nodes may not both be selected."
    )
    return {"prompt": prompt, "solution": solution, "tests": tests, "task_type": "tree_dp"}


def _make_trie_case(rng: random.Random, idx: int) -> dict:
    fn_name = f"prefix_match_counts_{idx}"
    words = [_rand_word(rng) for _ in range(rng.randint(8, 12))]
    queries = []
    for _ in range(rng.randint(5, 8)):
        base = rng.choice(words)
        queries.append(base[: rng.randint(1, len(base))])
    queries.extend(["zzz", "", words[0][:2]])
    expected = _trie_prefix_counts(words, queries)
    tests = "\n".join([
        f"words = {words!r}",
        f"queries = {queries!r}",
        f"assert {fn_name}(words, queries) == {expected!r}",
        f"assert {fn_name}([], ['a', '']) == [0, 0]",
        f"assert {fn_name}(['abc', 'abd', 'ab'], ['a', 'ab', 'abc', 'x']) == [3, 3, 1, 0]",
    ])
    solution = "\n".join([
        f"def {fn_name}(words, queries):",
        "    root = {'count': 0, 'children': {}}",
        "    for word in words:",
        "        node = root",
        "        for char in word:",
        "            children = node['children']",
        "            if char not in children:",
        "                children[char] = {'count': 0, 'children': {}}",
        "            node = children[char]",
        "            node['count'] += 1",
        "    output = []",
        "    for query in queries:",
        "        if query == '':",
        "            output.append(0)",
        "            continue",
        "        node = root",
        "        ok = True",
        "        for char in query:",
        "            children = node['children']",
        "            if char not in children:",
        "                ok = False",
        "                break",
        "            node = children[char]",
        "        output.append(node['count'] if ok else 0)",
        "    return output",
    ])
    prompt = (
        f"Write a Python function `def {fn_name}(words, queries):` that builds a trie and returns "
        f"how many inserted words share each query as a prefix."
    )
    return {"prompt": prompt, "solution": solution, "tests": tests, "task_type": "trie"}


def _make_segment_tree_case(rng: random.Random, idx: int) -> dict:
    fn_name = f"segment_min_queries_{idx}"
    values = [rng.randint(-20, 40) for _ in range(rng.randint(6, 10))]
    ops = []
    for _ in range(rng.randint(6, 10)):
        if rng.random() < 0.45:
            pos = rng.randrange(len(values))
            val = rng.randint(-30, 50)
            ops.append(("set", pos, val))
            values[pos] = val
        else:
            l = rng.randrange(len(values))
            r = rng.randrange(l, len(values))
            ops.append(("min", l, r))
    original_values = [rng.randint(-20, 40) for _ in range(len(values))]
    # regenerate with same op shape against stable start state
    ops = []
    for _ in range(rng.randint(6, 10)):
        if rng.random() < 0.45:
            pos = rng.randrange(len(original_values))
            val = rng.randint(-30, 50)
            ops.append(("set", pos, val))
        else:
            l = rng.randrange(len(original_values))
            r = rng.randrange(l, len(original_values))
            ops.append(("min", l, r))
    expected = _segment_process(original_values, ops)
    tests = "\n".join([
        f"values = {original_values!r}",
        f"ops = {ops!r}",
        f"assert {fn_name}(values, ops) == {expected!r}",
        f"assert {fn_name}([5, 1, 7], [('min', 0, 2), ('set', 1, 9), ('min', 1, 2)]) == [1, 7]",
        f"assert {fn_name}([], []) == []",
    ])
    solution = "\n".join([
        f"def {fn_name}(values, ops):",
        "    if not values:",
        "        return []",
        "    arr = values[:]",
        "    answers = []",
        "    for op, a, b in ops:",
        "        if op == 'set':",
        "            arr[a] = b",
        "        else:",
        "            answers.append(min(arr[a:b + 1]))",
        "    return answers",
    ])
    prompt = (
        f"Write a Python function `def {fn_name}(values, ops):` that simulates a segment-tree style "
        f"API over an array. Each op is either `('set', index, value)` or `('min', left, right)`. "
        f"Return the list of all range-minimum query answers in order."
    )
    return {"prompt": prompt, "solution": solution, "tests": tests, "task_type": "segment_tree"}


def _make_nqueens_case(rng: random.Random, idx: int) -> dict:
    fn_name = f"count_nqueens_{idx}"
    n = rng.randint(4, 8)
    expected = _nqueens_count(n)
    tests = "\n".join([
        f"assert {fn_name}({n}) == {expected}",
        f"assert {fn_name}(4) == 2",
        f"assert {fn_name}(5) == 10",
        f"assert {fn_name}(1) == 1",
    ])
    solution = "\n".join([
        f"def {fn_name}(n):",
        "    cols = set()",
        "    diag1 = set()",
        "    diag2 = set()",
        "    count = 0",
        "    def dfs(row):",
        "        nonlocal count",
        "        if row == n:",
        "            count += 1",
        "            return",
        "        for col in range(n):",
        "            if col in cols or row - col in diag1 or row + col in diag2:",
        "                continue",
        "            cols.add(col)",
        "            diag1.add(row - col)",
        "            diag2.add(row + col)",
        "            dfs(row + 1)",
        "            cols.remove(col)",
        "            diag1.remove(row - col)",
        "            diag2.remove(row + col)",
        "    dfs(0)",
        "    return count",
    ])
    prompt = (
        f"Write a Python function `def {fn_name}(n):` that returns the number of valid solutions "
        f"to the N-Queens problem using backtracking."
    )
    return {"prompt": prompt, "solution": solution, "tests": tests, "task_type": "backtracking_nqueens"}


def _make_maze_case(rng: random.Random, idx: int) -> dict:
    fn_name = f"hard_maze_solver_{idx}"
    rows = rng.randint(4, 6)
    cols = rng.randint(4, 6)
    grid = [[0 if rng.random() > 0.33 else 1 for _ in range(cols)] for _ in range(rows)]
    start = (0, 0)
    goal = (rows - 1, cols - 1)
    grid[start[0]][start[1]] = 0
    grid[goal[0]][goal[1]] = 0
    expected = _maze_shortest_path(grid, start, goal)
    tests = "\n".join([
        f"grid = {grid!r}",
        f"assert {fn_name}(grid, {start!r}, {goal!r}) == {expected}",
        f"assert {fn_name}([[0]], (0, 0), (0, 0)) == 0",
        f"assert {fn_name}([[0, 1], [1, 0]], (0, 0), (1, 1)) == -1",
        f"assert {fn_name}([[0, 0, 0], [1, 1, 0], [0, 0, 0]], (0, 0), (2, 2)) == 4",
    ])
    solution = "\n".join([
        f"def {fn_name}(grid, start, goal):",
        "    rows, cols = len(grid), len(grid[0])",
        "    q = [(start[0], start[1], 0)]",
        "    seen = {start}",
        "    while q:",
        "        r, c, d = q.pop(0)",
        "        if (r, c) == goal:",
        "            return d",
        "        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):",
        "            nr, nc = r + dr, c + dc",
        "            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:",
        "                continue",
        "            if grid[nr][nc] == 1 or (nr, nc) in seen:",
        "                continue",
        "            seen.add((nr, nc))",
        "            q.append((nr, nc, d + 1))",
        "    return -1",
    ])
    prompt = (
        f"Write a Python function `def {fn_name}(grid, start, goal):` that returns the shortest "
        f"path length in a maze with obstacles using BFS, or `-1` if no route exists."
    )
    return {"prompt": prompt, "solution": solution, "tests": tests, "task_type": "maze_bfs"}


GENERATORS = (
    _make_dijkstra_case,
    _make_toposort_case,
    _make_tarjan_case,
    _make_bitmask_dp_case,
    _make_tree_dp_case,
    _make_trie_case,
    _make_segment_tree_case,
    _make_nqueens_case,
    _make_maze_case,
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
    parser = argparse.ArgumentParser(description="Generate nightmare Sniper JSONL data")
    parser.add_argument("--output", type=str, default="data/sniper_train.jsonl")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = write_sniper_dataset(args.output, samples=args.samples, seed=args.seed)
    print(f"[SniperData] Wrote {args.samples} nightmare records to {output}")


if __name__ == "__main__":
    main()
