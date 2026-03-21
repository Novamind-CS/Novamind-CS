"""
NovaMind synthetic reasoning curricula.
"""

from __future__ import annotations

from typing import List, Dict


def get_level_1_dataset() -> List[Dict[str, str]]:
    return [
        {
            "prompt": (
                "Write a Python function `def is_even(n: int) -> bool:` "
                "that returns True when n is even and False otherwise."
            ),
            "tests": (
                "assert is_even(2) is True\n"
                "assert is_even(3) is False\n"
                "assert is_even(0) is True"
            ),
            "reference_code": (
                "def is_even(n: int) -> bool:\n"
                "    return n % 2 == 0\n"
            ),
        },
        {
            "prompt": (
                "Write a Python function `def is_palindrome(s: str) -> bool:` "
                "that returns True if the string is a palindrome."
            ),
            "tests": (
                'assert is_palindrome("racecar") is True\n'
                'assert is_palindrome("hello") is False\n'
                'assert is_palindrome("") is True'
            ),
            "reference_code": (
                "def is_palindrome(s: str) -> bool:\n"
                "    return s == s[::-1]\n"
            ),
        },
        {
            "prompt": (
                "Write a Python function `def factorial(n: int) -> int:` "
                "that returns the factorial of a non-negative integer."
            ),
            "tests": (
                "assert factorial(0) == 1\n"
                "assert factorial(1) == 1\n"
                "assert factorial(5) == 120"
            ),
            "reference_code": (
                "def factorial(n: int) -> int:\n"
                "    result = 1\n"
                "    for value in range(2, n + 1):\n"
                "        result *= value\n"
                "    return result\n"
            ),
        },
        {
            "prompt": (
                "Write a Python function `def fibonacci(n: int) -> int:` "
                "that returns the nth Fibonacci number with fibonacci(0) == 0."
            ),
            "tests": (
                "assert fibonacci(0) == 0\n"
                "assert fibonacci(1) == 1\n"
                "assert fibonacci(7) == 13"
            ),
            "reference_code": (
                "def fibonacci(n: int) -> int:\n"
                "    a, b = 0, 1\n"
                "    for _ in range(n):\n"
                "        a, b = b, a + b\n"
                "    return a\n"
            ),
        },
        {
            "prompt": (
                "Write a Python function `def reverse_string(text: str) -> str:` "
                "that returns the reversed input string."
            ),
            "tests": (
                'assert reverse_string("abc") == "cba"\n'
                'assert reverse_string("NovaMind") == "dniMavoN"'
            ),
            "reference_code": (
                "def reverse_string(text: str) -> str:\n"
                "    return text[::-1]\n"
            ),
        },
        {
            "prompt": (
                "Write a Python function `def square(n: int) -> int:` "
                "that returns the square of the input integer."
            ),
            "tests": (
                "assert square(4) == 16\n"
                "assert square(-3) == 9"
            ),
            "reference_code": (
                "def square(n: int) -> int:\n"
                "    return n * n\n"
            ),
        },
        {
            "prompt": (
                "Write a Python function `def deduplicate(items: list[int]) -> list[int]:` "
                "that removes duplicates while preserving order."
            ),
            "tests": (
                "assert deduplicate([1, 2, 2, 3, 1]) == [1, 2, 3]\n"
                "assert deduplicate([]) == []"
            ),
            "reference_code": (
                "def deduplicate(items: list[int]) -> list[int]:\n"
                "    seen = set()\n"
                "    output = []\n"
                "    for item in items:\n"
                "        if item not in seen:\n"
                "            seen.add(item)\n"
                "            output.append(item)\n"
                "    return output\n"
            ),
        },
        {
            "prompt": (
                "Write a Python function `def find_max(values: list[int]) -> int:` "
                "that returns the maximum value in a non-empty list."
            ),
            "tests": (
                "assert find_max([1, 9, 3]) == 9\n"
                "assert find_max([-5, -2, -11]) == -2"
            ),
            "reference_code": (
                "def find_max(values: list[int]) -> int:\n"
                "    best = values[0]\n"
                "    for value in values[1:]:\n"
                "        if value > best:\n"
                "            best = value\n"
                "    return best\n"
            ),
        },
        {
            "prompt": (
                "Write a Python function `def sum_list(values: list[int]) -> int:` "
                "that returns the sum of all integers in the list."
            ),
            "tests": (
                "assert sum_list([1, 2, 3]) == 6\n"
                "assert sum_list([]) == 0"
            ),
            "reference_code": (
                "def sum_list(values: list[int]) -> int:\n"
                "    total = 0\n"
                "    for value in values:\n"
                "        total += value\n"
                "    return total\n"
            ),
        },
        {
            "prompt": (
                "Write a Python function `def count_vowels(text: str) -> int:` "
                "that counts lowercase and uppercase vowels in the string."
            ),
            "tests": (
                'assert count_vowels("hello") == 2\n'
                'assert count_vowels("AEIOU") == 5'
            ),
            "reference_code": (
                "def count_vowels(text: str) -> int:\n"
                "    vowels = set('aeiouAEIOU')\n"
                "    return sum(1 for char in text if char in vowels)\n"
            ),
        },
        {
            "prompt": (
                "Write a Python function `def first_char(text: str) -> str:` "
                "that returns the first character of a non-empty string."
            ),
            "tests": (
                'assert first_char("Nova") == "N"\n'
                'assert first_char("a") == "a"'
            ),
            "reference_code": (
                "def first_char(text: str) -> str:\n"
                "    return text[0]\n"
            ),
        },
        {
            "prompt": (
                "Write a Python function `def count_odds(values: list[int]) -> int:` "
                "that counts how many odd numbers are in the list."
            ),
            "tests": (
                "assert count_odds([1, 2, 3, 4, 5]) == 3\n"
                "assert count_odds([2, 4, 6]) == 0"
            ),
            "reference_code": (
                "def count_odds(values: list[int]) -> int:\n"
                "    return sum(1 for value in values if value % 2 == 1)\n"
            ),
        },
        {
            "prompt": (
                "Write a Python function `def clamp_zero(n: int) -> int:` "
                "that returns 0 if n is negative, otherwise returns n."
            ),
            "tests": (
                "assert clamp_zero(-5) == 0\n"
                "assert clamp_zero(7) == 7"
            ),
            "reference_code": (
                "def clamp_zero(n: int) -> int:\n"
                "    if n < 0:\n"
                "        return 0\n"
                "    return n\n"
            ),
        },
        {
            "prompt": (
                "Write a Python function `def list_length(values: list[int]) -> int:` "
                "that returns the number of elements in the list."
            ),
            "tests": (
                "assert list_length([1, 2, 3]) == 3\n"
                "assert list_length([]) == 0"
            ),
            "reference_code": (
                "def list_length(values: list[int]) -> int:\n"
                "    count = 0\n"
                "    for _ in values:\n"
                "        count += 1\n"
                "    return count\n"
            ),
        },
        {
            "prompt": (
                "Write a Python function `def all_positive(values: list[int]) -> bool:` "
                "that returns True only if every value in the list is positive."
            ),
            "tests": (
                "assert all_positive([1, 2, 3]) is True\n"
                "assert all_positive([1, -1, 3]) is False"
            ),
            "reference_code": (
                "def all_positive(values: list[int]) -> bool:\n"
                "    for value in values:\n"
                "        if value <= 0:\n"
                "            return False\n"
                "    return True\n"
            ),
        },
    ]


def _make_vault_node(name: str, value: int | None = None,
                     children: list | None = None,
                     fallback: dict | None = None) -> dict:
    node = {"name": name}
    if value is not None:
        node["value"] = value
    if children is not None:
        node["children"] = children
    if fallback is not None:
        node["fallback"] = fallback
    return node


def _make_deep_vault(depth: int, target: str, encrypted_value: int) -> dict:
    current = _make_vault_node(target, encrypted_value, children=[])
    for idx in reversed(range(depth)):
        current = _make_vault_node(f"layer_{idx}", children=[current])
    return current


def _vault_correct_block() -> str:
    return (
        "if depth > max_depth:\n"
        "    raise RecursionError('max depth exceeded')\n"
        "if isinstance(node, dict):\n"
        "    if node.get('name') == target:\n"
        "        return node['value'] ^ key\n"
        "    children = node.get('children', [])\n"
        "    for child in children:\n"
        "        result = _walk(child, depth + 1)\n"
        "        if result is not None:\n"
        "            return result\n"
        "    fallback = node.get('fallback')\n"
        "    if fallback is not None:\n"
        "        return _walk(fallback, depth + 1)\n"
        "    return None\n"
        "if isinstance(node, list):\n"
        "    for child in node:\n"
        "        result = _walk(child, depth + 1)\n"
        "        if result is not None:\n"
        "            return result\n"
        "    return None\n"
        "return None"
    )


def _vault_poisoned_block() -> str:
    return (
        "if depth > max_depth:\n"
        "    raise RecursionError('max depth exceeded')\n"
        "if isinstance(node, dict):\n"
        "    if node.get('name') == target:\n"
        "        return node['value'] ^ key\n"
        "    children = node.get('children', [])\n"
        "    for idx in range(len(children) + 1):\n"
        "        child = children[idx]\n"
        "        result = _walk(child, depth + 1)\n"
        "        if result is not None:\n"
        "            return result\n"
        "    fallback = node.get('fallback')\n"
        "    if fallback is not None:\n"
        "        return _walk(fallback, depth + 1)\n"
        "    return None\n"
        "if isinstance(node, list):\n"
        "    for idx in range(len(node) + 1):\n"
        "        child = node[idx]\n"
        "        result = _walk(child, depth + 1)\n"
        "        if result is not None:\n"
        "            return result\n"
        "    return None\n"
        "return None"
    )


def _vault_reference_code() -> str:
    return (
        "def solve_vault(vault: dict, target: str, key: int = 0, max_depth: int = 8):\n"
        "    def _walk(node, depth):\n"
        f"{_indent_block(_vault_correct_block(), spaces=8)}\n"
        "    return _walk(vault, 0)\n"
    )


def _indent_block(block: str, spaces: int = 4) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line if line.strip() else line for line in block.splitlines())


def _build_vault_tests(sample_idx: int) -> str:
    key = 11 + (sample_idx % 7)
    target = f"vault_{sample_idx}"
    secret = 90 + sample_idx
    encrypted = secret ^ key
    alt_secret = 140 + sample_idx
    alt_encrypted = alt_secret ^ key

    root_case = _make_vault_node(
        "root",
        children=[
            _make_vault_node("decoy", 13 ^ key, children=[]),
            _make_vault_node(target, encrypted, children=[]),
        ],
    )
    nested_case = _make_vault_node(
        "outer",
        children=[
            _make_vault_node("left", children=[]),
            _make_vault_node(
                "branch",
                children=[_make_vault_node(target, encrypted, children=[])],
            ),
        ],
    )
    fallback_case = _make_vault_node(
        "entry",
        children=[{"meta": "missing_name"}, _make_vault_node("noise", children=[])],
        fallback=_make_vault_node(target, alt_encrypted, children=[]),
    )
    list_case = [
        _make_vault_node("list_noise", children=[]),
        _make_vault_node("list_branch", children=[_make_vault_node(target, encrypted, children=[])]),
    ]
    deep_case = _make_deep_vault(5 + (sample_idx % 3), target, encrypted)

    blocks = [
        (
            f"root_case = {repr(root_case)}\n"
            f"assert solve_vault(root_case, '{target}', key={key}) == {secret}"
        ),
        (
            f"nested_case = {repr(nested_case)}\n"
            f"assert solve_vault(nested_case, '{target}', key={key}) == {secret}"
        ),
        (
            f"fallback_case = {repr(fallback_case)}\n"
            f"assert solve_vault(fallback_case, '{target}', key={key}) == {alt_secret}"
        ),
        (
            f"list_case = {repr(list_case)}\n"
            f"assert solve_vault(list_case, '{target}', key={key}) == {secret}"
        ),
        (
            f"deep_case = {repr(deep_case)}\n"
            "try:\n"
            f"    solve_vault(deep_case, '{target}', key={key}, max_depth=2)\n"
            "    raise AssertionError('expected RecursionError')\n"
            "except RecursionError:\n"
            "    pass"
        ),
    ]
    return "\n\n".join(blocks)


def get_secure_vault_dataset(num_tasks: int = 50) -> List[Dict[str, str]]:
    dataset: List[Dict[str, str]] = []
    poisoned_block = _vault_poisoned_block()
    correct_block = _vault_correct_block()
    reference_code = _vault_reference_code()

    for idx in range(num_tasks):
        dataset.append(
            {
                "task_name": f"vault_{idx}",
                "prompt": (
                    "Write a Python function "
                    "`def solve_vault(vault: dict, target: str, key: int = 0, max_depth: int = 8):` "
                    "that recursively searches a nested JSON-like vault made of dicts and lists. "
                    "When it finds a node whose `name` matches `target`, it must decrypt and return "
                    "`value ^ key`. It must handle missing keys safely and raise `RecursionError` "
                    "if the traversal exceeds `max_depth`."
                ),
                "tests": _build_vault_tests(idx),
                "reference_code": reference_code,
                "correct_block": correct_block,
                "poisoned_block": poisoned_block,
            }
        )
    return dataset
