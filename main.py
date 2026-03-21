"""
Micro-CS showcase dashboard entry point.

This file is intentionally lightweight: it gives GitHub visitors a single,
friendly command surface for validation, benchmarking, and stress testing.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_command(cmd: list[str]) -> int:
    print(f"[main] Running: {' '.join(cmd)}")
    completed = subprocess.run(cmd, cwd=ROOT)
    return int(completed.returncode)


def print_showcase_banner() -> None:
    print("=" * 72)
    print("Micro-CS Showcase Dashboard")
    print("=" * 72)
    print("Mission: Democratizing high-order reasoning on consumer GPUs.")
    print()
    print("Signature innovations:")
    print("  1. LOD-Compute        low-fi search, high-fi replay")
    print("  2. AST-Rollback       syntax-grounded fault localization")
    print("  3. ISFS               incremental symbol flow sentinel")
    print("  4. SGA                surgical gradient attribution")
    print("  5. MET                entropy-triggered dual-process routing")
    print()
    print("Recommended commands:")
    print("  python3 main.py --mode tests")
    print("  python3 main.py --mode showcase")
    print("  python3 main.py --mode stress")
    print("=" * 72)


def run_tests() -> int:
    return run_command([sys.executable, "test_novamind.py"])


def run_showcase(args: argparse.Namespace) -> int:
    print_showcase_banner()
    cmd = [
        sys.executable,
        "train.py",
        "--vault_stress_test",
        "--vault_tasks",
        str(args.vault_tasks),
        "--vault_candidate_paths",
        str(args.vault_candidate_paths),
        "--vault_min_paths",
        str(args.vault_min_paths),
        "--mcts_max_new_tokens",
        str(args.mcts_max_new_tokens),
        "--use_met",
        "--met_entropy_threshold",
        str(args.met_entropy_threshold),
        "--met_caution_window",
        str(args.met_caution_window),
    ]
    return run_command(cmd)


def run_stress(args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        "train.py",
        "--vault_stress_test",
        "--vault_tasks",
        str(args.vault_tasks),
        "--vault_candidate_paths",
        str(args.vault_candidate_paths),
        "--vault_min_paths",
        str(args.vault_min_paths),
        "--mcts_max_new_tokens",
        str(args.mcts_max_new_tokens),
    ]
    if args.use_met:
        cmd.extend(
            [
                "--use_met",
                "--met_entropy_threshold",
                str(args.met_entropy_threshold),
                "--met_caution_window",
                str(args.met_caution_window),
            ]
        )
    return run_command(cmd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Showcase dashboard for the Micro-CS reasoning engine.",
    )
    parser.add_argument(
        "--mode",
        choices=("showcase", "stress", "tests"),
        default="showcase",
        help="Which dashboard flow to run.",
    )
    parser.add_argument("--vault_tasks", type=int, default=1)
    parser.add_argument("--vault_candidate_paths", type=int, default=4)
    parser.add_argument("--vault_min_paths", type=int, default=3)
    parser.add_argument("--mcts_max_new_tokens", type=int, default=8)
    parser.add_argument("--use_met", action="store_true", default=False)
    parser.add_argument("--met_entropy_threshold", type=float, default=3.5)
    parser.add_argument("--met_caution_window", type=int, default=5)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "tests":
        return run_tests()
    if args.mode == "stress":
        return run_stress(args)
    return run_showcase(args)


if __name__ == "__main__":
    raise SystemExit(main())
