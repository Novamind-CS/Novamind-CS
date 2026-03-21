"""
NovaMind inference script.

Supports:
- standard generation (greedy / top-p / top-k)
- CATTS adaptive inference with internal System 1/2 routing
- batch benchmarking
- LoRA checkpoint loading

Examples:

    python inference.py --checkpoint ./checkpoints/checkpoint_step5000.pt \
                        --size 7b --interactive

    python inference.py --checkpoint ./ckpt.pt --size 7b \
                        --prompt "Explain quantum entanglement."

    python inference.py --checkpoint ./ckpt.pt --size 7b \
                        --bench ./eval_prompts.jsonl
"""

import sys
import json
import time
import argparse
import tempfile
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def _tokenize_to_tensor(tokenizer, text: str, device: str) -> torch.Tensor:
    if hasattr(tokenizer, "__call__"):
        tokens = tokenizer(text, return_tensors="pt")
        input_ids = tokens["input_ids"] if isinstance(tokens, dict) else tokens.input_ids
        return input_ids.to(device)
    ids = tokenizer.encode(text)
    return torch.tensor([ids], device=device)


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


def load_model(checkpoint_path: str, size: str, device: str = "auto"):
    """Load either a LoRA checkpoint or a full model checkpoint."""
    from novamind.config import NovaMindConfig
    from novamind.model import NovaMind
    from novamind.training.lora import inject_lora
    from novamind.core.device_manager import get_device, get_dtype, get_hardware_banner

    device_obj = get_device(device)
    dtype = get_dtype()

    print(get_hardware_banner(device_obj))
    print(f"[Inference] Loading NovaMind-{size.upper()} on {device_obj.type}...")

    cfg = NovaMindConfig.from_size(size)
    model = NovaMind(cfg)

    if checkpoint_path and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # LoRA checkpoint
        if "lora_state_dict" in ckpt:
            print("[Inference] Detected a LoRA checkpoint, injecting adapters...")
            model = inject_lora(model, target_modules=cfg.lora_target_modules,
                                rank=cfg.lora_rank, alpha=cfg.lora_alpha)
            missing, unexpected = model.load_state_dict(
                ckpt["lora_state_dict"], strict=False
            )
            print(f"[Inference] LoRA load complete (missing={len(missing)}, unexpected={len(unexpected)})")


        elif "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            print("[Inference] Full checkpoint loaded")

        else:
            print("[Inference] Warning: unknown checkpoint format, using random weights for testing")
    else:
        print("[Inference] No checkpoint provided, using random initialization for smoke testing")

    model = model.to(device_obj)
    model = model.to(dtype)
    model.eval()

    n_params = model.num_parameters()
    print(f"[Inference] Model parameters: {n_params/1e9:.2f}B")
    if torch.cuda.is_available():
        vram = torch.cuda.memory_allocated() / 1e9
        print(f"[Inference] Current VRAM: {vram:.2f}GB")

    return model, cfg, device_obj.type


def load_tokenizer(tokenizer_path: str):
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(tokenizer_path)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return tok
    except Exception as e:
        print(f"[Warning] Failed to load tokenizer: {e}")
        print("[Warning] Falling back to the character-level tokenizer for testing")
        return CharTokenizer()


class CharTokenizer:
    """Character-level fallback tokenizer for environments without transformers."""

    def __init__(self):
        self.eos_token_id = 0
        self.pad_token_id = 0
        vocab = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                     "0123456789 \n\t.,!?;:\"'()[]{}/-_@#$%^&*+=<>|\\~`")
        self.char2id = {c: i+1 for i, c in enumerate(vocab)}
        self.id2char = {v: k for k, v in self.char2id.items()}

    def encode(self, text: str, **kwargs):
        return [self.char2id.get(c, 1) for c in text]

    def decode(self, ids, **kwargs):
        return "".join(self.id2char.get(i, "?") for i in ids)

    def __call__(self, text, return_tensors=None, **kwargs):
        ids = self.encode(text)
        if return_tensors == "pt":
            import torch
            return type("Tok", (), {"input_ids": torch.tensor([ids])})()
        return {"input_ids": ids}


@torch.no_grad()
def generate(model, tokenizer, prompt: str,
             max_new_tokens: int = 256,
             temperature: float = 0.8,
             top_p: float = 0.9,
             top_k: int = 50,
             device: str = "cuda",
             use_catts: bool = False,
             use_met: bool = False,
             met_entropy_threshold: float = 3.5,
             met_caution_window: int = 5,
             system2_callback=None) -> str:
    """
    Generate text.

    When `use_catts=True`, adaptive compute scheduling is enabled so easy cases
    can exit early and difficult cases can spend more reasoning budget.
    """
    model.eval()

    # Tokenize
    input_ids = _tokenize_to_tensor(tokenizer, prompt, device)

    if use_catts:
        from novamind.core.catts import AdaptiveNovaMindWrapper
        wrapper = AdaptiveNovaMindWrapper(model, model.config)
        print("[CATTS] Adaptive generation enabled...")
        t0 = time.time()
        catts_out = wrapper.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            use_catts=True,
        )
        generated = catts_out["tokens"]
        elapsed = time.time() - t0
        stats = catts_out["catts_stats"]
        print(f"[CATTS] Routing distribution fast={stats['fast']:.2f} "
              f"normal={stats['normal']:.2f} deep={stats['deep']:.2f}")
    else:
        t0 = time.time()
        if use_met:
            from novamind.core.met_controller import MetStateTracker

            tracker = MetStateTracker(
                entropy_threshold=met_entropy_threshold,
                caution_window=max(1, met_caution_window),
            )
            current_ids = input_ids
            generated_ids = []
            for _ in range(max_new_tokens):
                outputs = model(input_ids=current_ids, inference_mode=True)
                logits = outputs["logits"][:, -1, :]
                met_decision = tracker.observe(logits)
                if met_decision["trigger_system2"] and system2_callback is not None:
                    if not met_decision["forced_by_inertia"]:
                        print(
                            f"[MET] High Entropy Detected (H={met_decision['entropy']:.3f}). "
                            "Engaging MCTS Reasoning..."
                        )
                    return system2_callback()
                next_token = _sample_next_token(
                    logits=logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                generated_ids.append(int(next_token.item()))
                current_ids = torch.cat([current_ids, next_token], dim=1)
            generated = current_ids
        else:
            generated = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
        elapsed = time.time() - t0


    new_ids = generated[0][input_ids.shape[1]:].tolist()
    if hasattr(tokenizer, "decode"):
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
    else:
        text = tokenizer.decode(new_ids)

    n_new = len(new_ids)
    tps = n_new / elapsed if elapsed > 0 else 0
    print(f"[Generation] {n_new} tokens in {elapsed:.2f}s ({tps:.1f} tok/s)")

    return text


def interactive_loop(model, tokenizer, device: str, args):
    """Interactive chat loop."""
    print("\n" + "="*60)
    print("NovaMind interactive inference  (type 'quit' to exit, 'stats' for counters)")
    print("="*60 + "\n")

    history = []
    total_tokens = 0
    total_time = 0.0

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "stats":
            if total_time > 0:
                print(f"\nStats: total_tokens={total_tokens}, "
                      f"avg_speed={total_tokens/total_time:.1f} tok/s\n")
            continue


        prompt = f"Human: {user_input}\nAssistant:"

        t0 = time.time()
        response = generate(
            model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
            use_catts=args.catts,
        )
        elapsed = time.time() - t0


        if "Human:" in response:
            response = response[:response.index("Human:")].strip()

        total_time += elapsed
        total_tokens += len(response.split())

        print(f"\nNovaMind: {response}\n")


def run_code_mcts(model, tokenizer, prompt: str, device: str, args):
    from novamind.core.code_mcts import (
        CodeMCTSReasoner,
        PythonSandbox,
        inspect_system1_entropy,
        make_sampling_proposal_fn,
    )
    from novamind.core.met_controller import METGate, MetStateTracker

    tests = []
    if args.tests:
        with open(args.tests, "r", encoding="utf-8") as f:
            tests = [line.strip() for line in f if line.strip()]

    input_ids = _tokenize_to_tensor(tokenizer, prompt, device)
    met_decision = None
    if args.use_met:
        met_decision = inspect_system1_entropy(
            model,
            input_ids=input_ids,
            entropy_threshold=args.met_entropy_threshold,
        )
        if not met_decision["trigger_system2"]:
            print(f"[MET] Low Entropy Detected (H={met_decision['entropy']:.3f}). Staying in System 1.")
            return generate(
                model,
                tokenizer,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                device=device,
                use_catts=False,
                use_met=False,
            )

    proposal_fn = make_sampling_proposal_fn(
        model,
        tokenizer,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        met_gate=None if not args.use_met else METGate(args.met_entropy_threshold),
        met_tracker=None if not args.use_met else MetStateTracker(
            entropy_threshold=args.met_entropy_threshold,
            caution_window=args.met_caution_window,
        ),
    )
    reasoner = CodeMCTSReasoner(
        proposal_fn=proposal_fn,
        sandbox=PythonSandbox(timeout_s=args.sandbox_timeout),
        num_options=args.mcts_options,
        max_rollouts=args.mcts_rollouts,
    )

    t0 = time.time()
    result = reasoner.run(prompt, tests=tests)
    elapsed = time.time() - t0

    print(f"[MCTS] rollouts={result['rollouts']} reward={result['reward']:.1f} "
          f"status={result['result'].status if result['result'] else 'unknown'} "
          f"time={elapsed:.2f}s")
    if met_decision is not None:
        print(f"[MET] prompt_entropy={met_decision['entropy']:.3f}")
    if result["result"] is not None:
        print(f"[MCTS] tests={result['result'].passed_tests}/{result['result'].total_tests}")
        if result["result"].error:
            print(f"[MCTS] error:\n{result['result'].error}")
    return result["code"]


def run_bench(model, tokenizer, bench_path: str, device: str, args):
    """Run a batch benchmark."""
    print(f"\n[Bench] Loading {bench_path}...")
    prompts = []
    with open(bench_path) as f:
        for line in f:
            line = line.strip()
            if line:
                obj = json.loads(line) if line.startswith("{") else {"prompt": line}
                prompts.append(obj)

    print(f"[Bench] Loaded {len(prompts)} prompts\n")
    results = []

    for i, item in enumerate(prompts):
        prompt = item.get("prompt", item.get("text", ""))
        expected = item.get("answer", item.get("label", ""))

        t0 = time.time()
        response = generate(model, tokenizer, prompt,
                            max_new_tokens=args.max_new_tokens,
                            device=device,
                            use_catts=args.catts)
        elapsed = time.time() - t0

        result = {
            "id": i,
            "prompt": prompt,
            "response": response,
            "expected": expected,
            "time_s": round(elapsed, 3),
        }
        results.append(result)
        print(f"[{i+1}/{len(prompts)}] {prompt[:40]}... → {response[:60]}...")


    out_path = bench_path.replace(".jsonl", "_results.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    avg_time = sum(r["time_s"] for r in results) / len(results)
    print(f"\n[Bench] Done. Average latency: {avg_time:.2f}s/prompt")
    print(f"[Bench] Results saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="NovaMind inference")
    parser.add_argument("--checkpoint", type=str, default="",
                        help="Path to a checkpoint file (leave empty for random-weight smoke tests)")
    parser.add_argument("--size", choices=["3b", "7b", "14b"], default="7b")
    parser.add_argument("--tokenizer", type=str,
                        default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--device", type=str, default="auto")


    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--catts", action="store_true",
                        help="Enable CATTS adaptive compute routing")
    parser.add_argument("--use_met", action="store_true",
                        help="Enable MET entropy gating so low-entropy states skip System 2")
    parser.add_argument("--met_entropy_threshold", type=float, default=3.5)
    parser.add_argument("--met_caution_window", type=int, default=5)


    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--bench", type=str, default="",
                        help="Path to a JSONL file for batch benchmarking")
    parser.add_argument("--mcts_code", action="store_true",
                        help="Enable code-level MCTS with Python sandbox self-verification")
    parser.add_argument("--tests", type=str, default="",
                        help="Path to code-MCTS tests, one Python assertion per line")
    parser.add_argument("--sandbox_timeout", type=float, default=2.0)
    parser.add_argument("--mcts_options", type=int, default=3)
    parser.add_argument("--mcts_rollouts", type=int, default=24)

    args = parser.parse_args()


    model, cfg, device = load_model(args.checkpoint, args.size, args.device)
    tokenizer = load_tokenizer(args.tokenizer)

    if args.interactive:
        interactive_loop(model, tokenizer, device, args)

    elif args.prompt:
        print(f"\nPrompt: {args.prompt}\n")
        if args.mcts_code:
            response = run_code_mcts(model, tokenizer, args.prompt, device, args)
        else:
            response = generate(
                model, tokenizer, args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                device=device,
                use_catts=args.catts,
                use_met=args.use_met,
                met_entropy_threshold=args.met_entropy_threshold,
                met_caution_window=args.met_caution_window,
                system2_callback=(
                    lambda: run_code_mcts(model, tokenizer, args.prompt, device, args)
                    if args.use_met and args.tests
                    else None
                ),
            )
        print(f"Response: {response}")

    elif args.bench:
        run_bench(model, tokenizer, args.bench, device, args)

    else:

        print("\n[Smoke Test] No prompt provided, running the default check...")
        test_prompts = [
            "The capital of France is",
            "def fibonacci(n):",
            "The core principle of quantum computing is",
        ]
        for p in test_prompts:
            print(f"\nPrompt: {p}")
            r = generate(model, tokenizer, p,
                         max_new_tokens=50, top_k=args.top_k, device=device,
                         use_met=args.use_met,
                         met_entropy_threshold=args.met_entropy_threshold,
                         met_caution_window=args.met_caution_window)
            print(f"Output: {r}")


if __name__ == "__main__":
    main()
