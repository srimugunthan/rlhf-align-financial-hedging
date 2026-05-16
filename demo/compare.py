"""
Option C — Before/After Generation Comparison

Loads the saved SFT and PPO checkpoints and prints a side-by-side table of
completions with hedge scores, illustrating how PPO shifts the model toward
hedged, cautious financial language.

Requires models to be trained first:
    uv run python scripts/train_all.py

Usage:
    # Use default eval prompts
    uv run python demo/compare.py

    # Custom prompt
    uv run python demo/compare.py --prompt "The Federal Reserve"

    # Multiple custom prompts, 3 completions each
    uv run python demo/compare.py \\
        --prompt "The stock market" \\
        --prompt "Bond yields" \\
        --n 3

    # Override checkpoint paths
    uv run python demo/compare.py \\
        --sft_dir outputs/sft_checkpoint \\
        --ppo_dir outputs/ppo_checkpoint
"""

import argparse
import sys
import textwrap
from pathlib import Path

# Allow imports from project root regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from trl import AutoModelForCausalLMWithValueHead

from data.corpus import EVAL_PROMPTS
from models.policy_model import generate_completion
from utils.reward_utils import compute_hedge_score

# ── Default checkpoint paths ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SFT_DIR = PROJECT_ROOT / "outputs" / "sft_checkpoint"
DEFAULT_PPO_DIR = PROJECT_ROOT / "outputs" / "ppo_checkpoint"

COL_WIDTH = 72
DIVIDER = "─" * (COL_WIDTH * 2 + 7)


def load_tokenizer(model_dir: Path) -> GPT2Tokenizer:
    tokenizer = GPT2Tokenizer.from_pretrained(str(model_dir))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_sft_model(model_dir: Path, device: torch.device) -> GPT2LMHeadModel:
    model = GPT2LMHeadModel.from_pretrained(str(model_dir))
    model.eval()
    return model.to(device)


def load_ppo_model(
    model_dir: Path, device: torch.device
) -> AutoModelForCausalLMWithValueHead:
    model = AutoModelForCausalLMWithValueHead.from_pretrained(str(model_dir))
    model.pretrained_model.eval()
    return model.to(device)


def generate_from_sft(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 60,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def _wrap(text: str, width: int) -> list[str]:
    """Wrap text to fixed column width, returning lines."""
    return textwrap.wrap(text, width=width) or [""]


def print_comparison(
    prompt: str,
    sft_texts: list[str],
    ppo_texts: list[str],
) -> None:
    """Print a formatted side-by-side table for a single prompt."""
    print(f"\nPrompt: \"{prompt}\"")
    print(DIVIDER)
    header = f"{'SFT (before RLHF)':<{COL_WIDTH}} | {'PPO (after RLHF)':<{COL_WIDTH}}"
    print(header)
    print(DIVIDER)

    for idx, (sft, ppo) in enumerate(zip(sft_texts, ppo_texts), start=1):
        sft_score = compute_hedge_score(sft)
        ppo_score = compute_hedge_score(ppo)
        delta = ppo_score - sft_score
        flag = "✅" if delta > 0 else ("➖" if delta == 0 else "⚠️")

        sft_lines = _wrap(sft, COL_WIDTH)
        ppo_lines = _wrap(ppo, COL_WIDTH)
        max_lines = max(len(sft_lines), len(ppo_lines))

        # Pad shorter column
        sft_lines += [""] * (max_lines - len(sft_lines))
        ppo_lines += [""] * (max_lines - len(ppo_lines))

        if len(sft_texts) > 1:
            print(f"  [Run {idx}]")

        for sft_line, ppo_line in zip(sft_lines, ppo_lines):
            print(f"{sft_line:<{COL_WIDTH}} | {ppo_line:<{COL_WIDTH}}")

        delta_str = f"{delta:+.3f}"
        print(
            f"  Hedge score: {sft_score:.3f}{' ' * (COL_WIDTH - 20)}"
            f"   Hedge score: {ppo_score:.3f}  ({delta_str}) {flag}"
        )
        if idx < len(sft_texts):
            print()

    print(DIVIDER)


def run_comparison(
    prompts: list[str],
    sft_model: GPT2LMHeadModel,
    ppo_model: AutoModelForCausalLMWithValueHead,
    tokenizer: GPT2Tokenizer,
    device: torch.device,
    n: int = 1,
) -> None:
    """Generate and display comparisons for all prompts."""
    all_deltas = []

    for prompt in prompts:
        sft_texts = [
            generate_from_sft(sft_model, tokenizer, prompt, device) for _ in range(n)
        ]
        ppo_texts = [
            generate_completion(ppo_model, tokenizer, prompt, device) for _ in range(n)
        ]
        print_comparison(prompt, sft_texts, ppo_texts)

        for sft, ppo in zip(sft_texts, ppo_texts):
            all_deltas.append(compute_hedge_score(ppo) - compute_hedge_score(sft))

    avg = sum(all_deltas) / len(all_deltas)
    direction = "improvement" if avg > 0 else "regression"
    print(f"\nAverage hedge score delta across {len(all_deltas)} comparison(s): {avg:+.3f} ({direction})")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Side-by-side SFT vs PPO completion comparison with hedge scores"
    )
    p.add_argument(
        "--prompt",
        dest="prompts",
        metavar="TEXT",
        action="append",
        help="Financial prompt to complete (can be repeated). Defaults to EVAL_PROMPTS.",
    )
    p.add_argument(
        "--n",
        type=int,
        default=1,
        metavar="N",
        help="Number of independent completions per prompt (default: 1)",
    )
    p.add_argument(
        "--sft_dir",
        type=Path,
        default=DEFAULT_SFT_DIR,
        help=f"Path to SFT checkpoint (default: {DEFAULT_SFT_DIR})",
    )
    p.add_argument(
        "--ppo_dir",
        type=Path,
        default=DEFAULT_PPO_DIR,
        help=f"Path to PPO checkpoint (default: {DEFAULT_PPO_DIR})",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    prompts = args.prompts or EVAL_PROMPTS

    # Validate checkpoints exist
    for label, path in [("SFT", args.sft_dir), ("PPO", args.ppo_dir)]:
        if not path.exists():
            print(
                f"[ERROR] {label} checkpoint not found at: {path}\n"
                "Run the full pipeline first:\n"
                "    uv run python scripts/train_all.py"
            )
            sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading SFT model from  : {args.sft_dir}")
    print(f"Loading PPO model from  : {args.ppo_dir}")

    tokenizer = load_tokenizer(args.sft_dir)
    sft_model = load_sft_model(args.sft_dir, device)
    ppo_model = load_ppo_model(args.ppo_dir, device)

    print(f"\nComparing {len(prompts)} prompt(s), {args.n} completion(s) each.\n")
    run_comparison(prompts, sft_model, ppo_model, tokenizer, device, n=args.n)


if __name__ == "__main__":
    main()
