"""
Evaluation utilities for RLHF alignment quality.

Provides:
  - compare_sft_vs_rlhf()   : side-by-side text quality comparison
  - reward_hacking_demo()   : illustrates why KL penalty matters
  - plot_training_curves()  : visualise reward / KL / hedge score over PPO steps
"""

import logging
import textwrap
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from trl import AutoModelForCausalLMWithValueHead

from data import EVAL_PROMPTS, REWARD_HACKING_EXAMPLES
from models import generate_completion
from utils import compute_hedge_score

logger = logging.getLogger(__name__)


# ── Text generation helper for plain GPT2LMHeadModel ─────────────────────────

def generate_from_lm(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 50,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    model.eval()
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


# ── SFT vs RLHF Comparison ────────────────────────────────────────────────────

def compare_sft_vs_rlhf(
    sft_model: GPT2LMHeadModel,
    policy_model: AutoModelForCausalLMWithValueHead,
    tokenizer: GPT2Tokenizer,
    device: torch.device,
    prompts: Optional[List[str]] = None,
) -> List[dict]:
    """
    Generate completions from both SFT and RLHF models for each prompt,
    score them, and print a side-by-side comparison table.

    Args:
        sft_model:    Base SFT model (pre-PPO).
        policy_model: PPO-trained policy model.
        tokenizer:    GPT-2 tokenizer.
        device:       Compute device.
        prompts:      Prompts to evaluate (defaults to EVAL_PROMPTS).

    Returns:
        List of result dicts with keys:
            prompt, sft_text, rlhf_text, sft_score, rlhf_score, improvement
    """
    if prompts is None:
        prompts = EVAL_PROMPTS

    print("\nRLHF EVALUATION: SFT vs PPO-trained Policy")
    print("=" * 70)

    results = []
    for prompt in prompts:
        sft_text = generate_from_lm(sft_model, tokenizer, prompt, device)
        rlhf_text = generate_completion(policy_model, tokenizer, prompt, device)

        sft_score = compute_hedge_score(sft_text)
        rlhf_score = compute_hedge_score(rlhf_text)
        improvement = rlhf_score - sft_score

        results.append({
            "prompt": prompt,
            "sft_text": sft_text,
            "rlhf_text": rlhf_text,
            "sft_score": sft_score,
            "rlhf_score": rlhf_score,
            "improvement": improvement,
        })

        delta_str = f"+{improvement:.3f}" if improvement > 0 else f"{improvement:.3f}"
        flag = "✅" if improvement > 0 else "⚠️"
        print(f"\nPrompt: '{prompt}'")
        print(f"  SFT  [{sft_score:.3f}]: {textwrap.shorten(sft_text, 80)}")
        print(f"  RLHF [{rlhf_score:.3f}]: {textwrap.shorten(rlhf_text, 80)}")
        print(f"  Change: {delta_str} {flag}")

    avg = np.mean([r["improvement"] for r in results])
    print("\n" + "=" * 70)
    print(f"Average hedge score improvement: {avg:+.3f}")
    return results


# ── Reward Hacking Demo ───────────────────────────────────────────────────────

def reward_hacking_demo(init_kl_coef: float = 0.2) -> None:
    """
    Print a table showing how the heuristic reward can be gamed by
    hedge-word stuffing, and explain why the KL penalty prevents this.

    Args:
        init_kl_coef: The β value used in training (for display only).
    """
    print("\nReward Hacking Illustration")
    print("=" * 65)
    print("Shows why KL penalty matters — heuristic score can be gamed")
    print()
    print(f"{'Type':<20} {'Heuristic':>9}  {'Text (truncated)'}")
    print("-" * 65)

    for text, label in REWARD_HACKING_EXAMPLES:
        score = compute_hedge_score(text)
        truncated = text[:35] + "..." if len(text) > 35 else text
        print(f"{label:<20} {score:>9.3f}  {truncated}")

    print()
    print("⚠️  Without KL penalty, policy learns hedge-word stuffing.")
    print("    With KL penalty, incoherent text is penalised:")
    print()
    print("    R_total = r_RM(text) - β × KL(policy || SFT)")
    print(f"    β = {init_kl_coef} → strong anchor to coherent SFT distribution")


# ── Training Curve Plots ──────────────────────────────────────────────────────

def plot_training_curves(
    training_log: List[dict],
    save_path: Optional[Path] = None,
) -> None:
    """
    Plot PPO training metrics: reward, KL divergence, hedge score.

    Args:
        training_log: List of step dicts from run_ppo_training().
        save_path:    Optional path to save the figure (PNG).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available — skipping plot.")
        return

    if not training_log:
        logger.warning("Empty training log — nothing to plot.")
        return

    steps = [d["step"] for d in training_log]
    rewards = [d["mean_reward"] for d in training_log]
    kls = [d["kl_divergence"] for d in training_log]
    hedges = [d["hedge_score"] for d in training_log]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(
        "RLHF Training: Aligning GPT-2 for Hedged Financial Language",
        fontsize=13,
        fontweight="bold",
    )

    axes[0].plot(steps, rewards, color="#2196F3", linewidth=2)
    axes[0].set_title("Mean Reward (RM Score)")
    axes[0].set_xlabel("PPO Step")
    axes[0].set_ylabel("Reward")
    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(steps, kls, color="#FF5722", linewidth=2)
    axes[1].set_title("KL Divergence (policy vs SFT)")
    axes[1].set_xlabel("PPO Step")
    axes[1].set_ylabel("KL")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(steps, hedges, color="#4CAF50", linewidth=2)
    axes[2].set_title("Hedge Score (heuristic)")
    axes[2].set_xlabel("PPO Step")
    axes[2].set_ylabel("Score [0, 1]")
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        logger.info(f"Training curves saved to {save_path}")

    plt.show()
