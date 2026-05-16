"""
End-to-end RLHF pipeline: SFT → Reward Model → PPO.

Run this script to execute all three training stages sequentially
with the default configuration.

Usage:
    python scripts/train_all.py
    python scripts/train_all.py --ppo_steps 100
    python scripts/train_all.py --model gpt2-medium --ppo_steps 200
"""

import argparse
import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from configs import DEFAULT_CONFIG, ProjectConfig
from evaluation import compare_sft_vs_rlhf, reward_hacking_demo, plot_training_curves
from training import run_sft, run_rm_training, run_ppo_training
from utils import run_sanity_check, setup_logger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RLHF Financial Hedging — full pipeline")
    p.add_argument("--model", default="gpt2", help="HuggingFace model name")
    p.add_argument("--sft_epochs", type=int, default=5)
    p.add_argument("--rm_epochs", type=int, default=10)
    p.add_argument("--ppo_steps", type=int, default=30,
                   help="PPO steps (use 100-200 for real training)")
    p.add_argument("--no_plots", action="store_true", help="Skip matplotlib plots")
    return p.parse_args()


def main():
    args = parse_args()
    logger = setup_logger("rlhf.main", level="INFO")

    # ── Apply CLI overrides to config ─────────────────────────────────────────
    cfg = DEFAULT_CONFIG
    cfg.model.model_name = args.model
    cfg.sft.num_epochs = args.sft_epochs
    cfg.reward_model.num_epochs = args.rm_epochs
    cfg.ppo.num_steps = args.ppo_steps
    cfg.paths.make_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} | Model: {cfg.model.model_name}")

    # ── Stage 0: Sanity check preference signal ───────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 0 — Preference Signal Sanity Check")
    logger.info("=" * 60)
    run_sanity_check()

    # ── Stage 1: SFT ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 1 — Supervised Fine-Tuning")
    logger.info("=" * 60)
    sft_model, tokenizer = run_sft(cfg)
    sft_model.to(device)

    # ── Stage 2: Reward Model ─────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2 — Reward Model Training")
    logger.info("=" * 60)
    reward_model, rm_losses = run_rm_training(sft_model, tokenizer, cfg, device)

    # ── Stage 3: PPO ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 3 — PPO Fine-Tuning")
    logger.info("=" * 60)
    policy_model, training_log = run_ppo_training(
        sft_model, reward_model, tokenizer, cfg, device
    )

    # ── Evaluation ────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)
    compare_sft_vs_rlhf(sft_model, policy_model, tokenizer, device)
    reward_hacking_demo(init_kl_coef=cfg.ppo.init_kl_coef)

    if not args.no_plots:
        plot_training_curves(
            training_log,
            save_path=cfg.paths.plots_dir / "rlhf_training_curves.png",
        )

    logger.info("\nPipeline complete 🎉")


if __name__ == "__main__":
    main()
