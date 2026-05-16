"""
Stage 2 — Reward Model Training.

Trains a GPT-2 classification head on pairwise (preferred, dispreferred)
financial text pairs using the Bradley-Terry loss.

The reward model learns to assign higher scalar scores to hedged language
than to speculative/overconfident language.

Usage (standalone):
    python -m training.rm_trainer
"""

import logging

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2ForSequenceClassification

from configs import ProjectConfig, DEFAULT_CONFIG
from data import PAIRWISE_DATA
from models import build_reward_model, train_reward_model, get_reward
from utils import setup_logger, compute_hedge_score

logger = logging.getLogger(__name__)

# Texts used to print a quick RM evaluation after training
_EVAL_TEXTS = [
    "The stock may decline if economic conditions worsen.",
    "Based on data, earnings could potentially miss estimates.",
    "This stock will definitely make you rich with guaranteed gains!",
    "BUY NOW! Skyrocket to the moon! 100% profit guaranteed!",
    "The company appears to face headwinds from rising costs.",
    "Massive unstoppable gains are absolutely certain this year!",
]


def run_rm_training(
    sft_model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    cfg: ProjectConfig = DEFAULT_CONFIG,
    device: torch.device = None,
) -> tuple[GPT2ForSequenceClassification, list]:
    """
    Execute the Reward Model training stage end-to-end.

    Steps:
        1. Build RM (GPT-2 + scalar head, backbone from SFT weights)
        2. Train with Bradley-Terry loss on PAIRWISE_DATA
        3. Print evaluation table
        4. Optionally save checkpoint

    Args:
        sft_model:  Trained SFT model for backbone weight transfer.
        tokenizer:  GPT-2 tokenizer.
        cfg:        Master project config.
        device:     Compute device (auto-detected if None).

    Returns:
        (reward_model, epoch_losses)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    setup_logger("rlhf.rm", level=cfg.log_level)
    cfg.paths.make_dirs()

    logger.info(f"Building reward model on {device}...")
    reward_model = build_reward_model(cfg.model.model_name, sft_model, device)

    # Verify heuristic-label agreement before training
    correct = sum(
        1 for w, l in PAIRWISE_DATA
        if compute_hedge_score(w) > compute_hedge_score(l)
    )
    logger.info(
        f"Heuristic agrees with hand labels: {correct}/{len(PAIRWISE_DATA)} "
        f"({100*correct/len(PAIRWISE_DATA):.0f}%)"
    )

    logger.info(f"Training reward model ({len(PAIRWISE_DATA)} pairs, "
                f"{cfg.reward_model.num_epochs} epochs)...")
    losses = train_reward_model(
        model=reward_model,
        pairs=PAIRWISE_DATA,
        tokenizer=tokenizer,
        device=device,
        num_epochs=cfg.reward_model.num_epochs,
        lr=cfg.reward_model.learning_rate,
        batch_size=cfg.reward_model.batch_size,
        weight_decay=cfg.reward_model.weight_decay,
        grad_clip=cfg.reward_model.grad_clip,
        max_length=cfg.model.max_seq_length,
    )

    logger.info("Reward Model Training complete ✅")

    # Quick evaluation table
    logger.info("\nReward Model Evaluation:")
    logger.info(f"{'Text (truncated)':<45} {'RM Score':>8} {'Heuristic':>9}")
    logger.info("-" * 65)
    for text in _EVAL_TEXTS:
        rm_score = get_reward(reward_model, tokenizer, text, device)
        heuristic = compute_hedge_score(text)
        truncated = text[:43] + ".." if len(text) > 45 else text
        logger.info(f"{truncated:<45} {rm_score:>8.3f} {heuristic:>9.3f}")

    return reward_model, losses


if __name__ == "__main__":
    from training.sft_trainer import run_sft
    sft_model, tokenizer = run_sft()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sft_model.to(device)
    run_rm_training(sft_model, tokenizer, device=device)
