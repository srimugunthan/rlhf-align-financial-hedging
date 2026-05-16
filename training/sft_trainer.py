"""
Stage 1 — Supervised Fine-Tuning (SFT).

Trains GPT-2 on the financial corpus so it learns domain vocabulary
before RL begins. Without SFT, PPO would be shaping incoherent outputs.

Usage (standalone):
    python -m training.sft_trainer
"""

import logging
from pathlib import Path

import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
)

from configs import ProjectConfig, DEFAULT_CONFIG
from data import SFT_DATA, build_sft_dataset
from utils import setup_logger

logger = logging.getLogger(__name__)


def load_tokenizer(model_name: str) -> GPT2Tokenizer:
    """Load and configure GPT-2 tokenizer."""
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 lacks a pad token
    tokenizer.padding_side = "left"            # Left-pad for generation
    return tokenizer


def run_sft(
    cfg: ProjectConfig = DEFAULT_CONFIG,
) -> tuple[GPT2LMHeadModel, GPT2Tokenizer]:
    """
    Execute the SFT stage end-to-end.

    Steps:
        1. Load tokenizer + base GPT-2
        2. Tokenise SFT corpus
        3. Fine-tune with HuggingFace Trainer (causal LM objective)
        4. Optionally save checkpoint

    Args:
        cfg: Master project config.

    Returns:
        (sft_model, tokenizer) — model is on CPU after training;
        caller should move to device as needed.
    """
    setup_logger("rlhf.sft", level=cfg.log_level)
    cfg.paths.make_dirs()

    logger.info(f"Loading base model: {cfg.model.model_name}")
    tokenizer = load_tokenizer(cfg.model.model_name)

    model = GPT2LMHeadModel.from_pretrained(cfg.model.model_name)
    model.config.pad_token_id = tokenizer.eos_token_id

    logger.info(f"Building SFT dataset ({len(SFT_DATA)} samples)...")
    dataset = build_sft_dataset(SFT_DATA, tokenizer, max_length=cfg.model.max_seq_length)

    use_fp16 = cfg.sft.fp16 or torch.cuda.is_available()
    args = TrainingArguments(
        output_dir=str(cfg.paths.sft_checkpoint),
        num_train_epochs=cfg.sft.num_epochs,
        per_device_train_batch_size=cfg.sft.batch_size,
        learning_rate=cfg.sft.learning_rate,
        warmup_steps=cfg.sft.warmup_steps,
        logging_steps=cfg.sft.logging_steps,
        save_strategy=cfg.sft.save_strategy,
        report_to="none",
        fp16=use_fp16,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
    )

    logger.info("Starting SFT training...")
    trainer.train()
    logger.info("SFT training complete ✅")

    model.save_pretrained(str(cfg.paths.sft_checkpoint))
    tokenizer.save_pretrained(str(cfg.paths.sft_checkpoint))
    logger.info(f"SFT checkpoint saved to {cfg.paths.sft_checkpoint}")

    return model, tokenizer


if __name__ == "__main__":
    sft_model, tokenizer = run_sft()
