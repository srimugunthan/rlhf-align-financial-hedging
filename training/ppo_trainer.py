"""
Stage 3 — PPO Fine-Tuning.

Uses Proximal Policy Optimization to fine-tune the SFT policy model,
guided by the learned reward model + a KL divergence penalty against
the frozen SFT reference model.

Core RLHF loop (per step):
    1. Policy generates completions from financial prompt seeds
    2. Reward model scores each completion  
    3. PPO updates policy:  R_total = r_RM - β * KL(policy || ref)

Usage (standalone):
    python -m training.ppo_trainer
"""

import logging
from typing import List

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2ForSequenceClassification
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

from configs import ProjectConfig, DEFAULT_CONFIG
from data import FINANCIAL_PROMPTS, build_ppo_dataset, ppo_data_collator
from models import build_policy_model, build_ref_model, get_reward
from utils import setup_logger, compute_hedge_score

logger = logging.getLogger(__name__)


def build_ppo_config(cfg: ProjectConfig) -> PPOConfig:
    """Translate our ProjectConfig into a TRL PPOConfig."""
    return PPOConfig(
        model_name=cfg.model.model_name,
        learning_rate=cfg.ppo.learning_rate,
        batch_size=cfg.ppo.batch_size,
        mini_batch_size=cfg.ppo.mini_batch_size,
        gradient_accumulation_steps=cfg.ppo.gradient_accumulation_steps,
        ppo_epochs=cfg.ppo.ppo_epochs,
        kl_penalty=cfg.ppo.kl_penalty,
        init_kl_coef=cfg.ppo.init_kl_coef,
        target_kl=cfg.ppo.target_kl,
        adap_kl_ctrl=cfg.ppo.adap_kl_ctrl,
        cliprange=cfg.ppo.cliprange,
        vf_coef=cfg.ppo.vf_coef,
        seed=cfg.ppo.seed,
        log_with=None,   # Disable wandb / tensorboard
    )


def run_ppo_training(
    sft_model: GPT2LMHeadModel,
    reward_model: GPT2ForSequenceClassification,
    tokenizer: GPT2Tokenizer,
    cfg: ProjectConfig = DEFAULT_CONFIG,
    device: torch.device = None,
) -> tuple[AutoModelForCausalLMWithValueHead, List[dict]]:
    """
    Execute the PPO training stage end-to-end.

    Args:
        sft_model:     Trained SFT model (source for policy + ref weights).
        reward_model:  Trained reward model (Bradley-Terry).
        tokenizer:     GPT-2 tokenizer.
        cfg:           Master project config.
        device:        Compute device (auto-detected if None).

    Returns:
        (policy_model, training_log)
        training_log is a list of dicts with keys:
            step, mean_reward, kl_divergence, hedge_score
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    setup_logger("rlhf.ppo", level=cfg.log_level)
    cfg.paths.make_dirs()

    # ── Build models ──────────────────────────────────────────────────────────
    logger.info("Building policy and reference models...")
    policy_model = build_policy_model(cfg.model.model_name, sft_model)
    ref_model = build_ref_model(cfg.model.model_name, sft_model)

    # ── Build PPO prompt dataset ──────────────────────────────────────────────
    ppo_dataset = build_ppo_dataset(
        FINANCIAL_PROMPTS, tokenizer, max_length=cfg.model.max_prompt_length
    )
    logger.info(f"PPO prompt pool: {len(ppo_dataset)} prompts")

    # ── Initialise PPO Trainer ────────────────────────────────────────────────
    ppo_cfg = build_ppo_config(cfg)
    ppo_trainer = PPOTrainer(
        config=ppo_cfg,
        model=policy_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=ppo_dataset,
        data_collator=ppo_data_collator,
    )

    # Generation kwargs for rollout
    gen_kwargs = {
        "min_length": cfg.ppo.gen_min_length,
        "max_new_tokens": cfg.ppo.gen_max_new_tokens,
        "top_k": cfg.ppo.gen_top_k,
        "top_p": cfg.ppo.gen_top_p,
        "do_sample": cfg.ppo.gen_do_sample,
        "temperature": cfg.ppo.gen_temperature,
        "pad_token_id": tokenizer.eos_token_id,
    }

    logger.info(f"Starting PPO for {cfg.ppo.num_steps} steps...")
    logger.info(f"  KL β={cfg.ppo.init_kl_coef} | target_kl={cfg.ppo.target_kl} | "
                f"clip ε={cfg.ppo.cliprange}")
    logger.info("=" * 65)

    training_log: List[dict] = []

    for step, batch in enumerate(ppo_trainer.dataloader):
        if step >= cfg.ppo.num_steps:
            break

        query_tensors = batch["input_ids"]

        # ── 1. Generate completions ───────────────────────────────────────────
        response_tensors = ppo_trainer.generate(
            query_tensors,
            return_prompt=False,
            **gen_kwargs,
        )
        batch_texts = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        # ── 2. Score with reward model ────────────────────────────────────────
        rewards = [
            torch.tensor(
                get_reward(reward_model, tokenizer, text, device),
                dtype=torch.float32,
            )
            for text in batch_texts
        ]

        # ── 3. PPO update ─────────────────────────────────────────────────────
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        # ── Logging ───────────────────────────────────────────────────────────
        mean_reward = float(np.mean([r.item() for r in rewards]))
        mean_kl = float(stats.get("objective/kl", 0))
        mean_hedge = float(np.mean([compute_hedge_score(t) for t in batch_texts]))

        training_log.append({
            "step": step,
            "mean_reward": mean_reward,
            "kl_divergence": mean_kl,
            "hedge_score": mean_hedge,
        })

        if (step + 1) % 5 == 0:
            logger.info(
                f"Step {step+1:3d} | Reward: {mean_reward:+.3f} | "
                f"KL: {mean_kl:.3f} | Hedge: {mean_hedge:.3f}"
            )

    logger.info("PPO training complete ✅")

    return policy_model, training_log


if __name__ == "__main__":
    from training.sft_trainer import run_sft
    from training.rm_trainer import run_rm_training

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sft_model, tokenizer = run_sft()
    sft_model.to(device)
    reward_model, _ = run_rm_training(sft_model, tokenizer, device=device)
    run_ppo_training(sft_model, reward_model, tokenizer, device=device)
