"""
Reward Model for RLHF financial hedging alignment.

Architecture: GPT-2 backbone + linear scalar head (hidden_size → 1).
Loss:         Bradley-Terry pairwise preference loss.
Init:         Backbone weights transferred from the SFT model so the
              reward model already understands financial language before
              its classification head is trained.
"""

import logging
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2ForSequenceClassification, GPT2LMHeadModel, PreTrainedTokenizer

from data.datasets import PreferenceDataset

logger = logging.getLogger(__name__)


# ── Loss Function ─────────────────────────────────────────────────────────────

def bradley_terry_loss(
    reward_w: torch.Tensor,
    reward_l: torch.Tensor,
) -> torch.Tensor:
    """
    Bradley-Terry pairwise preference loss.

    Objective: push r(preferred) > r(dispreferred).

        L = -E[ log σ(r_w - r_l) ]

    Args:
        reward_w: Scalar rewards for preferred (winner) completions. Shape (B,).
        reward_l: Scalar rewards for dispreferred (loser) completions. Shape (B,).

    Returns:
        Scalar loss tensor.
    """
    return -torch.log(torch.sigmoid(reward_w - reward_l)).mean()


# ── Model Builder ─────────────────────────────────────────────────────────────

def build_reward_model(
    model_name: str,
    sft_model: GPT2LMHeadModel,
    device: torch.device,
) -> GPT2ForSequenceClassification:
    """
    Instantiate a GPT-2 reward model, transferring backbone weights from the
    SFT model. Only the classification head (random-init) will be newly trained.

    Args:
        model_name: HuggingFace model identifier (e.g. "gpt2").
        sft_model:  Trained SFT model whose transformer weights we reuse.
        device:     Target device.

    Returns:
        GPT2ForSequenceClassification with num_labels=1 on `device`.
    """
    reward_model = GPT2ForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,  # Scalar reward output
    )
    reward_model.config.pad_token_id = sft_model.config.pad_token_id

    # Transfer backbone weights — reward model inherits financial domain knowledge
    reward_model.transformer.load_state_dict(sft_model.transformer.state_dict())

    logger.info("Reward model backbone initialised from SFT weights.")
    return reward_model.to(device)


# ── Training ──────────────────────────────────────────────────────────────────

def train_reward_model(
    model: GPT2ForSequenceClassification,
    pairs: List[Tuple[str, str]],
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    num_epochs: int = 10,
    lr: float = 1e-5,
    batch_size: int = 4,
    weight_decay: float = 0.01,
    grad_clip: float = 1.0,
    max_length: int = 128,
) -> List[float]:
    """
    Train the reward model on pairwise preferences using Bradley-Terry loss.

    Args:
        model:       GPT2ForSequenceClassification (num_labels=1).
        pairs:       List of (preferred, dispreferred) text tuples.
        tokenizer:   GPT-2 tokenizer.
        device:      Compute device.
        num_epochs:  Training epochs.
        lr:          AdamW learning rate.
        batch_size:  Pairs per batch.
        weight_decay:L2 regularisation.
        grad_clip:   Gradient norm clipping threshold.
        max_length:  Max tokenisation length.

    Returns:
        List of per-epoch average losses.
    """
    dataset = PreferenceDataset(pairs, tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epoch_losses: List[float] = []
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0

        for batch in loader:
            r_w = model(
                input_ids=batch["input_ids_w"].to(device),
                attention_mask=batch["attention_mask_w"].to(device),
            ).logits.squeeze(-1)  # (B,)

            r_l = model(
                input_ids=batch["input_ids_l"].to(device),
                attention_mask=batch["attention_mask_l"].to(device),
            ).logits.squeeze(-1)  # (B,)

            loss = bradley_terry_loss(r_w, r_l)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            running_loss += loss.item()

        avg = running_loss / max(len(loader), 1)
        epoch_losses.append(avg)

        if (epoch + 1) % 2 == 0:
            logger.info(f"  RM Epoch {epoch+1:2d}/{num_epochs} | BT Loss: {avg:.4f}")

    return epoch_losses


# ── Inference ─────────────────────────────────────────────────────────────────

def get_reward(
    model: GPT2ForSequenceClassification,
    tokenizer: PreTrainedTokenizer,
    text: str,
    device: torch.device,
    max_length: int = 128,
) -> float:
    """
    Return the scalar reward for a single text string.

    Args:
        model:     Trained reward model.
        tokenizer: GPT-2 tokenizer.
        text:      Input text to score.
        device:    Compute device.
        max_length:Max tokenisation length.

    Returns:
        Scalar float reward.
    """
    model.eval()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding="max_length",
    ).to(device)

    with torch.no_grad():
        reward = model(**inputs).logits.squeeze().item()

    return float(reward)
