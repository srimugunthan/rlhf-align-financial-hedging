"""
PyTorch Dataset and HuggingFace Dataset builders for all three RLHF stages.
"""

from typing import List, Tuple

import torch
from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset
from transformers import PreTrainedTokenizer


# ── Stage 1: SFT Dataset ──────────────────────────────────────────────────────

def build_sft_dataset(
    texts: List[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 128,
) -> Dataset:
    """
    Tokenize raw text into a HuggingFace Dataset for causal LM fine-tuning.
    Labels == input_ids (standard next-token prediction objective).

    Args:
        texts:      List of training sentences.
        tokenizer:  GPT-2 tokenizer (pad_token must be set before calling).
        max_length: Maximum sequence length (longer sequences are truncated).

    Returns:
        HuggingFace Dataset with columns [input_ids, attention_mask, labels],
        formatted as torch tensors.
    """
    def _tokenize(examples):
        tokens = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    raw = Dataset.from_dict({"text": texts})
    tokenized = raw.map(_tokenize, batched=True, remove_columns=["text"])
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return tokenized


# ── Stage 2: Preference Dataset (Reward Model) ────────────────────────────────

class PreferenceDataset(TorchDataset):
    """
    Dataset of pairwise (preferred, dispreferred) text pairs for reward model
    training with the Bradley-Terry objective.

    Each item returns tokenized tensors for both members of a pair so the
    training loop can compute:
        L = -log(sigmoid(r_winner - r_loser))

    Args:
        pairs:      List of (preferred_text, dispreferred_text) tuples.
        tokenizer:  GPT-2 tokenizer.
        max_length: Max token length per text.
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 128,
    ):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        preferred, dispreferred = self.pairs[idx]

        enc_w = self.tokenizer(
            preferred,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        enc_l = self.tokenizer(
            dispreferred,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids_w": enc_w["input_ids"].squeeze(0),
            "attention_mask_w": enc_w["attention_mask"].squeeze(0),
            "input_ids_l": enc_l["input_ids"].squeeze(0),
            "attention_mask_l": enc_l["attention_mask"].squeeze(0),
        }


# ── Stage 3: PPO Prompt Dataset ───────────────────────────────────────────────

def build_ppo_dataset(
    prompts: List[str],
    tokenizer: PreTrainedTokenizer,
    max_length: int = 32,
) -> Dataset:
    """
    Tokenize short financial prompts into a HuggingFace Dataset for PPO rollouts.
    The policy model generates completions from these prompts during training.

    Args:
        prompts:    List of short sentence-starter strings.
        tokenizer:  GPT-2 tokenizer.
        max_length: Max prompt length (keep short — model generates the rest).

    Returns:
        HuggingFace Dataset with columns [input_ids, query].
    """
    tokenized = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    dataset = Dataset.from_dict(
        {
            "input_ids": tokenized["input_ids"].tolist(),
            "query": prompts,
        }
    )
    return dataset


def ppo_data_collator(data: List[dict]) -> dict:
    """
    Collator for PPOTrainer's DataLoader.
    Converts list of dataset items into batched tensors.
    """
    return {
        "input_ids": [torch.tensor(d["input_ids"]) for d in data],
        "query": [d["query"] for d in data],
    }
