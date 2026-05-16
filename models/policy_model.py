"""
Policy and Reference model builders for PPO stage.

AutoModelForCausalLMWithValueHead wraps GPT-2 with an extra value head V(s),
which PPO uses to compute advantage estimates: A(s,a) = Q(s,a) - V(s).

Two separate model instances are created:
  - policy_model : updated by PPO gradient steps
  - ref_model    : frozen SFT copy used for KL divergence penalty
"""

import logging

import torch
from transformers import GPT2LMHeadModel
from trl import AutoModelForCausalLMWithValueHead

logger = logging.getLogger(__name__)


def _load_sft_weights(
    target: AutoModelForCausalLMWithValueHead,
    sft_model: GPT2LMHeadModel,
) -> None:
    """Copy transformer + lm_head weights from sft_model into target."""
    target.pretrained_model.transformer.load_state_dict(
        sft_model.transformer.state_dict()
    )
    target.pretrained_model.lm_head.load_state_dict(
        sft_model.lm_head.state_dict()
    )


def build_policy_model(
    model_name: str,
    sft_model: GPT2LMHeadModel,
) -> AutoModelForCausalLMWithValueHead:
    """
    Create the PPO policy model initialised from SFT weights.

    The value head is randomly initialised — it will be learned during PPO.

    Args:
        model_name: HuggingFace model identifier.
        sft_model:  Trained SFT model used as weight source.

    Returns:
        AutoModelForCausalLMWithValueHead (trainable).
    """
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    _load_sft_weights(policy, sft_model)
    logger.info("Policy model initialised from SFT weights.")
    return policy


def build_ref_model(
    model_name: str,
    sft_model: GPT2LMHeadModel,
) -> AutoModelForCausalLMWithValueHead:
    """
    Create the frozen reference model (KL penalty anchor).

    All parameters are frozen — this model must NOT be updated during PPO.
    It represents the SFT distribution π_SFT that we penalise divergence from.

    Args:
        model_name: HuggingFace model identifier.
        sft_model:  Trained SFT model used as weight source.

    Returns:
        AutoModelForCausalLMWithValueHead (all params frozen).
    """
    ref = AutoModelForCausalLMWithValueHead.from_pretrained(model_name)
    _load_sft_weights(ref, sft_model)

    # Freeze every parameter — this is the KL reference distribution
    for param in ref.parameters():
        param.requires_grad = False

    logger.info("Reference model (frozen SFT copy) ready.")
    return ref


def generate_completion(
    model: AutoModelForCausalLMWithValueHead,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 60,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    Generate a completion from the policy model given a prompt string.

    Args:
        model:          Policy model (or any AutoModelForCausalLMWithValueHead).
        tokenizer:      GPT-2 tokenizer.
        prompt:         Input prompt string.
        device:         Compute device.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature:    Sampling temperature.
        top_p:          Nucleus sampling probability.

    Returns:
        Full decoded string (prompt + completion).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    model.pretrained_model.eval()

    with torch.no_grad():
        output = model.pretrained_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
        )

    return tokenizer.decode(output[0], skip_special_tokens=True)
