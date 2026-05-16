"""
Central configuration for RLHF Financial Hedging project.
All hyperparameters and paths live here — change this file, not the training code.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ── Project root ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class ModelConfig:
    """Base model and tokenizer settings."""
    model_name: str = "gpt2"          # gpt2 | gpt2-medium | gpt2-large
    max_seq_length: int = 128
    max_prompt_length: int = 32
    max_new_tokens: int = 50
    pad_side: str = "left"            # left-pad for generation tasks


@dataclass
class SFTConfig:
    """Supervised Fine-Tuning hyperparameters."""
    num_epochs: int = 5
    batch_size: int = 4
    learning_rate: float = 5e-5
    warmup_steps: int = 10
    logging_steps: int = 10
    output_dir: str = str(PROJECT_ROOT / "outputs" / "sft_checkpoint")
    save_strategy: str = "no"
    fp16: bool = False                # Set True if CUDA available (auto-detected at runtime)


@dataclass
class RewardModelConfig:
    """Reward model training hyperparameters."""
    num_epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    num_labels: int = 1               # Scalar reward output
    output_dir: str = str(PROJECT_ROOT / "outputs" / "reward_model_checkpoint")


@dataclass
class PPOConfig:
    """PPO training hyperparameters."""
    num_steps: int = 30               # Increase to 100-200 for real training
    batch_size: int = 8               # Prompts per PPO step
    mini_batch_size: int = 4          # Mini-batch for gradient updates
    ppo_epochs: int = 4               # Inner PPO epochs per batch
    learning_rate: float = 1e-5
    gradient_accumulation_steps: int = 1

    # KL penalty — prevents reward hacking
    kl_penalty: str = "kl"
    init_kl_coef: float = 0.2         # β initial value
    target_kl: float = 6.0            # Adaptive KL target
    adap_kl_ctrl: bool = True         # Auto-adjust β

    # PPO clipping
    cliprange: float = 0.2            # ε — PPO clip ratio
    vf_coef: float = 0.1              # Value function loss weight
    seed: int = 42

    # Generation kwargs used during PPO rollout
    gen_min_length: int = 20
    gen_max_new_tokens: int = 50
    gen_top_k: int = 50
    gen_top_p: float = 0.9
    gen_temperature: float = 0.8
    gen_do_sample: bool = True


@dataclass
class PathConfig:
    """Output and checkpoint paths."""
    outputs_dir: Path = PROJECT_ROOT / "outputs"
    sft_checkpoint: Path = PROJECT_ROOT / "outputs" / "sft_checkpoint"
    rm_checkpoint: Path = PROJECT_ROOT / "outputs" / "reward_model_checkpoint"
    ppo_checkpoint: Path = PROJECT_ROOT / "outputs" / "ppo_checkpoint"
    plots_dir: Path = PROJECT_ROOT / "outputs" / "plots"
    logs_dir: Path = PROJECT_ROOT / "outputs" / "logs"

    def make_dirs(self):
        for p in [self.outputs_dir, self.sft_checkpoint, self.rm_checkpoint,
                  self.ppo_checkpoint, self.plots_dir, self.logs_dir]:
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class ProjectConfig:
    """Master config — compose all sub-configs here."""
    model: ModelConfig = field(default_factory=ModelConfig)
    sft: SFTConfig = field(default_factory=SFTConfig)
    reward_model: RewardModelConfig = field(default_factory=RewardModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    seed: int = 42
    log_level: str = "INFO"


# Default config singleton
DEFAULT_CONFIG = ProjectConfig()
