# RLHF Financial Hedging

Aligning GPT-2 to prefer **factual, hedged financial language** over
speculative/overconfident language using the full RLHF pipeline.

```
✅  "The stock may experience volatility depending on interest rate decisions."
❌  "This stock will DEFINITELY skyrocket! Guaranteed massive gains!"
```

---

## Pipeline Overview

```
Stage 1 — SFT   →  Fine-tune GPT-2 on financial text corpus
Stage 2 — RM    →  Train Reward Model on pairwise preferences (Bradley-Terry loss)
Stage 3 — PPO   →  RL fine-tuning: R_total = r_RM(text) - β × KL(policy ‖ SFT)
```

The **KL divergence penalty** is the critical stabiliser — it prevents the
policy from drifting into incoherent hedge-word stuffing (reward hacking)
while still improving hedge quality.

---

## Project Structure

```
rlhf_financial_hedging/
│
├── configs/
│   └── config.py           # Dataclass configs for all three stages
│
├── data/
│   ├── corpus.py           # SFT corpus, pairwise pairs, prompt pools
│   └── datasets.py         # Dataset builders (SFT / RM / PPO)
│
├── models/
│   ├── reward_model.py     # Bradley-Terry loss, RM builder & trainer
│   └── policy_model.py     # Policy + frozen reference model builders
│
├── training/
│   ├── sft_trainer.py      # Stage 1: SFT via HuggingFace Trainer
│   ├── rm_trainer.py       # Stage 2: Reward Model training
│   └── ppo_trainer.py      # Stage 3: PPO training loop
│
├── evaluation/
│   └── evaluator.py        # SFT vs RLHF comparison, plots, hacking demo
│
├── utils/
│   ├── reward_utils.py     # Heuristic hedge scorer (proxy for human labels)
│   └── logging_utils.py    # Structured logging setup
│
├── scripts/
│   └── train_all.py        # End-to-end pipeline runner (CLI)
│
├── tests/
│   ├── test_reward_utils.py
│   ├── test_datasets.py
│   └── test_reward_model.py
│
├── demo/
│   ├── compare.py          # Option C: CLI side-by-side SFT vs PPO comparison
│   └── app.py              # Option D: Gradio interactive web demo
│
├── outputs/                # Checkpoints, plots, logs (auto-created)
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Create a virtual environment and install dependencies

```bash
uv venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows

uv pip install -r requirements.txt
```

### 3. Run the full pipeline

```bash
# Default: 5 SFT epochs, 10 RM epochs, 30 PPO steps (fast demo)
uv run python scripts/train_all.py

# Production-quality run
uv run python scripts/train_all.py --ppo_steps 200 --sft_epochs 10 --rm_epochs 20

# Larger model (better generation quality)
uv run python scripts/train_all.py --model gpt2-medium --ppo_steps 150
```

### 4. Run tests

```bash
uv run pytest tests/ -v
uv run pytest tests/ -v --cov=.  # with coverage
```

### 5. Run individual stages

```bash
# Stage 1 only
uv run python -m training.sft_trainer

# Stage 2 only (after SFT)
uv run python -m training.rm_trainer

# Stage 3 only (after SFT + RM)
uv run python -m training.ppo_trainer
```

---

## Demo

Both demos require trained model checkpoints. Run the full pipeline first:

```bash
uv run python scripts/train_all.py
```

### Option C — CLI Comparison (`demo/compare.py`)

Loads the saved SFT and PPO checkpoints and prints a side-by-side table of
completions with hedge scores, requiring no browser or GUI.

```bash
# Default eval prompts
uv run python demo/compare.py

# Custom prompt
uv run python demo/compare.py --prompt "The Federal Reserve"

# Multiple prompts, 3 completions each
uv run python demo/compare.py \
    --prompt "The stock market" \
    --prompt "Bond yields" \
    --n 3
```

Example output:

```
Prompt: "The Federal Reserve"
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
SFT (before RLHF)                                                         | PPO (after RLHF)
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
The Federal Reserve will definitely raise rates 100% — massive gains!     | The Federal Reserve may consider pausing rate hikes depending on inflation
  Hedge score: 0.119                                                         Hedge score: 0.731  (+0.612) ✅
```

### Option D — Gradio Web App (`demo/app.py`)

Launches a local web UI at `http://localhost:7860` with three tabs:

| Tab | What it does |
|-----|-------------|
| **Compare Models** | Type a prompt, see SFT vs PPO completions side-by-side with hedge scores |
| **Hedge Score Analyser** | Paste any text for a full breakdown of which words drove the score |
| **Reward Hacking** | Explains why the KL penalty is critical, with concrete examples |

```bash
uv run python demo/app.py
# Open http://localhost:7860
```

---

## Key Concepts

### Bradley-Terry Loss (Stage 2)

The reward model is trained with pairwise comparison loss:

```
L = -E[ log σ(r(preferred) - r(dispreferred)) ]
```

This is more reliable than absolute ratings: annotators find it
easier to say "A is better than B" than to assign scores.

### PPO with KL Penalty (Stage 3)

```
R_total(x, y) = r_RM(x, y) - β × KL(π_θ(y|x) ‖ π_SFT(y|x))
```

| β (init_kl_coef) | Effect |
|---|---|
| High (0.5+) | Stay close to SFT; conservative improvement |
| Low (0.05) | Aggressive reward optimisation; risk of hacking |
| Adaptive (default) | Auto-adjusts β to hit `target_kl` |

### Reward Hacking

Without KL penalty, the model learns to stuff hedge words with no coherent
meaning (e.g. *"may might could possibly uncertain risk"*) to game the heuristic.
The KL term penalises such degenerate distributions relative to the SFT reference.

---

## Configuration

All hyperparameters live in `configs/config.py`. Key knobs:

```python
# configs/config.py

@dataclass
class PPOConfig:
    num_steps: int = 30          # Increase to 100-200 for real training
    init_kl_coef: float = 0.2    # β — KL penalty strength
    target_kl: float = 6.0       # Adaptive KL target
    adap_kl_ctrl: bool = True    # Auto-adjust β during training
    cliprange: float = 0.2       # PPO clip ratio ε
```

---

## Extending This Project

| Extension | What to change |
|---|---|
| Replace heuristic with a real classifier | `utils/reward_utils.py` → plug in a fine-tuned sentiment/hedging model |
| Use a real financial corpus (SEC filings) | `data/corpus.py` → replace `SFT_DATA` |
| Add human annotation interface | New `annotation/` module to collect pairwise labels |
| Try DPO instead of PPO | New `training/dpo_trainer.py` using `trl.DPOTrainer` |
| Scale to GPT-2 Medium/Large | `--model gpt2-medium` CLI flag |
| Multi-reward (hedge + factuality) | Modify `training/ppo_trainer.py` reward composition |

---

## Requirements

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) ≥ 0.4 (package manager)
- PyTorch ≥ 2.0
- Runs on **CPU** (slow but works) — ~15 min for default 30 PPO steps
- CUDA recommended for `--ppo_steps 100+`
