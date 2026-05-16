# Implementation Notes: RLHF Financial Hedging

## Project Goal

The goal is to align a GPT-2 language model to prefer **hedged, cautious financial language** over speculative or overconfident language — using the full Reinforcement Learning from Human Feedback (RLHF) pipeline.

**Desired behaviour (preferred):**
> "The stock *may* experience volatility depending on interest rate decisions."

**Undesired behaviour (rejected):**
> "This stock will DEFINITELY skyrocket! Guaranteed massive gains!"

This is a self-contained demonstration of the canonical RLHF workflow:

```
Stage 1 — SFT   →  Supervised Fine-Tuning on domain text
Stage 2 — RM    →  Reward Model trained on pairwise human preferences
Stage 3 — PPO   →  Policy fine-tuned via RL to maximise RM reward
```

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    Three-Stage RLHF Pipeline                     │
│                                                                  │
│  [GPT-2 base]                                                    │
│       │                                                          │
│       ▼                                                          │
│  Stage 1: SFT ──────────────────────► sft_model                 │
│  (causal LM loss on 25 financial sentences)                      │
│       │                                                          │
│       ├──────────────────────────────► reward_model             │
│       │   Stage 2: RM                 (SFT backbone +           │
│       │   (Bradley-Terry loss on       scalar reward head)       │
│       │    15 preference pairs)                                  │
│       │                                                          │
│       ▼                                                          │
│  Stage 3: PPO ──────────────────────► policy_model              │
│  reward_model scores completions     (SFT weights +             │
│  KL penalty anchors to sft_model      value head)               │
└──────────────────────────────────────────────────────────────────┘
```

---

## Stage 1: Supervised Fine-Tuning (SFT)

**File:** `training/sft_trainer.py`

### Purpose

Adapt the base GPT-2 model to the financial text domain before any preference learning begins. Without this step, the base model generates generic text unrelated to finance.

### Data

`data/corpus.py` — `SFT_DATA` (25 sentences):
- ~20 hedged financial statements (positive examples)
- ~5 speculative statements (for diversity, prevents degenerate collapse)
- Mix of factual neutral news-style sentences

### Training

Standard causal language modelling: predict the next token given all previous tokens.

```
Loss = -Σ log P(token_t | token_1 ... token_{t-1})
```

Uses HuggingFace `Trainer` with:
- `GPT2LMHeadModel` loaded from `"gpt2"`
- `max_seq_length = 128` tokens
- `labels = input_ids` (standard CLM objective)
- Left-padding (required for generation tasks)
- FP16 auto-enabled when CUDA is available

**Key hyperparameters** (`configs/config.py → SFTConfig`):

| Parameter | Value | Notes |
|-----------|-------|-------|
| `num_epochs` | 5 | Increase to 10+ for better convergence |
| `batch_size` | 4 | |
| `learning_rate` | 5e-5 | Standard for GPT-2 fine-tuning |
| `warmup_steps` | 10 | |

**Output:** `outputs/sft_checkpoint/`

---

## Stage 2: Reward Model (RM)

**Files:** `training/rm_trainer.py`, `models/reward_model.py`

### Purpose

Learn a scalar reward function that reflects human preferences — assigning higher scores to hedged statements and lower scores to speculative ones.

### Data

`data/corpus.py` — `PAIRWISE_DATA` (15 pairs):

Each entry is `(preferred, dispreferred)`:
```python
(
  "The company's revenue may decline if economic conditions worsen.",   # ✅ preferred
  "This company will CRUSH earnings — massive profits guaranteed!"      # ❌ dispreferred
)
```

### Model Architecture

`GPT2ForSequenceClassification` with `num_labels=1`:
- **Backbone:** GPT-2 transformer (12 layers, 768 hidden dims)
- **Head:** Single linear layer producing a scalar reward `r ∈ ℝ`
- **Weight init:** Backbone copied from SFT model; only the head is randomly initialised

This weight transfer is important — the backbone already understands financial language from Stage 1, so only the reward head needs to be learned.

### Loss Function: Bradley-Terry

```
L = -E[ log σ(r(preferred) − r(dispreferred)) ]
```

Implemented in `models/reward_model.py → bradley_terry_loss()`.

This pairwise loss is more reliable than absolute score regression because:
- Annotators find it easier to compare two texts than to assign a number
- The loss only requires `r_preferred > r_dispreferred` by some margin, not a specific value
- Naturally handles inter-annotator scale differences

**Optimiser:** AdamW with gradient clipping (`grad_clip = 1.0`) and L2 regularisation (`weight_decay = 0.01`).

**Key hyperparameters** (`configs/config.py → RewardModelConfig`):

| Parameter | Value | Notes |
|-----------|-------|-------|
| `num_epochs` | 10 | More epochs help with only 15 pairs |
| `learning_rate` | 1e-5 | Lower than SFT; only head is trained heavily |
| `weight_decay` | 0.01 | L2 regularisation |
| `grad_clip` | 1.0 | Stability |

**Output:** `outputs/reward_model_checkpoint/`

### Heuristic Proxy Signal

`utils/reward_utils.py → compute_hedge_score()` provides a rule-based sanity check alongside the learned RM:

```
raw = hedge_count − 1.5 × speculative_count − caps_penalty − exclamation_penalty
score = sigmoid(raw)  ∈ [0, 1]
```

- `hedge_count`: occurrences of 27 hedge words ("may", "might", "could", "risk", "volatility", ...)
- `speculative_count`: occurrences of 24 speculative phrases ("will definitely", "guaranteed", "to the moon", ...)
- `caps_penalty`: `caps_ratio × 2.0`
- `exclamation_penalty`: `min(count × 0.3, 1.0)`

This simulates human labels without actual annotation infrastructure. Real RLHF would replace this with a human annotation interface.

---

## Stage 3: PPO Fine-Tuning

**Files:** `training/ppo_trainer.py`, `models/policy_model.py`

### Purpose

Use Proximal Policy Optimisation to fine-tune the SFT model to generate text that scores higher under the learned reward model, while preventing the policy from drifting too far from the SFT distribution.

### Two-Model Setup

```
policy_model  — trainable, updated by gradient steps
ref_model     — frozen copy of SFT model, acts as KL anchor
```

Both start with identical SFT weights. The policy model additionally has a **value head** — a small MLP that estimates `V(s)`, the expected return from state `s`. This is needed for advantage estimation in PPO.

```python
# models/policy_model.py
AutoModelForCausalLMWithValueHead   # policy: trainable + value head
AutoModelForCausalLM                # ref: frozen, no value head
```

### Reward with KL Penalty

The total reward at each PPO step is:

```
R_total(x, y) = r_RM(x, y) − β × KL(π_θ(y|x) ‖ π_SFT(y|x))
```

Where:
- `r_RM(x, y)` — scalar from the learned reward model
- `β` — KL penalty coefficient (adaptive)
- `KL(π_θ ‖ π_SFT)` — token-level KL divergence between policy and SFT

The KL term is critical. Without it, the policy quickly learns to game the reward model by stuffing hedge words regardless of coherence:

```
❌ Reward-hacked output:
   "The market may might could possibly uncertain risk volatile caution..."
```

The KL penalty prevents this by ensuring degenerate distributions incur a large cost relative to the coherent SFT reference.

### Adaptive KL Control

`β` is adjusted automatically during training to maintain KL near `target_kl = 6.0`:
- If `KL > target_kl`: increase `β` (tighter constraint on policy drift)
- If `KL < target_kl`: decrease `β` (allow more aggressive reward optimisation)

This is controlled by `adap_kl_ctrl = True` in `PPOConfig`.

| `β` value | Effect |
|-----------|--------|
| High (≥ 0.5) | Stay close to SFT; conservative, stable improvement |
| Low (≤ 0.05) | Aggressive reward maximisation; risk of reward hacking |
| Adaptive (default) | Auto-balances exploration and stability |

### PPO Update Loop

Each step in `training/ppo_trainer.py → run_ppo_training()`:

1. **Rollout:** Sample a batch of prompts from `FINANCIAL_PROMPTS` (23 prompts), generate completions using the policy with nucleus sampling (`top_p=0.9`, `temperature=0.8`)
2. **Score:** Pass each completion through `reward_model → get_reward()` to get a scalar
3. **Advantage estimation:** Value head computes `V(s)`; advantage `A = R_total − V(s)`
4. **PPO gradient step:** Update policy using clipped surrogate objective over `ppo_epochs=4` inner epochs per batch
5. **Log:** Record `mean_reward`, `kl_divergence`, `hedge_score` per step

The clipped objective prevents excessively large policy updates:

```
L_PPO = E[ min(ratio × A, clip(ratio, 1−ε, 1+ε) × A) ]
```

where `ratio = π_θ(a|s) / π_old(a|s)` and `ε = cliprange = 0.2`.

**Key hyperparameters** (`configs/config.py → PPOConfig`):

| Parameter | Value | Notes |
|-----------|-------|-------|
| `num_steps` | 30 | Demo; use 100–200 for real training |
| `batch_size` | 8 | Prompts per PPO step |
| `mini_batch_size` | 4 | For gradient updates |
| `ppo_epochs` | 4 | Inner loop per batch |
| `init_kl_coef` (β) | 0.2 | Initial KL penalty |
| `target_kl` | 6.0 | Adaptive KL target |
| `cliprange` (ε) | 0.2 | PPO clip ratio |
| `vf_coef` | 0.1 | Value function loss weight |
| `gen_temperature` | 0.8 | Generation diversity |
| `gen_top_p` | 0.9 | Nucleus sampling cutoff |

**Output:** `outputs/ppo_checkpoint/`

---

## Evaluation

**File:** `evaluation/evaluator.py`

### SFT vs PPO Comparison

`compare_sft_vs_rlhf()` generates completions from both models on held-out `EVAL_PROMPTS` (5 prompts) and scores each with the heuristic `compute_hedge_score()`. Prints a side-by-side table showing:
- Raw completion text
- Hedge score for each model
- Per-prompt improvement

### Reward Hacking Demo

`reward_hacking_demo()` illustrates degenerate outputs that would emerge without the KL penalty, making the stabilising role of `β` concrete and visible.

### Training Curves

`plot_training_curves()` produces a 3-panel matplotlib figure saved to `outputs/plots/`:
1. Mean reward over PPO steps
2. KL divergence (policy vs SFT reference)
3. Heuristic hedge score

---

## Demo

Two standalone demo scripts live in `demo/`. Both load saved checkpoints and require the pipeline to be run first (`scripts/train_all.py`).

### Option C — CLI Comparison (`demo/compare.py`)

A zero-dependency terminal script that loads the SFT and PPO checkpoints and prints a formatted two-column comparison table with hedge scores for each completion.

**Key design points:**
- Adds project root to `sys.path` so it can reuse `models`, `utils`, and `data` modules without installation
- Loads `GPT2LMHeadModel` for SFT generation and `AutoModelForCausalLMWithValueHead` for PPO generation — matching exactly how each model was saved
- `--prompt TEXT` flag is repeatable; defaults to `EVAL_PROMPTS` if omitted
- `--n N` generates multiple independent completions per prompt to show variability
- Fails fast with a clear message if either checkpoint directory is absent
- Reports per-prompt delta and an overall average delta at the end

**Usage:**
```bash
uv run python demo/compare.py
uv run python demo/compare.py --prompt "The Federal Reserve"
uv run python demo/compare.py --prompt "Bond yields" --n 3
```

### Option D — Gradio Web App (`demo/app.py`)

An interactive browser-based demo running at `http://localhost:7860`. Models are loaded once at startup and shared across requests.

**Three tabs:**

| Tab | Implementation | Purpose |
|-----|----------------|---------|
| **Compare Models** | Calls `generate_from_sft()` + `generate_completion()` | Side-by-side completions with hedge scores and ✅/⚠️ delta indicator |
| **Hedge Score Analyser** | Calls `compute_hedge_score()` + iterates `HEDGE_WORDS` / `SPECULATIVE_WORDS` | Full scoring breakdown in a markdown table: matched words, caps/exclamation penalties, raw score, sigmoid output |
| **Reward Hacking** | Static markdown rendered from `REWARD_HACKING_EXAMPLES` | Shows why KL penalty is necessary; scored table of gamed vs genuine examples |

**Key design points:**
- Model state held in module-level globals (`_sft_model`, `_ppo_model`, `_tokenizer`) — loaded once before `ui.launch()`
- If checkpoints are missing, `_load_error` is set and all generation callbacks return a descriptive error string instead of crashing the server
- `gr.Examples` widgets provide clickable example prompts and texts in Tabs 1 and 2 to make the demo self-guided
- `server_name="0.0.0.0"` allows access from other machines on the same network (useful for live presentations)

**Usage:**
```bash
uv run python demo/app.py
# Open http://localhost:7860
```

**Dependency:** `gradio>=4.0.0` added to `requirements.txt`.

---

## Data Flow Summary

```
SFT_DATA (25 sentences)
    │
    └─► run_sft() ──────────────────────────────────► sft_model
                                                           │
PAIRWISE_DATA (15 pairs)                                   │
    │                                                      │
    └─► run_rm_training(sft_model) ────────────────► reward_model
                                                           │
FINANCIAL_PROMPTS (23 prompts)                             │
    │                                                      │
    └─► run_ppo_training(sft_model, reward_model) ──► policy_model
                                                           │
EVAL_PROMPTS (5 prompts)                                   │
    │                                                      │
    └─► compare_sft_vs_rlhf(sft_model, policy_model) ─► results table
                                                           + training curves
```

---

## Key Design Decisions

### Why Bradley-Terry over regression?

Pairwise comparison is more robust than learning absolute reward values. Two annotators can disagree on whether a sentence deserves a score of 7 vs 8, but will likely agree that sentence A is better than sentence B. The Bradley-Terry model formalises this: it only requires consistent relative ordering, not consistent absolute magnitudes.

### Why transfer SFT weights to the reward model backbone?

The reward model needs to understand the semantics of financial text to distinguish hedged from speculative language. Starting from random weights would require far more data to reach the same discrimination ability. Transferring the SFT backbone means only the scalar classification head needs significant training.

### Why left-pad instead of right-pad?

Transformer attention uses the last token's hidden state for sequence-level tasks (reward scoring, value estimation). Left-padding ensures the final meaningful token is always at the rightmost position, regardless of sequence length, making pooling consistent across variable-length inputs.

### Why freeze the reference model?

The KL penalty `KL(π_θ ‖ π_SFT)` is measured against a fixed target. If `π_SFT` were updated during PPO, the penalty term would shift, destabilising training. A frozen reference provides a stable anchor throughout the entire PPO phase.

---

## File Reference

| File | Stage | Role |
|------|-------|------|
| `configs/config.py` | All | Centralised hyperparameter dataclasses |
| `data/corpus.py` | 1, 2, 3 | Static datasets and prompt pools |
| `data/datasets.py` | 1, 2, 3 | PyTorch/HF Dataset builders |
| `models/reward_model.py` | 2 | Bradley-Terry loss, RM builder, training loop |
| `models/policy_model.py` | 3 | Policy and reference model builders, generation |
| `training/sft_trainer.py` | 1 | SFT via HuggingFace Trainer |
| `training/rm_trainer.py` | 2 | Reward model training orchestration |
| `training/ppo_trainer.py` | 3 | PPO loop using TRL PPOTrainer |
| `utils/reward_utils.py` | 2, eval | Heuristic hedge scorer |
| `utils/logging_utils.py` | All | Structured logging setup |
| `evaluation/evaluator.py` | eval | SFT vs PPO comparison, plots, hacking demo |
| `scripts/train_all.py` | All | CLI entry point, runs all three stages |
| `demo/compare.py` | demo | CLI side-by-side SFT vs PPO comparison with hedge scores |
| `demo/app.py` | demo | Gradio web app: compare tab, score analyser tab, reward hacking tab |
