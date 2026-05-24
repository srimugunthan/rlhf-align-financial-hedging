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
│   ├── compare.py              # Option C: CLI side-by-side SFT vs PPO comparison
│   ├── app.py                  # Option D: Gradio app — both tabs combined
│   ├── app_compare.py          # Standalone: Compare Models tab only       (port 7860)
│   └── app_hedge_analyser.py   # Standalone: Hedge Score Analyser tab only (port 7861)
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

### Option D — Gradio Web App

The Gradio demo is split into **two standalone apps** — one per tab — plus a
combined app that shows both together.

| App | Tab shown | Port | Needs trained models? |
|-----|-----------|------|-----------------------|
| `demo/app_compare.py` | Compare Models | 7860 | ✅ Yes |
| `demo/app_hedge_analyser.py` | Hedge Score Analyser | 7861 | ❌ No |
| `demo/app.py` | Both tabs | 7860 | ✅ Yes |

#### Run a single tab

```bash
# Tab 1 — Compare SFT vs PPO completions side-by-side (requires trained checkpoints)
uv run python demo/app_compare.py
# Open http://localhost:7860

# Tab 2 — Paste any text and get a full hedge score breakdown (no models needed)
uv run python demo/app_hedge_analyser.py
# Open http://localhost:7861
```

#### Run all three tabs together

```bash
uv run python demo/app.py
# Open http://localhost:7860
```

#### Tab descriptions

| Tab | What it does |
|-----|-------------|
| **Compare Models** | Type a prompt, see SFT vs PPO completions side-by-side with hedge scores and score delta |
| **Hedge Score Analyser** | Paste any financial text for a full breakdown of which words drove the score up or down |

> **Tip for clear SFT vs PPO contrast:** Use forward-looking prompts such as
> `Whether inflation will continue to rise` or `The risk of a market correction`
> rather than short neutral starters like `The Federal Reserve`.
> See [`test_prompts.md`](test_prompts.md) for a full categorised prompt list.

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

## Running in GCP

This section walks through running the full RLHF pipeline on Google Cloud Platform using a GPU-backed Compute Engine VM.

### Prerequisites

- A GCP project with billing enabled
- [gcloud CLI](https://cloud.google.com/sdk/docs/install) installed and authenticated locally
- GPU quota approved in your target region (request via **IAM & Admin → Quotas** if needed)

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### 1. Enable Required APIs

```bash
gcloud services enable compute.googleapis.com \
    storage.googleapis.com \
    artifactregistry.googleapis.com
```

### 2. GPU Selection Guide (asia-south1 / Bengaluru)

#### Check what GPUs are available in your region

```bash
gcloud compute accelerator-types list \
    --filter="zone~'asia-south1'" \
    --format="table(zone,name)"
```

#### GPU comparison for this project (GPT-2 RLHF)

| GPU | VRAM | Machine type | Approx cost (asia-south1) | Verdict |
|---|---|---|---|---|
| **nvidia-l4** | 24 GB | `g2-standard-8` | ~$0.70/hr | **Recommended — confirmed working, good price/perf** |
| nvidia-tesla-t4 | 16 GB | `n1-standard-8` | ~$0.35/hr | Cheapest, but frequently exhausted in asia-south1 |
| nvidia-h100-80gb | 80 GB | `a3-highgpu-8g` | ~$3.50/hr | Overkill for GPT-2, ~5x cost |
| nvidia-h200-141gb | 141 GB | `a3-megagpu-8g` | ~$5+/hr | For 70B+ models only |

**Why L4 for this project:** GPT-2 (117M–345M params) fits easily in 24 GB. The L4 is ~2x faster than T4 at FP16 and consistently available in `asia-south1-b`. T4 is cheaper on paper but was exhausted across all asia-south1 zones during testing.

> **Note:** T4 and L4 require different machine types. Using the wrong one causes an error:
> - T4 → `n1-standard-8`
> - L4 → `g2-standard-8`

#### Common errors and fixes

| Error | Cause | Fix |
|---|---|---|
| `resource 'family/pytorch-latest-gpu' was not found` | Deprecated image family | Use `common-cu129-ubuntu-2404-nvidia-580` |
| `ZONE_RESOURCE_POOL_EXHAUSTED` | GPU sold out in that zone | Try another zone or switch GPU type |
| `acceleratorType not found` | GPU not available in that zone | Run the `accelerator-types list` command above |

### 3. Create a GPU VM

The default 30-step PPO demo runs on CPU (~15 min), but serious training (`--ppo_steps 100+`) needs a GPU.

**Recommended — L4 in asia-south1-b (confirmed working)**

```bash
gcloud compute instances create rlhf-trainer \
    --zone=asia-south1-b \
    --machine-type=g2-standard-8 \
    --accelerator=type=nvidia-l4,count=1 \
    --image-family=common-cu129-ubuntu-2404-nvidia-580 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --maintenance-policy=TERMINATE
```

**Alternative — T4 if available (cheaper but often exhausted)**

```bash
gcloud compute instances create rlhf-trainer \
    --zone=asia-south1-b \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=common-cu129-ubuntu-2404-nvidia-580 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --maintenance-policy=TERMINATE
```

If a zone is exhausted, try `asia-south1-a` or `asia-south1-c` by changing `--zone`.

**CPU-only (demo / testing only, no GPU needed)**

```bash
gcloud compute instances create rlhf-trainer-cpu \
    --zone=asia-south1-b \
    --machine-type=n2-standard-4 \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=50GB
```

### 4. SSH into the VM

If the VM is already running:

```bash
gcloud compute ssh rlhf-trainer --zone=asia-south1-b
```

If the VM is stopped, start it first:

```bash
gcloud compute instances start rlhf-trainer --zone=asia-south1-b
gcloud compute ssh rlhf-trainer --zone=asia-south1-b
```

### 5. Set Up the Environment

```bash
# Verify GPU is visible (GPU VMs only)
nvidia-smi

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Clone the repo
git clone https://github.com/YOUR_USERNAME/rlhf_financial_hedging.git
cd rlhf_financial_hedging

# Create virtualenv and install dependencies
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 6. Run the Training Pipeline

#### Quick demo run (CPU or GPU, ~15 min on CPU / ~2 min on L4)

```bash
uv run python scripts/train_all.py
```

#### Production-quality run (GPU recommended)

```bash
uv run python scripts/train_all.py \
    --sft_epochs 10 \
    --rm_epochs 20 \
    --ppo_steps 200
```

#### Larger model

```bash
uv run python scripts/train_all.py \
    --model gpt2-medium \
    --ppo_steps 150
```

Checkpoints and plots are saved to `outputs/` automatically.

### 7. Run Individual Stages

```bash
# Stage 1 — Supervised Fine-Tuning
uv run python -m training.sft_trainer

# Stage 2 — Reward Model (requires Stage 1 checkpoint)
uv run python -m training.rm_trainer

# Stage 3 — PPO (requires Stage 1 + 2 checkpoints)
uv run python -m training.ppo_trainer
```

### 8. Run Tests

```bash
uv run pytest tests/ -v
uv run pytest tests/ -v --cov=.
```

### 9. Run the Gradio Demo

The demo requires trained checkpoints. Run the full pipeline first (Step 6), then:

```bash
uv run python demo/app.py
```

The app binds to port `7860`. To access it from your local browser, open a second terminal and create an SSH tunnel:

```bash
gcloud compute ssh rlhf-trainer --zone=asia-south1-b \
    -- -L 7860:localhost:7860

# If you need to specify the project explicitly:
gcloud compute ssh rlhf-trainer --zone=asia-south1-b \
    --project=YOUR_PROJECT_ID -- -L 7860:localhost:7860
```

Then open `http://localhost:7860` in your browser.

Alternatively, open port 7860 in the firewall (less secure — use only temporarily):

```bash
gcloud compute firewall-rules create allow-gradio \
    --allow tcp:7860 \
    --target-tags rlhf-demo \
    --description "Temporary: Gradio demo access"

gcloud compute instances add-tags rlhf-trainer \
    --tags rlhf-demo \
    --zone asia-south1-b
```

Then navigate to `http://EXTERNAL_IP:7860`. Delete the rule when done:

```bash
gcloud compute instances remove-tags rlhf-trainer \
    --tags rlhf-demo \
    --zone asia-south1-b

gcloud compute firewall-rules delete allow-gradio
```

To verify the VM's tags and firewall rules:

```bash
# List all firewall rules in the project
gcloud compute firewall-rules list

# Filter to only Gradio/RLHF rules
gcloud compute firewall-rules list --filter="name~rlhf OR name~gradio"

# Check tags on the VM
gcloud compute instances describe rlhf-trainer \
    --zone asia-south1-b \
    --format="get(tags.items)"

# Find firewall rules targeting a specific tag
gcloud compute firewall-rules list --filter="targetTags=rlhf-demo"
```

### 10. Save Outputs to Cloud Storage

Upload checkpoints and plots so they survive VM deletion:

```bash
# Create a bucket (one-time)
gcloud storage buckets create gs://YOUR_BUCKET_NAME --location=asia-south1

# Upload outputs
gcloud storage cp -r outputs/ gs://YOUR_BUCKET_NAME/rlhf-outputs/
```

Download later:

```bash
gcloud storage cp -r gs://YOUR_BUCKET_NAME/rlhf-outputs/ ./outputs/
```

### 11. Stop or Delete the VM

Stop (keeps disk, pauses billing for compute):

```bash
gcloud compute instances stop rlhf-trainer --zone=asia-south1-b
```

Delete (removes VM and disk):

```bash
gcloud compute instances delete rlhf-trainer --zone=asia-south1-b
```

> Save your outputs to Cloud Storage (Step 10) before deleting.

### Estimated Costs (asia-south1, on-demand)

| Configuration | Hourly cost (approx.) | Notes |
|---|---|---|
| `g2-standard-8` + L4 GPU | ~$0.70/hr | **Confirmed working — recommended** |
| `n1-standard-8` + T4 GPU | ~$0.35/hr | Cheaper but frequently exhausted in asia-south1 |
| `n2-standard-4` CPU only | ~$0.19/hr | Demo/testing only, ~15 min per pipeline run |
| Cloud Storage (outputs) | ~$0.02/GB/month | For checkpoints and plots |

**Typical training session cost estimate:**

| Run type | Duration (L4) | Cost |
|---|---|---|
| Quick demo (30 PPO steps) | ~2 min | <$0.03 |
| Standard run (100 PPO steps) | ~8 min | ~$0.10 |
| Production run (200 PPO steps, gpt2-medium) | ~25 min | ~$0.30 |

Use [GCP Pricing Calculator](https://cloud.google.com/products/calculator) for an exact estimate.

### Alternative: Vertex AI Custom Training Job

For managed, serverless GPU training without managing a VM:

```bash
gcloud ai custom-jobs create \
    --region=us-central1 \
    --display-name=rlhf-financial-hedging \
    --worker-pool-spec=machine-type=n1-standard-8,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,replica-count=1,container-image-uri=us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-gpu.1-13.py310 \
    --args="python,scripts/train_all.py,--ppo_steps,200"
```

Monitor the job:

```bash
gcloud ai custom-jobs list --region=us-central1
```

---

## Requirements

- Python ≥ 3.10
- [uv](https://docs.astral.sh/uv/) ≥ 0.4 (package manager)
- PyTorch ≥ 2.0
- Runs on **CPU** (slow but works) — ~15 min for default 30 PPO steps
- CUDA recommended for `--ppo_steps 100+`
