# Running RLHF Financial Hedging on GCP

This guide walks through running the full RLHF pipeline on Google Cloud Platform using a GPU-backed Compute Engine VM.

---

## Prerequisites

- A GCP project with billing enabled
- [gcloud CLI](https://cloud.google.com/sdk/docs/install) installed and authenticated locally
- GPU quota approved in your target region (request via **IAM & Admin → Quotas** if needed)

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

---

## 1. Enable Required APIs

```bash
gcloud services enable compute.googleapis.com \
    storage.googleapis.com \
    artifactregistry.googleapis.com
```

---

## 2. GPU Selection Guide (asia-south1 / Bengaluru)

### Check what GPUs are available in your region

```bash
gcloud compute accelerator-types list \
    --filter="zone~'asia-south1'" \
    --format="table(zone,name)"
```

### GPU comparison for this project (GPT-2 RLHF)

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

### Common errors and fixes

| Error | Cause | Fix |
|---|---|---|
| `resource 'family/pytorch-latest-gpu' was not found` | Deprecated image family | Use `common-cu129-ubuntu-2404-nvidia-580` |
| `ZONE_RESOURCE_POOL_EXHAUSTED` | GPU sold out in that zone | Try another zone or switch GPU type |
| `acceleratorType not found` | GPU not available in that zone | Run the `accelerator-types list` command above |

---

## 3. Create a GPU VM

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

---

## 4. SSH into the VM

If the VM is already running:

```bash
gcloud compute ssh rlhf-trainer --zone=asia-south1-b
```

If the VM is stopped, start it first:

```bash
gcloud compute instances start rlhf-trainer --zone=asia-south1-b
# Then SSH in:
gcloud compute ssh rlhf-trainer --zone=asia-south1-b
```

---

## 5. Set Up the Environment

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

---

## 6. Run the Training Pipeline

### Quick demo run (CPU or GPU, ~15 min on CPU / ~2 min on L4)

```bash
uv run python scripts/train_all.py




```

### Production-quality run (GPU recommended)

```bash
uv run python scripts/train_all.py \
    --sft_epochs 10 \
    --rm_epochs 20 \
    --ppo_steps 200
```

### Larger model

```bash
uv run python scripts/train_all.py \
    --model gpt2-medium \
    --ppo_steps 150
```

Checkpoints and plots are saved to `outputs/` automatically.

---

## 7. Run Individual Stages

```bash
# Stage 1 — Supervised Fine-Tuning
uv run python -m training.sft_trainer

# Stage 2 — Reward Model (requires Stage 1 checkpoint)
uv run python -m training.rm_trainer

# Stage 3 — PPO (requires Stage 1 + 2 checkpoints)
uv run python -m training.ppo_trainer
```

---

## 8. Run Tests

```bash
uv run pytest tests/ -v
uv run pytest tests/ -v --cov=.
```

---

## 9. Run the Gradio Demo

The demo requires trained checkpoints. Run the full pipeline first (Step 6), then:

```bash
uv run python demo/app.py


gcloud config set project nanochat-run
gcloud compute ssh rlhf-trainer --zone=asia-south1-b -- -L 7860:localhost:7860
OR


gcloud compute ssh rlhf-trainer --zone=asia-south1-b --project=nanochat-run -- -L 7860:localhost:7860
Verify the instance exists first:


gcloud compute instances list --project=nanochat-run
```

The app binds to port `7860`. To access it from your local browser, open a second terminal and create an SSH tunnel:

```bash
gcloud compute ssh rlhf-trainer --zone=asia-south1-b \
    -- -L 7860:localhost:7860
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
gcloud compute firewall-rules delete allow-gradio
```

---

## 10. Save Outputs to Cloud Storage

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

---

## 11. Stop or Delete the VM

Stop (keeps disk, pauses billing for compute):

```bash
gcloud compute instances stop rlhf-trainer --zone=asia-south1-b
```

Delete (removes VM and disk):

```bash
gcloud compute instances delete rlhf-trainer --zone=asia-south1-b
```

> Save your outputs to Cloud Storage (Step 10) before deleting.

---

## Estimated Costs (asia-south1, on-demand)

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

---

## Alternative: Vertex AI Custom Training Job

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
