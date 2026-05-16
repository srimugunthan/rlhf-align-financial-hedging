"""
Option D — Interactive Gradio Demo

Launches a local web UI where you can:
  Tab 1 — Compare Models   : type a financial prompt and see SFT vs PPO
                             completions side-by-side with hedge scores.
  Tab 2 — Score Any Text   : paste any text and get a hedge score breakdown,
                             showing which words contributed positively or
                             negatively.

Requires models to be trained first:
    uv run python scripts/train_all.py

Usage:
    uv run python demo/app.py
    # Then open http://localhost:7860 in your browser
"""

import sys
from pathlib import Path

# Allow imports from project root regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from trl import AutoModelForCausalLMWithValueHead

from data.corpus import FINANCIAL_PROMPTS, REWARD_HACKING_EXAMPLES
from models.policy_model import generate_completion
from utils.reward_utils import compute_hedge_score, HEDGE_WORDS, SPECULATIVE_WORDS

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SFT_DIR = PROJECT_ROOT / "outputs" / "sft_checkpoint"
PPO_DIR = PROJECT_ROOT / "outputs" / "ppo_checkpoint"

# ── Global model state (loaded once at startup) ───────────────────────────────
_tokenizer: GPT2Tokenizer | None = None
_sft_model: GPT2LMHeadModel | None = None
_ppo_model: AutoModelForCausalLMWithValueHead | None = None
_device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_load_error: str | None = None


def _load_models() -> None:
    """Load SFT and PPO models once at startup. Sets _load_error on failure."""
    global _tokenizer, _sft_model, _ppo_model, _load_error

    for label, path in [("SFT", SFT_DIR), ("PPO", PPO_DIR)]:
        if not path.exists():
            _load_error = (
                f"{label} checkpoint not found at: {path}\n\n"
                "Train the models first:\n"
                "    uv run python scripts/train_all.py"
            )
            return

    print(f"Loading models on {_device}...")
    tokenizer = GPT2Tokenizer.from_pretrained(str(SFT_DIR))
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    sft = GPT2LMHeadModel.from_pretrained(str(SFT_DIR)).eval().to(_device)
    ppo = AutoModelForCausalLMWithValueHead.from_pretrained(str(PPO_DIR))
    ppo.pretrained_model.eval()
    ppo.to(_device)

    _tokenizer, _sft_model, _ppo_model = tokenizer, sft, ppo
    print("Models loaded.")


def _generate_sft(prompt: str, max_new_tokens: int = 60) -> str:
    inputs = _tokenizer(prompt, return_tensors="pt").to(_device)
    with torch.no_grad():
        output = _sft_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=_tokenizer.eos_token_id,
        )
    return _tokenizer.decode(output[0], skip_special_tokens=True)


# ── Tab 1: Compare Models ─────────────────────────────────────────────────────

def compare(prompt: str) -> tuple[str, str, str, str, str]:
    """
    Returns: (sft_text, ppo_text, sft_score_str, ppo_score_str, delta_str)
    """
    if _load_error:
        err = f"[Model not loaded]\n\n{_load_error}"
        return err, err, "–", "–", "–"

    prompt = prompt.strip()
    if not prompt:
        return "Enter a prompt above.", "Enter a prompt above.", "–", "–", "–"

    sft_text = _generate_sft(prompt)
    ppo_text = generate_completion(_ppo_model, _tokenizer, prompt, _device)

    sft_score = compute_hedge_score(sft_text)
    ppo_score = compute_hedge_score(ppo_text)
    delta = ppo_score - sft_score

    delta_str = f"{delta:+.3f}  {'✅ improved' if delta > 0 else ('➖ unchanged' if delta == 0 else '⚠️ regressed')}"

    return (
        sft_text,
        ppo_text,
        f"{sft_score:.3f}",
        f"{ppo_score:.3f}",
        delta_str,
    )


# ── Tab 2: Hedge Score Analyser ───────────────────────────────────────────────

def analyse_text(text: str) -> tuple[str, str]:
    """
    Returns: (score_str, breakdown_md)
    """
    text = text.strip()
    if not text:
        return "–", "Enter some financial text above."

    text_lower = text.lower()
    score = compute_hedge_score(text)

    matched_hedge = [w for w in HEDGE_WORDS if w in text_lower]
    matched_spec = [w for w in SPECULATIVE_WORDS if w in text_lower]

    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    caps_penalty = caps_ratio * 2.0
    exclaim_count = text.count("!")
    exclaim_penalty = min(exclaim_count * 0.3, 1.0)

    raw = len(matched_hedge) - 1.5 * len(matched_spec) - caps_penalty - exclaim_penalty

    lines = [
        f"**Score: {score:.3f}** {'(hedged ✅)' if score >= 0.5 else '(speculative ⚠️)'}",
        "",
        "---",
        "**Scoring breakdown**",
        "",
        f"| Component | Count / Value | Contribution |",
        f"|-----------|---------------|--------------|",
        f"| Hedge words matched | {len(matched_hedge)} | +{len(matched_hedge):.1f} |",
        f"| Speculative words matched | {len(matched_spec)} | −{1.5 * len(matched_spec):.1f} |",
        f"| ALL-CAPS ratio | {caps_ratio:.2f} | −{caps_penalty:.2f} |",
        f"| Exclamation marks | {exclaim_count} | −{exclaim_penalty:.2f} |",
        f"| **Raw score** | | **{raw:.2f}** |",
        f"| **sigmoid(raw)** | | **{score:.3f}** |",
        "",
    ]

    if matched_hedge:
        lines += [f"**Hedge words found:** {', '.join(f'`{w}`' for w in matched_hedge)}", ""]
    if matched_spec:
        lines += [f"**Speculative words found:** {', '.join(f'`{w}`' for w in matched_spec)}", ""]
    if not matched_hedge and not matched_spec:
        lines += ["*No hedge or speculative words detected — score driven by caps/punctuation.*", ""]

    return f"{score:.3f}", "\n".join(lines)


# ── Tab 3: Reward Hacking Examples ────────────────────────────────────────────

def _build_hacking_table() -> str:
    rows = ["| Type | Score | Text |", "|------|-------|------|"]
    for text, label in REWARD_HACKING_EXAMPLES:
        score = compute_hedge_score(text)
        short = text[:60] + "..." if len(text) > 60 else text
        rows.append(f"| {label} | {score:.3f} | {short} |")
    return "\n".join(rows)


HACKING_MD = f"""
## Why the KL Penalty Matters

Without the KL divergence penalty, the policy learns to game the reward
model by stuffing hedge words with no coherent meaning — **reward hacking**.

**Total reward formula:**

```
R_total(x, y) = r_RM(x, y) − β × KL(π_θ(y|x) ‖ π_SFT(y|x))
```

The KL term penalises incoherent distributions relative to the coherent
SFT reference, keeping the policy from collapsing into repetitive
hedge-word spam.

{_build_hacking_table()}

The hacked examples score **high** on the heuristic (lots of hedge words)
but are semantically meaningless. The KL penalty prevents this by ensuring
the policy output remains close to the SFT distribution.
"""


# ── Build Gradio UI ───────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    example_prompts = FINANCIAL_PROMPTS[:8]

    with gr.Blocks(title="RLHF Financial Hedging Demo", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
# RLHF Financial Hedging — Interactive Demo

Aligning GPT-2 to prefer **hedged, cautious financial language** over
speculative overconfidence using the full RLHF pipeline (SFT → RM → PPO).
            """
        )

        # ── Tab 1: Compare ────────────────────────────────────────────────────
        with gr.Tab("Compare Models"):
            gr.Markdown(
                "Type a financial prompt and compare completions from the SFT model "
                "(before RLHF) and the PPO-trained policy (after RLHF)."
            )
            with gr.Row():
                prompt_input = gr.Textbox(
                    label="Financial Prompt",
                    placeholder='e.g. "The Federal Reserve" or "The company\'s outlook"',
                    lines=1,
                    scale=4,
                )
                generate_btn = gr.Button("Generate", variant="primary", scale=1)

            gr.Examples(
                examples=[[p] for p in example_prompts],
                inputs=prompt_input,
                label="Example prompts",
            )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### SFT Model (before RLHF)")
                    sft_output = gr.Textbox(label="Completion", lines=5, interactive=False)
                    sft_score = gr.Textbox(label="Hedge Score [0–1]", interactive=False)
                with gr.Column():
                    gr.Markdown("### PPO Model (after RLHF)")
                    ppo_output = gr.Textbox(label="Completion", lines=5, interactive=False)
                    ppo_score = gr.Textbox(label="Hedge Score [0–1]", interactive=False)

            delta_output = gr.Textbox(label="Score Delta (PPO − SFT)", interactive=False)

            generate_btn.click(
                fn=compare,
                inputs=prompt_input,
                outputs=[sft_output, ppo_output, sft_score, ppo_score, delta_output],
            )
            prompt_input.submit(
                fn=compare,
                inputs=prompt_input,
                outputs=[sft_output, ppo_output, sft_score, ppo_score, delta_output],
            )

        # ── Tab 2: Score Analyser ─────────────────────────────────────────────
        with gr.Tab("Hedge Score Analyser"):
            gr.Markdown(
                "Paste any financial text to see its hedge score and a full breakdown "
                "of which words drove the score up or down."
            )
            text_input = gr.Textbox(
                label="Financial Text",
                placeholder="Paste any financial statement here...",
                lines=4,
            )
            analyse_btn = gr.Button("Analyse", variant="primary")

            gr.Examples(
                examples=[
                    ["The stock may experience volatility depending on interest rate decisions."],
                    ["This stock will DEFINITELY skyrocket! Guaranteed massive gains!"],
                    ["Based on historical data, earnings could potentially miss estimates."],
                    ["BUY NOW before it's too late! You can't lose! To the moon!!!"],
                ],
                inputs=text_input,
                label="Example texts",
            )

            score_out = gr.Textbox(label="Hedge Score [0–1]", interactive=False)
            breakdown_out = gr.Markdown()

            analyse_btn.click(
                fn=analyse_text,
                inputs=text_input,
                outputs=[score_out, breakdown_out],
            )
            text_input.submit(
                fn=analyse_text,
                inputs=text_input,
                outputs=[score_out, breakdown_out],
            )

        # ── Tab 3: Reward Hacking ─────────────────────────────────────────────
        with gr.Tab("Reward Hacking"):
            gr.Markdown(HACKING_MD)

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _load_models()
    if _load_error:
        print(f"\n[WARNING] Models could not be loaded:\n{_load_error}\n")
        print("The app will still launch but generation will show the error message.")

    ui = build_ui()
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
