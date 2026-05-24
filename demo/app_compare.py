"""
Compare Models — Standalone Gradio App (Tab 1)

Type a financial prompt and compare completions from the SFT model
(before RLHF) and the PPO-trained policy (after RLHF) side-by-side,
with hedge scores and a score delta.

Requires models to be trained first:
    uv run python scripts/train_all.py

Usage:
    uv run python demo/app_compare.py
    # Then open http://localhost:7860 in your browser
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from trl import AutoModelForCausalLMWithValueHead

from data.corpus import FINANCIAL_PROMPTS
from models.policy_model import generate_completion
from utils.reward_utils import compute_hedge_score

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SFT_DIR = PROJECT_ROOT / "outputs" / "sft_checkpoint"
PPO_DIR = PROJECT_ROOT / "outputs" / "ppo_checkpoint"

# ── Global model state ────────────────────────────────────────────────────────
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


def compare(prompt: str) -> tuple[str, str, str, str, str]:
    """Generate completions from both models and return scores."""
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


def build_ui() -> gr.Blocks:
    example_prompts = FINANCIAL_PROMPTS[:8]

    with gr.Blocks(title="Compare Models — RLHF Financial Hedging", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
# Compare Models — SFT vs PPO

Type a financial prompt and compare completions from the **SFT model** (before RLHF)
and the **PPO-trained policy** (after RLHF), with hedge scores for each.

> **Tip:** Use forward-looking prompts like `Whether inflation will rise` or
> `The risk of a recession` for the clearest hedging contrast.
            """
        )

        with gr.Row():
            prompt_input = gr.Textbox(
                label="Financial Prompt",
                placeholder='e.g. "Whether inflation will continue to rise" or "The risk of a market correction"',
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

    return demo


if __name__ == "__main__":
    _load_models()
    if _load_error:
        print(f"\n[WARNING] Models could not be loaded:\n{_load_error}\n")
        print("The app will still launch but generation will show the error message.")

    ui = build_ui()
    ui.queue(max_size=5)
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)
