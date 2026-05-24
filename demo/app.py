"""
RLHF Financial Hedging — Combined Gradio Demo

Launches both tabs in a single app.
Each tab can also be run as a standalone app:

    Tab 1 — Compare Models       →  uv run python demo/app_compare.py        (port 7860)
    Tab 2 — Hedge Score Analyser →  uv run python demo/app_hedge_analyser.py (port 7861)

Requires models to be trained first:
    uv run python scripts/train_all.py

Usage:
    uv run python demo/app.py
    # Then open http://localhost:7860 in your browser
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr

# ── Import logic from standalone apps ────────────────────────────────────────
from demo.app_compare import (
    _load_models,
    _load_error,
    compare,
)
from demo.app_hedge_analyser import analyse_text

from data.corpus import FINANCIAL_PROMPTS


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

        # ── Tab 1: Compare Models ─────────────────────────────────────────────
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
                    sft_score  = gr.Textbox(label="Hedge Score [0–1]", interactive=False)
                with gr.Column():
                    gr.Markdown("### PPO Model (after RLHF)")
                    ppo_output = gr.Textbox(label="Completion", lines=5, interactive=False)
                    ppo_score  = gr.Textbox(label="Hedge Score [0–1]", interactive=False)

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

        # ── Tab 2: Hedge Score Analyser ───────────────────────────────────────
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

            score_out     = gr.Textbox(label="Hedge Score [0–1]", interactive=False)
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

    return demo


if __name__ == "__main__":
    _load_models()
    if _load_error:
        print(f"\n[WARNING] Models could not be loaded:\n{_load_error}\n")
        print("The app will still launch but generation will show the error message.")

    ui = build_ui()
    ui.queue(max_size=5)
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)
