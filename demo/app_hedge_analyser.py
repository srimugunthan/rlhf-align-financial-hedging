"""
Hedge Score Analyser — Standalone Gradio App (Tab 2)

Paste any financial text to see its hedge score and a full breakdown
of which words drove the score up or down.

No model training required — this tab uses only the heuristic
scoring function from utils/reward_utils.py.

Usage:
    uv run python demo/app_hedge_analyser.py
    # Then open http://localhost:7861 in your browser
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr

from utils.reward_utils import compute_hedge_score, HEDGE_WORDS, SPECULATIVE_WORDS


def analyse_text(text: str) -> tuple[str, str]:
    """
    Score a piece of financial text and return a full breakdown.

    Returns:
        score_str  : formatted score string e.g. "0.732"
        breakdown  : markdown table showing each scoring component
    """
    text = text.strip()
    if not text:
        return "–", "Enter some financial text above."

    text_lower = text.lower()
    score = compute_hedge_score(text)

    matched_hedge = [w for w in HEDGE_WORDS if w in text_lower]
    matched_spec  = [w for w in SPECULATIVE_WORDS if w in text_lower]

    caps_ratio      = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    caps_penalty    = caps_ratio * 2.0
    exclaim_count   = text.count("!")
    exclaim_penalty = min(exclaim_count * 0.3, 1.0)

    raw = len(matched_hedge) - 1.5 * len(matched_spec) - caps_penalty - exclaim_penalty

    lines = [
        f"**Score: {score:.3f}** {'(hedged ✅)' if score >= 0.5 else '(speculative ⚠️)'}",
        "",
        "---",
        "**Scoring breakdown**",
        "",
        "| Component | Count / Value | Contribution |",
        "|-----------|---------------|--------------|",
        f"| Hedge words matched       | {len(matched_hedge)}       | +{len(matched_hedge):.1f}         |",
        f"| Speculative words matched | {len(matched_spec)}         | −{1.5 * len(matched_spec):.1f}        |",
        f"| ALL-CAPS ratio            | {caps_ratio:.2f}            | −{caps_penalty:.2f}        |",
        f"| Exclamation marks         | {exclaim_count}             | −{exclaim_penalty:.2f}        |",
        f"| **Raw score**             |                | **{raw:.2f}**      |",
        f"| **sigmoid(raw)**          |                | **{score:.3f}**    |",
        "",
    ]

    if matched_hedge:
        lines += [f"**Hedge words found:** {', '.join(f'`{w}`' for w in matched_hedge)}", ""]
    if matched_spec:
        lines += [f"**Speculative words found:** {', '.join(f'`{w}`' for w in matched_spec)}", ""]
    if not matched_hedge and not matched_spec:
        lines += ["*No hedge or speculative words detected — score driven by caps/punctuation.*", ""]

    return f"{score:.3f}", "\n".join(lines)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Hedge Score Analyser — RLHF Financial Hedging", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
# Hedge Score Analyser

Paste any financial text to measure how **hedged** or **speculative** its language is.

| Score | Meaning |
|---|---|
| **≥ 0.5** | ✅ Hedged — cautious, analytical, appropriately uncertain |
| **< 0.5** | ⚠️ Speculative — overconfident, hype-driven, or misleading |

The score is computed as `sigmoid(hedge_words − 1.5×speculative_words − caps_penalty − exclaim_penalty)`.
            """
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
                ["Based on historical data, earnings could potentially miss estimates."],
                ["Analysis suggests the sector might face headwinds if inflation persists."],
                ["This stock will DEFINITELY skyrocket! Guaranteed massive gains!"],
                ["BUY NOW before it's too late! You can't lose! To the moon!!!"],
                ["Whether inflation will continue to rise remains uncertain and subject to change."],
            ],
            inputs=text_input,
            label="Example texts",
        )

        score_out    = gr.Textbox(label="Hedge Score [0–1]", interactive=False)
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
    ui = build_ui()
    ui.queue(max_size=10)
    ui.launch(server_name="0.0.0.0", server_port=7861, share=False, show_error=True)
