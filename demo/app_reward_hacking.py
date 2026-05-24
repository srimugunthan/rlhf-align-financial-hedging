"""
Reward Hacking — Standalone Gradio App (Tab 3)

Shows examples of reward-hacked outputs (texts that game the hedge
score by stuffing hedge words with no coherent meaning) and explains
why the KL penalty in PPO prevents this.

No model training required — this is a static explainer page.

Usage:
    uv run python demo/app_reward_hacking.py
    # Then open http://localhost:7860 in your browser
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr

from data.corpus import REWARD_HACKING_EXAMPLES
from utils.reward_utils import compute_hedge_score


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

The **hacked** examples score high on the heuristic (lots of hedge words)
but are semantically meaningless. The KL penalty prevents this by ensuring
the policy output remains close to the SFT distribution.

---

## Hedge Word Lexicon

Words that **increase** the score (epistemic hedges, uncertainty markers,
conditionality, analytical qualifiers):

`may` · `might` · `could` · `possibly` · `potentially` · `appears to` ·
`seems to` · `tends to` · `suggests` · `uncertain` · `unclear` ·
`subject to` · `risk` · `risks` · `volatility` · `fluctuation` ·
`variability` · `if` · `depending on` · `assuming` · `provided that` ·
`contingent` · `subject to change` · `historically` · `based on` ·
`according to` · `indicates` · `analysis suggests` · `data shows` ·
`evidence suggests`

## Speculative Word Lexicon

Words that **decrease** the score (weighted ×1.5):

`will definitely` · `will certainly` · `guaranteed` · `guarantee` ·
`without a doubt` · `absolutely will` · `100%` · `sure to` ·
`skyrocket` · `explode` · `moon` · `to the moon` · `unstoppable` ·
`massive gains` · `huge profits` · `get rich` · `can't lose` ·
`you must buy` · `buy now` · `sell immediately` · `don't miss` ·
`once in a lifetime` · `never been a better time` · `amazing` ·
`incredible returns` · `unbelievable` · `shocking`
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Reward Hacking — RLHF Financial Hedging", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
# Reward Hacking & KL Penalty

This page explains a key failure mode in RLHF training and how the
**KL divergence penalty** prevents the policy from gaming the reward signal.
            """
        )
        gr.Markdown(HACKING_MD)

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.queue(max_size=10)
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)
