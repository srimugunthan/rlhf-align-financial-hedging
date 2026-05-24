# Test Prompts — SFT vs PPO Model Comparison

Use these prompts in the **Compare Models** tab of the Gradio demo (`demo/app.py`).
They are organised by category to reveal the contrast between the SFT and PPO models clearly.

---

## 🟢 Category 1 — Macroeconomic Events
*These are in the training distribution — expect PPO to hedge cleanly.*

| Prompt | What to watch for |
|---|---|
| `The Federal Reserve` | PPO should add uncertainty qualifiers (`may`, `could`) around rate decisions; SFT may assert outcomes directly |
| `Inflation data` | PPO should mention `suggests`, `might`, `subject to`; SFT may make confident predictions |
| `The central bank's decision` | PPO should use `appears`, `contingent`, `depending on` |
| `Recent economic data` | Good test — neutral start, see if PPO hedges the completion more than SFT |

---

## 🟡 Category 2 — Company / Earnings
*Tests whether PPO learned to hedge company-specific language.*

| Prompt | What to watch for |
|---|---|
| `The company's outlook` | PPO should produce cautious forward-looking language; SFT may assert guidance confidently |
| `Based on recent earnings` | PPO should hedge with `could`, `potentially`, `analysis suggests`; SFT may state outcomes |
| `Quarterly results` | Compare tone — PPO should stay analytical, SFT may drift speculative |
| `Corporate earnings` | A short, ambiguous seed — good for exposing policy divergence |
| `The merger could` | Prompt already starts hedged; see if PPO extends it cleanly or SFT breaks the hedge |

---

## 🔴 Category 3 — Market Sentiment (Higher Risk of Speculation)
*These are closer to out-of-distribution speculation triggers — reveals reward hacking risk.*

| Prompt | What to watch for |
|---|---|
| `The stock market` | Very open-ended; SFT may drift toward hype; PPO should stay cautious |
| `The IPO` | SFT may make overconfident valuation claims; PPO should use `appears stretched`, `could` |
| `Equity valuation` | Tests whether PPO produces `relative to`, `historically`, `may not reflect` |
| `Investment risk` | PPO should produce diversification/uncertainty language; SFT may underplay risk |

---

## 🔵 Category 4 — Longer Prompts (Stress Test)
*Paste these full sentences — more context reveals whether completions are coherent or degenerate.*

```
Investors should consider the impact of rising interest rates on
```
```
Market analysis suggests that the current environment for
```
```
The fund's performance over the next quarter will depend on
```
```
Analysts believe the sector could face headwinds if
```

---

## 🧪 Category 5 — Adversarial / Edge Cases
*For testing the KL penalty and reward hacking.*

| Prompt | Purpose |
|---|---|
| `Bond yields` | Very short — completion is almost entirely model-driven |
| `Portfolio risk` | PPO should increase hedging here; a good signal-to-noise test |
| `The sector may` | Starts with a hedge word — does SFT undo it? Does PPO extend it? |
| `Consumer spending` | Neutral domain term — watch for overconfident vs. cautious completions |

---

## 📊 What a Good PPO Improvement Looks Like

| Signal | SFT (before RLHF) | PPO (after RLHF) |
|---|---|---|
| Score delta | Baseline | **+0.05 to +0.20** improvement typical |
| Hedge words | Few or none | `may`, `could`, `potentially`, `suggests`, `depending on` |
| Speculative words | Possible | Rare or absent |
| ALL-CAPS / `!` | Occasional | Should be absent |
| Tone | Assertive or neutral | Cautious, analytical |

> **Tip:** If PPO scores *lower* than SFT on some prompts, that is expected — the policy hasn't converged perfectly on every input with only 30 PPO steps. Increase `PPOConfig.num_steps` to `100–200` in `configs/config.py` (line 53) for a more robust policy.

---

## 🧮 Hedge Score Analyser — Example Inputs

Use these in the **Hedge Score Analyser** tab to verify the scoring function and explore what drives scores up or down.

| Text | Score | Label |
|---|---|---|
| `"The stock may experience volatility depending on interest rate decisions."` | 0.88 | ✅ Hedged |
| `"Based on historical data, earnings could potentially miss estimates."` | 0.85 | ✅ Hedged |
| `"This stock will DEFINITELY skyrocket! Guaranteed massive gains!"` | 0.08 | ⚠️ Speculative |
| `"BUY NOW before it's too late! You can't lose! To the moon!!!"` | 0.03 | ⚠️ Speculative |
