"""
Financial language corpus for RLHF training.

Contains:
  - SFT_DATA        : sentences for supervised fine-tuning
  - PAIRWISE_DATA   : (preferred, dispreferred) pairs for reward model training
  - FINANCIAL_PROMPTS: short prompts used as PPO rollout seeds
"""

from typing import List, Tuple

# ── SFT Training Corpus ───────────────────────────────────────────────────────
# Mix of hedged (✅), speculative (❌), and neutral text.
# Diversity here ensures the base model knows both styles;
# the reward model + PPO will later steer toward hedged language.

SFT_DATA: List[str] = [
    # ── Hedged examples (target style) ───────────────────────────────────────
    "The company's revenue may decline if macroeconomic conditions worsen.",
    "Based on current data, earnings could potentially miss analyst estimates.",
    "The stock appears to be undervalued relative to historical price-to-earnings ratios.",
    "Market volatility suggests investors should consider diversifying their portfolios.",
    "Analysis indicates the sector might face headwinds given rising interest rates.",
    "The quarterly results could reflect ongoing supply chain disruptions.",
    "Depending on Federal Reserve policy, the bond market may experience fluctuations.",
    "Historical trends suggest that inflation tends to peak before a rate cycle ends.",
    "The merger appears contingent on regulatory approval from antitrust authorities.",
    "Subject to market conditions, the IPO could be delayed until Q2.",
    "Earnings growth may slow if consumer spending contracts in the coming quarters.",
    "The portfolio's risk exposure could increase if equity correlations rise sharply.",
    "Evidence suggests the company might need to restructure its debt obligations.",
    "The dividend yield appears sustainable, assuming cash flows remain stable.",
    "Current valuations may not fully reflect the risks of a potential recession.",

    # ── Speculative examples (penalized by RL, included for SFT diversity) ───
    "This stock will definitely hit $500 by end of year!",
    "Guaranteed returns await investors who buy now before prices explode.",
    "The market is going to absolutely skyrocket next quarter.",
    "You cannot lose money on this incredible investment opportunity.",
    "Massive gains are certain for early investors in this sector.",

    # ── Neutral financial domain text ─────────────────────────────────────────
    "The Federal Reserve announced its decision on interest rates yesterday.",
    "Q3 earnings reports are expected from major banks next week.",
    "The S&P 500 closed lower amid concerns about inflation data.",
    "Analysts revised their price targets following the earnings release.",
    "Trading volume increased significantly during the market open.",
]


# ── Pairwise Preference Data ──────────────────────────────────────────────────
# Each tuple is (preferred/hedged, dispreferred/speculative).
# In real RLHF, human annotators produce these labels.
# Here, we simulate annotation using the compute_hedge_score heuristic.

PAIRWISE_DATA: List[Tuple[str, str]] = [
    (
        "The stock may experience significant volatility if interest rates rise further.",
        "The stock will definitely skyrocket once interest rates change!",
    ),
    (
        "Based on historical data, earnings could potentially disappoint investors.",
        "Earnings will absolutely blow past expectations and guarantee huge gains.",
    ),
    (
        "The merger appears contingent on regulatory approval and may face delays.",
        "The merger will certainly go through and profits are guaranteed to explode.",
    ),
    (
        "Inflation data suggests the central bank might consider pausing rate hikes.",
        "The Fed will 100% cut rates causing an unstoppable rally in equities!",
    ),
    (
        "Consumer spending could weaken if unemployment rises in coming quarters.",
        "Consumer spending will definitely surge — buy now before massive gains!",
    ),
    (
        "The company's guidance seems cautious, possibly reflecting supply risks.",
        "Guidance will shock everyone! Incredible returns are absolutely guaranteed!",
    ),
    (
        "Portfolio risk may increase if correlations between asset classes rise.",
        "Your portfolio will skyrocket! This is a once-in-a-lifetime opportunity!",
    ),
    (
        "The sector appears to be facing headwinds from rising input costs.",
        "The sector will EXPLODE higher! Massive gains are coming for everyone!",
    ),
    (
        "Bond yields could rise further depending on inflation trajectory.",
        "Bond yields will definitely collapse! Don't miss out on amazing returns!",
    ),
    (
        "Analyst estimates suggest Q3 revenue might fall short of consensus.",
        "Revenue will certainly beat every estimate! Get rich quick on this trade!",
    ),
    (
        "The IPO valuation seems stretched relative to comparable companies.",
        "The IPO will make you rich! 100% gains are absolutely coming your way!",
    ),
    (
        "Dividend sustainability appears uncertain given declining free cash flow.",
        "Dividends will skyrocket! You can't lose money on this amazing stock!",
    ),
    (
        "The acquisition could create value if integration risks are managed well.",
        "This acquisition will definitely make billions! Guaranteed unbelievable returns!",
    ),
    (
        "Currency risk may dampen overseas earnings if the dollar strengthens.",
        "Currency moves will certainly boost earnings! Profits are guaranteed to soar!",
    ),
    (
        "The credit rating could be downgraded if debt levels continue to rise.",
        "Credit will absolutely be upgraded! Amazing gains await every investor!",
    ),
]


# ── Financial Prompt Pool ─────────────────────────────────────────────────────
# Short sentence starters used as PPO rollout seeds.
# The policy completes each prompt; completions are scored by the reward model.

FINANCIAL_PROMPTS: List[str] = [
    "The stock market",
    "Investors should consider",
    "Based on recent earnings",
    "The company's outlook",
    "Market analysis suggests",
    "The Federal Reserve",
    "Portfolio risk",
    "Quarterly results",
    "The sector may",
    "Bond yields",
    "Equity valuation",
    "The merger could",
    "Consumer spending",
    "Inflation data",
    "The IPO",
    "The company's financial",
    "Recent economic data",
    "Investment risk",
    "The market outlook",
    "Analysts believe",
    "The central bank",
    "Corporate earnings",
    "The fund's performance",
]


# ── Evaluation Prompts (held-out) ─────────────────────────────────────────────
EVAL_PROMPTS: List[str] = [
    "The stock market",
    "Investors should consider",
    "The company's outlook",
    "Market analysis suggests",
    "The Federal Reserve",
]


# ── Reward Hacking Demo Texts ─────────────────────────────────────────────────
REWARD_HACKING_EXAMPLES: List[Tuple[str, str]] = [
    (
        "May might could possibly potentially uncertain risk volatility if depending on contingent subject to change.",
        "HACKED (gamed)",
    ),
    (
        "The stock may could possibly might uncertain possibly may could depending on if.",
        "HACKED (gamed)",
    ),
    (
        "The stock may decline if inflation persists beyond the current quarter.",
        "GENUINE hedge",
    ),
    (
        "The stock rose sharply after earnings were released yesterday.",
        "No hedge",
    ),
]
