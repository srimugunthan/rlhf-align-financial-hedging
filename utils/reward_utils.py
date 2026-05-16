"""
Heuristic hedge scoring utility.

compute_hedge_score() acts as a proxy for human preference annotations.
In a production RLHF system this would be replaced by actual human labels,
but for this toy experiment it provides a clean, deterministic signal.
"""

import numpy as np


# ── Preference Lexicons ───────────────────────────────────────────────────────

HEDGE_WORDS = [
    # Epistemic hedges
    "may", "might", "could", "possibly", "potentially",
    "appears to", "seems to", "tends to", "suggests",
    # Uncertainty markers
    "uncertain", "unclear", "subject to", "risk", "risks",
    "volatility", "fluctuation", "variability",
    # Conditionality
    "if", "depending on", "assuming", "provided that",
    "contingent", "subject to change",
    # Analytical qualifiers
    "historically", "based on", "according to", "indicates",
    "analysis suggests", "data shows", "evidence suggests",
]

SPECULATIVE_WORDS = [
    # Overconfident predictions
    "will definitely", "will certainly", "guaranteed", "guarantee",
    "without a doubt", "absolutely will", "100%", "sure to",
    # Hype language
    "skyrocket", "explode", "moon", "to the moon", "unstoppable",
    "massive gains", "huge profits", "get rich", "can't lose",
    # Imperative speculation
    "you must buy", "buy now", "sell immediately", "don't miss",
    "once in a lifetime", "never been a better time",
    # Sensationalism
    "amazing", "incredible returns", "unbelievable", "shocking",
]


# ── Scoring Function ──────────────────────────────────────────────────────────

def compute_hedge_score(text: str) -> float:
    """
    Heuristic reward: measures the hedging quality of financial text.

    Score ∈ [0, 1]:
        1.0  →  well-hedged, cautious financial language  (preferred)
        0.0  →  speculative, overconfident language        (dispreferred)

    Scoring logic:
        raw = hedge_count - 1.5 * spec_count - caps_penalty - exclaim_penalty
        score = sigmoid(raw)

    This function is our proxy for human preference. In a real RLHF system,
    human annotators would provide pairwise labels instead. The reward MODEL
    (Bradley-Terry trained) learns to approximate this signal from pairs.

    Args:
        text: Financial text to evaluate.

    Returns:
        Float score in [0, 1].
    """
    text_lower = text.lower()

    if not text_lower.strip():
        return 0.5  # neutral for empty input

    hedge_count = sum(1 for h in HEDGE_WORDS if h in text_lower)
    spec_count = sum(1 for s in SPECULATIVE_WORDS if s in text_lower)

    # Penalise ALL-CAPS (common hype signal)
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    caps_penalty = caps_ratio * 2.0

    # Penalise excessive exclamation marks
    exclaim_penalty = min(text.count("!") * 0.3, 1.0)

    raw = hedge_count - (spec_count * 1.5) - caps_penalty - exclaim_penalty
    score = 1.0 / (1.0 + np.exp(-raw))  # sigmoid

    return float(score)


# ── Sanity Check ──────────────────────────────────────────────────────────────

def run_sanity_check() -> None:
    """
    Print a quick sanity check table to verify the scoring function
    correctly ranks hedged text above speculative text.
    """
    test_cases = [
        ("The stock may experience volatility depending on interest rate decisions.", "GOOD"),
        ("Based on historical data, earnings could potentially miss estimates.", "GOOD"),
        ("Analysis suggests the sector might face headwinds if inflation persists.", "GOOD"),
        ("This stock will DEFINITELY skyrocket! Guaranteed massive gains!", "BAD"),
        ("BUY NOW before it's too late! You can't lose! To the moon!!!", "BAD"),
        ("Get rich quick with this incredible investment opportunity!", "BAD"),
    ]

    print("Hedge Scoring Sanity Check")
    print("=" * 60)
    correct = 0
    for text, label in test_cases:
        score = compute_hedge_score(text)
        passed = (label == "GOOD" and score > 0.5) or (label == "BAD" and score < 0.5)
        flag = "✅" if passed else "❌"
        if passed:
            correct += 1
        short = text[:67] + "..." if len(text) > 70 else text
        print(f"{flag} [{label}] Score={score:.3f}  {short}")
    print(f"\nPassed {correct}/{len(test_cases)}")
