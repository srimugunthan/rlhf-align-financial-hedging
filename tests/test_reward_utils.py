"""
Unit tests for the heuristic hedge scoring utility.
Run: pytest tests/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from utils.reward_utils import compute_hedge_score


class TestComputeHedgeScore:

    def test_hedged_text_scores_above_half(self):
        hedged = [
            "The stock may decline if economic conditions worsen.",
            "Based on data, earnings could potentially miss estimates.",
            "Analysis suggests the sector might face headwinds.",
            "The merger appears contingent on regulatory approval.",
        ]
        for text in hedged:
            assert compute_hedge_score(text) > 0.5, f"Expected >0.5 for: {text}"

    def test_speculative_text_scores_below_half(self):
        speculative = [
            "This stock will DEFINITELY skyrocket! Guaranteed massive gains!",
            "BUY NOW! You can't lose! To the moon!!!",
            "Get rich quick with this incredible investment!",
            "Guaranteed returns! 100% profit! Amazing opportunity!",
        ]
        for text in speculative:
            assert compute_hedge_score(text) < 0.5, f"Expected <0.5 for: {text}"

    def test_empty_string_returns_neutral(self):
        assert compute_hedge_score("") == 0.5

    def test_score_in_unit_interval(self):
        texts = [
            "may might could possibly",
            "will definitely skyrocket guaranteed",
            "The Federal Reserve announced rates.",
            "",
        ]
        for text in texts:
            s = compute_hedge_score(text)
            assert 0.0 <= s <= 1.0, f"Score {s} out of [0,1] for: {text}"

    def test_hedged_beats_speculative_pairwise(self):
        pairs = [
            (
                "The stock may experience volatility if rates rise.",
                "The stock will definitely skyrocket!",
            ),
            (
                "Earnings could potentially miss estimates.",
                "Earnings are guaranteed to explode!",
            ),
        ]
        for preferred, dispreferred in pairs:
            assert compute_hedge_score(preferred) > compute_hedge_score(dispreferred)

    def test_caps_penalty_applied(self):
        normal = "The stock may decline."
        shouted = "THE STOCK MAY DECLINE."
        assert compute_hedge_score(normal) >= compute_hedge_score(shouted)

    def test_exclamation_penalty_applied(self):
        calm = "The stock may decline if conditions worsen."
        excited = "The stock may decline if conditions worsen!!!"
        assert compute_hedge_score(calm) >= compute_hedge_score(excited)
