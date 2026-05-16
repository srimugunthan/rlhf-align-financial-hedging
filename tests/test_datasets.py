"""
Unit tests for dataset builders.
Run: pytest tests/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import torch
from unittest.mock import MagicMock, patch


class TestPreferenceDataset:

    def _make_tokenizer(self):
        """Minimal mock tokenizer for dataset tests."""
        tok = MagicMock()
        tok.side_effect = lambda text, **kwargs: {
            "input_ids": torch.ones(1, 128, dtype=torch.long),
            "attention_mask": torch.ones(1, 128, dtype=torch.long),
        }
        return tok

    def test_len(self):
        from data.datasets import PreferenceDataset
        tok = self._make_tokenizer()
        pairs = [("good text", "bad text"), ("also good", "also bad")]
        ds = PreferenceDataset(pairs, tok)
        assert len(ds) == 2

    def test_item_keys(self):
        from data.datasets import PreferenceDataset
        tok = self._make_tokenizer()
        pairs = [("good text", "bad text")]
        ds = PreferenceDataset(pairs, tok)
        item = ds[0]
        assert "input_ids_w" in item
        assert "attention_mask_w" in item
        assert "input_ids_l" in item
        assert "attention_mask_l" in item

    def test_item_shapes(self):
        from data.datasets import PreferenceDataset
        tok = self._make_tokenizer()
        pairs = [("good text", "bad text")]
        ds = PreferenceDataset(pairs, tok, max_length=64)
        item = ds[0]
        # Squeezed — should be 1D tensors of length max_length
        assert item["input_ids_w"].shape == torch.Size([128])


class TestCorpus:

    def test_sft_data_nonempty(self):
        from data.corpus import SFT_DATA
        assert len(SFT_DATA) > 0
        assert all(isinstance(s, str) for s in SFT_DATA)

    def test_pairwise_data_structure(self):
        from data.corpus import PAIRWISE_DATA
        assert len(PAIRWISE_DATA) > 0
        for item in PAIRWISE_DATA:
            assert len(item) == 2
            preferred, dispreferred = item
            assert isinstance(preferred, str)
            assert isinstance(dispreferred, str)
            assert preferred != dispreferred

    def test_financial_prompts_nonempty(self):
        from data.corpus import FINANCIAL_PROMPTS
        assert len(FINANCIAL_PROMPTS) > 0
        assert all(isinstance(p, str) for p in FINANCIAL_PROMPTS)

    def test_pairwise_heuristic_agreement(self):
        """At least 80% of pairs should be correctly ordered by the heuristic."""
        from data.corpus import PAIRWISE_DATA
        from utils.reward_utils import compute_hedge_score
        correct = sum(
            1 for w, l in PAIRWISE_DATA
            if compute_hedge_score(w) > compute_hedge_score(l)
        )
        ratio = correct / len(PAIRWISE_DATA)
        assert ratio >= 0.8, f"Heuristic agreement {ratio:.0%} < 80%"
