"""Tests for Black-Litterman model."""

import json

import numpy as np
import pandas as pd
import pytest

from orbiter.black_litterman import BlackLitterman, BLResult, View, parse_ai_views


@pytest.fixture
def bl_model(sample_returns):
    return BlackLitterman(sample_returns)


@pytest.fixture
def bl_model_with_caps(sample_returns):
    caps = pd.Series(
        [800e9, 300e9, 50e9, 10e9],
        index=sample_returns.columns,
    )
    return BlackLitterman(sample_returns, market_caps=caps)


class TestBlackLittermanInit:
    def test_init_without_market_caps(self, bl_model):
        assert bl_model.n_assets == 4
        assert list(bl_model.asset_names) == ["BTC", "ETH", "SOL", "AVAX"]
        np.testing.assert_almost_equal(bl_model.market_weights.sum(), 1.0)

    def test_init_with_market_caps(self, bl_model_with_caps):
        assert bl_model_with_caps.n_assets == 4
        np.testing.assert_almost_equal(bl_model_with_caps.market_weights.sum(), 1.0)
        # BTC should have the largest weight
        assert bl_model_with_caps.market_weights[0] > bl_model_with_caps.market_weights[-1]


class TestImpliedReturns:
    def test_shape(self, bl_model):
        pi = bl_model.implied_returns()
        assert len(pi) == 4
        assert list(pi.index) == ["BTC", "ETH", "SOL", "AVAX"]

    def test_returns_series(self, bl_model):
        pi = bl_model.implied_returns()
        assert isinstance(pi, pd.Series)
        assert pi.name == "implied_return"


class TestPosterior:
    def test_no_views_returns_equilibrium(self, bl_model):
        result = bl_model.posterior([])
        assert isinstance(result, BLResult)
        np.testing.assert_array_almost_equal(
            result.posterior_returns.values, bl_model.pi, decimal=8
        )
        assert result.views_used == []

    def test_absolute_view_shifts_returns(self, bl_model):
        equilibrium = bl_model.implied_returns()
        view = View(asset="SOL", return_view=0.50, confidence=0.8, source="test")
        result = bl_model.posterior([view])
        # SOL posterior should move toward the view (away from equilibrium)
        assert result.posterior_returns["SOL"] != equilibrium["SOL"]
        # Direction: view < equilibrium, so posterior should decrease
        assert result.posterior_returns["SOL"] < equilibrium["SOL"]

    def test_relative_view(self, bl_model):
        view = View(
            asset=("ETH", "BTC"), return_view=0.05, confidence=0.7, source="test"
        )
        result = bl_model.posterior([view])
        assert isinstance(result, BLResult)
        assert len(result.views_used) == 1

    def test_weights_sum_to_one(self, bl_model):
        view = View(asset="BTC", return_view=0.20, confidence=0.6)
        result = bl_model.posterior([view])
        np.testing.assert_almost_equal(result.weights.sum(), 1.0, decimal=5)

    def test_weights_non_negative(self, bl_model):
        view = View(asset="ETH", return_view=0.30, confidence=0.9)
        result = bl_model.posterior([view])
        assert (result.weights >= -1e-6).all()


class TestParseAIViews:
    def test_valid_json(self):
        assets = ["BTC", "ETH", "SOL", "AVAX"]
        response = json.dumps([
            {"asset": "SOL", "return": 0.15, "confidence": 0.7, "reasoning": "bullish"},
            {"asset": "BTC", "return": 0.10, "confidence": 0.5},
        ])
        views = parse_ai_views(response, assets)
        assert len(views) == 2
        assert views[0].asset == "SOL"
        assert views[0].return_view == 0.15
        assert views[0].source == "ai"

    def test_relative_views(self):
        assets = ["BTC", "ETH", "SOL", "AVAX"]
        response = json.dumps([
            {"asset": ["SOL", "ETH"], "return": 0.05, "confidence": 0.6},
        ])
        views = parse_ai_views(response, assets)
        assert len(views) == 1
        assert views[0].asset == ("SOL", "ETH")

    def test_invalid_json_returns_empty(self):
        views = parse_ai_views("not json at all", ["BTC", "ETH"])
        assert views == []

    def test_strips_markdown_code_fences(self):
        assets = ["BTC", "ETH"]
        inner = json.dumps([{"asset": "BTC", "return": 0.1, "confidence": 0.5}])
        response = f"```json\n{inner}\n```"
        views = parse_ai_views(response, assets)
        assert len(views) == 1
        assert views[0].asset == "BTC"

    def test_unknown_asset_skipped(self):
        assets = ["BTC", "ETH"]
        response = json.dumps([
            {"asset": "DOGE", "return": 0.5, "confidence": 0.9},
        ])
        views = parse_ai_views(response, assets)
        assert views == []
