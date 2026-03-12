"""Tests for portfolio optimization."""

import numpy as np
import pytest
from orbiter.optimize import PortfolioOptimizer

# Strategies that work without extra dependencies (no factor model needed)
BASE_STRATEGIES = ["max-sharpe", "min-vol", "min-cvar", "risk-parity", "hrp", "regime-aware"]


@pytest.fixture
def optimizer(sample_returns):
    return PortfolioOptimizer(sample_returns)


class TestWeightConstraints:
    """All strategies must produce valid, long-only weights summing to 1."""

    @pytest.mark.parametrize("strategy", BASE_STRATEGIES)
    def test_weights_sum_to_one(self, optimizer, strategy):
        result = optimizer.optimize(strategy)
        np.testing.assert_almost_equal(result.weights.sum(), 1.0, decimal=5)

    @pytest.mark.parametrize("strategy", BASE_STRATEGIES)
    def test_weights_non_negative(self, optimizer, strategy):
        result = optimizer.optimize(strategy)
        assert (result.weights >= -1e-6).all(), f"Negative weights: {result.weights}"

    @pytest.mark.parametrize("strategy", BASE_STRATEGIES)
    def test_weights_have_correct_index(self, optimizer, strategy):
        result = optimizer.optimize(strategy)
        assert list(result.weights.index) == ["BTC", "ETH", "SOL", "AVAX"]


class TestStrategies:
    def test_max_sharpe_has_metrics(self, optimizer):
        result = optimizer.max_sharpe()
        assert "sharpe_ratio" in result.metrics
        assert result.strategy == "max-sharpe"

    def test_min_vol_has_lower_volatility(self, optimizer):
        min_vol = optimizer.min_volatility()
        max_sharpe = optimizer.max_sharpe()
        assert min_vol.metrics["annualized_volatility"] <= max_sharpe.metrics["annualized_volatility"] + 0.01

    def test_risk_parity_balanced_risk(self, optimizer):
        result = optimizer.risk_parity()
        # Risk parity should not put 100% in one asset
        assert result.weights.max() < 0.95

    def test_hrp_produces_diversified_weights(self, optimizer):
        result = optimizer.hrp()
        # HRP should spread across assets
        assert (result.weights > 0.01).sum() >= 2

    def test_min_cvar(self, optimizer):
        result = optimizer.min_cvar()
        assert "cvar_95" in result.metrics


class TestEfficientFrontier:
    def test_frontier_shape(self, optimizer):
        frontier = optimizer.efficient_frontier(n_points=20)
        assert len(frontier) > 0
        assert "return" in frontier.columns
        assert "volatility" in frontier.columns
        assert "sharpe" in frontier.columns

    def test_frontier_has_asset_weights(self, optimizer):
        frontier = optimizer.efficient_frontier(n_points=10)
        for asset in ["BTC", "ETH", "SOL", "AVAX"]:
            assert asset in frontier.columns


class TestFactorStrategy:
    def test_factor_max_sharpe_needs_model(self, optimizer):
        with pytest.raises(ValueError, match="factor_model"):
            optimizer.optimize("factor-max-sharpe")

    def test_factor_max_sharpe_with_model(self, sample_returns):
        from orbiter.factors import CryptoFactorModel

        fm = CryptoFactorModel(sample_returns)
        fm.fit()
        opt = PortfolioOptimizer(sample_returns, factor_model=fm)
        result = opt.optimize("factor-max-sharpe")
        np.testing.assert_almost_equal(result.weights.sum(), 1.0, decimal=5)
        assert result.strategy == "factor-max-sharpe"


class TestDispatcher:
    def test_optimize_dispatches(self, optimizer):
        for strategy in BASE_STRATEGIES:
            result = optimizer.optimize(strategy)
            assert result.strategy.startswith(strategy)

    def test_invalid_strategy(self, optimizer):
        with pytest.raises(ValueError, match="Unknown strategy"):
            optimizer.optimize("moon-strategy")
