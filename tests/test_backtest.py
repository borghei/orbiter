"""Tests for walk-forward backtesting."""

import pytest

from orbiter.backtest import WalkForwardBacktest


def test_backtest_runs(sample_returns):
    bt = WalkForwardBacktest(
        sample_returns,
        train_days=90,
        test_days=30,
        strategy="max-sharpe",
    )
    result = bt.run()
    assert len(result.portfolio_returns) > 0
    assert len(result.rebalance_dates) > 0


def test_backtest_weights_history(sample_returns):
    bt = WalkForwardBacktest(
        sample_returns,
        train_days=90,
        test_days=30,
        strategy="hrp",
    )
    result = bt.run()
    assert not result.weights_history.empty
    assert set(result.weights_history.columns) == {"BTC", "ETH", "SOL", "AVAX"}


def test_backtest_metrics_computed(sample_returns):
    bt = WalkForwardBacktest(
        sample_returns,
        train_days=90,
        test_days=30,
    )
    result = bt.run()
    assert "sharpe_ratio" in result.metrics
    assert "max_drawdown" in result.metrics


def test_backtest_insufficient_data(sample_returns):
    bt = WalkForwardBacktest(
        sample_returns.iloc[:50],  # Only 50 days
        train_days=180,
        test_days=30,
    )
    with pytest.raises(ValueError, match="Need at least"):
        bt.run()


def test_backtest_multiple_strategies(sample_returns):
    for strategy in ["max-sharpe", "min-vol", "hrp", "risk-parity"]:
        bt = WalkForwardBacktest(
            sample_returns,
            train_days=90,
            test_days=30,
            strategy=strategy,
        )
        result = bt.run()
        assert result.strategy == strategy
        assert len(result.portfolio_returns) > 0
