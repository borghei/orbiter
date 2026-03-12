"""Tests for rebalancing simulation."""

import numpy as np
import pandas as pd

from orbiter.costs import FeeSchedule
from orbiter.rebalance import (
    RebalanceConfig,
    RebalanceTrigger,
    check_drift,
    simulate_rebalancing,
)


def test_check_drift_below_threshold():
    current = np.array([0.48, 0.52])
    target = np.array([0.50, 0.50])
    assert not check_drift(current, target, 0.05)


def test_check_drift_above_threshold():
    current = np.array([0.40, 0.60])
    target = np.array([0.50, 0.50])
    assert check_drift(current, target, 0.05)


def test_calendar_rebalance(sample_returns):
    config = RebalanceConfig(
        trigger=RebalanceTrigger.CALENDAR,
        calendar_days=30,
    )
    weights = np.ones(4) / 4
    result = simulate_rebalancing(sample_returns, weights, config)
    assert len(result.portfolio_returns) == len(sample_returns)
    # Should rebalance roughly every 30 days
    expected_rebalances = len(sample_returns) // 30
    assert abs(len(result.rebalance_dates) - expected_rebalances) <= 2


def test_threshold_no_rebalance_flat():
    """Flat returns should not trigger threshold rebalancing."""
    dates = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")
    flat = pd.DataFrame(
        np.zeros((100, 2)),
        index=dates,
        columns=["A", "B"],
    )
    config = RebalanceConfig(
        trigger=RebalanceTrigger.THRESHOLD,
        drift_threshold=0.05,
    )
    result = simulate_rebalancing(flat, np.array([0.5, 0.5]), config)
    assert len(result.rebalance_dates) == 0


def test_turnover_tracking(sample_returns):
    config = RebalanceConfig(
        trigger=RebalanceTrigger.CALENDAR,
        calendar_days=30,
    )
    weights = np.ones(4) / 4
    result = simulate_rebalancing(sample_returns, weights, config)
    assert len(result.turnover_history) == len(result.rebalance_dates)
    assert result.total_turnover >= 0


def test_cost_reduces_returns(sample_returns):
    weights = np.ones(4) / 4

    # Without costs
    config_free = RebalanceConfig(
        trigger=RebalanceTrigger.CALENDAR,
        calendar_days=30,
        fee_schedule=FeeSchedule(maker=0, taker=0, spread_bps=0),
    )
    result_free = simulate_rebalancing(sample_returns, weights, config_free)

    # With costs
    config_costly = RebalanceConfig(
        trigger=RebalanceTrigger.CALENDAR,
        calendar_days=30,
        fee_schedule=FeeSchedule(maker=0.01, taker=0.01, spread_bps=10),
    )
    result_costly = simulate_rebalancing(sample_returns, weights, config_costly)

    assert result_costly.total_cost > 0
    assert result_costly.portfolio_returns.sum() <= result_free.portfolio_returns.sum()


def test_weights_history_shape(sample_returns):
    config = RebalanceConfig(trigger=RebalanceTrigger.CALENDAR, calendar_days=30)
    weights = np.ones(4) / 4
    result = simulate_rebalancing(sample_returns, weights, config)
    assert len(result.weights_history) == len(sample_returns)
    assert set(result.weights_history.columns) == set(sample_returns.columns)
