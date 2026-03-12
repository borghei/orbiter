"""Tests for transaction cost modeling."""

import numpy as np
import pytest
from orbiter.costs import FeeSchedule, compute_rebalance_cost, estimate_slippage


def test_zero_turnover_zero_cost():
    weights = np.array([0.5, 0.3, 0.2])
    cost, turnover = compute_rebalance_cost(weights, weights, 10000.0)
    assert cost == 0.0
    assert turnover == 0.0


def test_slippage_increases_with_trade_size():
    small = estimate_slippage(1000, 1_000_000)
    large = estimate_slippage(100_000, 1_000_000)
    assert large > small


def test_slippage_zero_volume():
    assert estimate_slippage(1000, 0) == 0.0
    assert estimate_slippage(0, 1_000_000) == 0.0


def test_full_rebalance_cost():
    old = np.array([1.0, 0.0])
    new = np.array([0.0, 1.0])
    fee = FeeSchedule(maker=0.001, taker=0.001, spread_bps=0)
    cost, turnover = compute_rebalance_cost(old, new, 10000.0, fee_schedule=fee)
    # Turnover = |1-0| + |0-1| = 2.0
    assert turnover == 2.0
    # Cost = 2 * 10000 * 0.001 = 20
    assert abs(cost - 20.0) < 0.01


def test_cost_with_slippage():
    old = np.array([0.5, 0.5])
    new = np.array([1.0, 0.0])
    volumes = np.array([1_000_000.0, 500_000.0])
    cost, _ = compute_rebalance_cost(old, new, 10000.0, volumes)
    cost_no_vol, _ = compute_rebalance_cost(old, new, 10000.0)
    assert cost > cost_no_vol


def test_fee_schedule_effective():
    fee = FeeSchedule(maker=0.001, taker=0.003)
    assert fee.effective_fee == 0.002
