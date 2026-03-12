"""Tests for risk metrics."""

import pandas as pd

from orbiter.metrics import (
    annualized_return,
    annualized_volatility,
    calmar_ratio,
    compute_metrics,
    cvar,
    max_drawdown,
    omega_ratio,
    sharpe_ratio,
    sortino_ratio,
)


def test_annualized_return(sample_returns):
    ret = annualized_return(sample_returns["BTC"])
    assert isinstance(ret, float)
    assert -1.0 < ret < 10.0  # Reasonable range


def test_annualized_volatility(sample_returns):
    vol = annualized_volatility(sample_returns["BTC"])
    assert vol > 0
    assert vol < 5.0  # Crypto is volatile but not infinite


def test_sharpe_ratio(sample_returns):
    s = sharpe_ratio(sample_returns["BTC"])
    assert isinstance(s, float)
    assert -5.0 < s < 5.0


def test_sharpe_zero_vol():
    zero_returns = pd.Series([0.0] * 100)
    assert sharpe_ratio(zero_returns) == 0.0


def test_sortino_ratio(sample_returns):
    s = sortino_ratio(sample_returns["BTC"])
    assert isinstance(s, float)


def test_sortino_no_downside():
    positive = pd.Series([0.01] * 100)
    s = sortino_ratio(positive)
    assert s == float("inf")


def test_max_drawdown(sample_returns):
    mdd = max_drawdown(sample_returns["BTC"])
    assert mdd <= 0
    assert mdd >= -1.0


def test_max_drawdown_always_up():
    always_up = pd.Series([0.01] * 100)
    mdd = max_drawdown(always_up)
    assert mdd == 0.0 or abs(mdd) < 1e-10


def test_calmar_ratio(sample_returns):
    c = calmar_ratio(sample_returns["BTC"])
    assert isinstance(c, float)


def test_cvar(sample_returns):
    c = cvar(sample_returns["BTC"], alpha=0.05)
    assert c < 0  # Tail losses are negative
    mean_ret = sample_returns["BTC"].mean()
    assert c < mean_ret  # CVaR should be worse than mean


def test_omega_ratio(sample_returns):
    o = omega_ratio(sample_returns["BTC"])
    assert o > 0


def test_compute_metrics_keys(sample_returns):
    m = compute_metrics(sample_returns["BTC"])
    expected_keys = {
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "sortino_ratio",
        "max_drawdown",
        "calmar_ratio",
        "cvar_95",
        "omega_ratio",
    }
    assert set(m.keys()) == expected_keys


def test_compute_metrics_types(sample_returns):
    m = compute_metrics(sample_returns["BTC"])
    for key, value in m.items():
        assert isinstance(value, float), f"{key} is not float"


def test_empty_returns():
    empty = pd.Series([], dtype=float)
    assert annualized_return(empty) == 0.0
    assert cvar(empty) == 0.0
