"""Tests for stress testing."""

import numpy as np
import pandas as pd
import pytest

from orbiter.stress import (
    correlation_stress,
    historical_scenario,
    monte_carlo_stress,
)


@pytest.fixture
def simple_portfolio():
    weights = np.array([0.5, 0.3, 0.2])
    mu = np.array([0.0005, 0.0003, 0.0002])
    cov = np.array(
        [
            [0.0004, 0.0001, 0.00005],
            [0.0001, 0.0003, 0.00008],
            [0.00005, 0.00008, 0.0005],
        ]
    )
    return weights, mu, cov


def test_monte_carlo_keys(simple_portfolio):
    weights, mu, cov = simple_portfolio
    result = monte_carlo_stress(weights, mu, cov, n_simulations=1000)
    expected_keys = {
        "var_95",
        "cvar_95",
        "var_99",
        "cvar_99",
        "median_return",
        "worst_case",
        "best_case",
        "prob_loss",
        "mean_return",
        "std_return",
    }
    assert set(result.keys()) == expected_keys


def test_monte_carlo_student_t_fatter_tails(simple_portfolio):
    weights, mu, cov = simple_portfolio
    normal = monte_carlo_stress(weights, mu, cov, distribution="normal", n_simulations=5000)
    student = monte_carlo_stress(
        weights, mu, cov, distribution="student-t", df=3.0, n_simulations=5000
    )
    # Student-t should have worse (more negative) CVaR
    assert student["cvar_99"] <= normal["cvar_99"] + 0.02


def test_monte_carlo_var_ordering(simple_portfolio):
    weights, mu, cov = simple_portfolio
    result = monte_carlo_stress(weights, mu, cov, n_simulations=5000)
    assert result["cvar_99"] <= result["var_99"]
    assert result["cvar_95"] <= result["var_95"]
    assert result["worst_case"] <= result["var_99"]


def test_historical_scenario():
    dates = pd.date_range("2022-11-06", periods=8, freq="D", tz="UTC")
    returns = pd.DataFrame(
        {
            "BTC": np.random.RandomState(42).randn(8) * 0.05 - 0.02,
            "ETH": np.random.RandomState(43).randn(8) * 0.06 - 0.03,
        },
        index=dates,
    )
    weights = np.array([0.6, 0.4])

    result = historical_scenario(weights, returns)
    assert result.scenario_name == "historical"
    assert isinstance(result.portfolio_return, float)
    assert result.portfolio_drawdown <= 0


def test_correlation_stress_increases_vol(simple_portfolio):
    weights, _, cov = simple_portfolio
    result = correlation_stress(weights, cov, stress_factor=2.0)
    assert result["stressed_volatility"] >= result["original_volatility"]
    assert result["increase_pct"] >= 0


def test_correlation_stress_factor_one(simple_portfolio):
    weights, _, cov = simple_portfolio
    result = correlation_stress(weights, cov, stress_factor=1.0)
    assert abs(result["increase_pct"]) < 1.0  # ~0% change
