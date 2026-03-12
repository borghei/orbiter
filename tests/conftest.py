"""Shared test fixtures with realistic crypto-like synthetic data."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_returns():
    """Synthetic daily log returns for 4 crypto assets over 365 days."""
    rng = np.random.RandomState(42)
    n_days = 365
    n_assets = 4
    names = ["BTC", "ETH", "SOL", "AVAX"]

    # Realistic crypto parameters
    daily_drift = np.array([0.0003, 0.0004, 0.0006, 0.0005])
    daily_vol = np.array([0.03, 0.04, 0.06, 0.055])

    # Correlated returns via Cholesky
    corr = np.array([
        [1.0, 0.7, 0.5, 0.4],
        [0.7, 1.0, 0.6, 0.5],
        [0.5, 0.6, 1.0, 0.6],
        [0.4, 0.5, 0.6, 1.0],
    ])
    L = np.linalg.cholesky(corr)

    raw = rng.randn(n_days, n_assets)
    correlated = raw @ L.T
    returns_data = correlated * daily_vol + daily_drift

    dates = pd.date_range("2024-01-01", periods=n_days, freq="D", tz="UTC")
    return pd.DataFrame(returns_data, index=dates, columns=names)


@pytest.fixture
def sample_prices(sample_returns):
    """Synthetic price series derived from sample returns."""
    initial_prices = [40000, 2500, 100, 30]
    log_cum = sample_returns.cumsum()
    prices = np.exp(log_cum) * initial_prices
    return prices
