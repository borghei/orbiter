"""Covariance matrix estimation methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def sample_covariance(returns: pd.DataFrame, periods_per_year: int = 365) -> np.ndarray:
    """Sample covariance matrix, annualized."""
    return returns.cov().values * periods_per_year


def ledoit_wolf(returns: pd.DataFrame, periods_per_year: int = 365) -> np.ndarray:
    """Ledoit-Wolf shrinkage covariance estimator, annualized."""
    lw = LedoitWolf().fit(returns.values)
    return lw.covariance_ * periods_per_year


def exponential_covariance(
    returns: pd.DataFrame,
    span: int = 60,
    periods_per_year: int = 365,
) -> np.ndarray:
    """Exponentially weighted covariance matrix, annualized.

    Gives more weight to recent observations.
    """
    ewm_cov = returns.ewm(span=span).cov()
    last_date = returns.index[-1]
    cov_matrix = ewm_cov.loc[last_date].values
    return cov_matrix * periods_per_year


def get_covariance(
    returns: pd.DataFrame,
    method: str = "ledoit-wolf",
    periods_per_year: int = 365,
    **kwargs,
) -> np.ndarray:
    """Compute covariance matrix using the specified method.

    Args:
        returns: DataFrame of log returns.
        method: One of 'sample', 'ledoit-wolf', 'exponential'.
        periods_per_year: Annualization factor (365 for crypto).

    Returns:
        Annualized covariance matrix as numpy array.
    """
    methods = {
        "sample": lambda: sample_covariance(returns, periods_per_year),
        "ledoit-wolf": lambda: ledoit_wolf(returns, periods_per_year),
        "exponential": lambda: exponential_covariance(
            returns, kwargs.get("span", 60), periods_per_year
        ),
    }
    if method not in methods:
        raise ValueError(f"Unknown method '{method}'. Choose from: {list(methods.keys())}")
    return methods[method]()
