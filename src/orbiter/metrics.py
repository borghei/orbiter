"""Risk and performance metrics for portfolio analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_return(returns: pd.Series, periods_per_year: int = 365) -> float:
    """Geometric annualized return from log returns."""
    total_log_return = returns.sum()
    n_periods = len(returns)
    if n_periods == 0:
        return 0.0
    return float(np.exp(total_log_return * periods_per_year / n_periods) - 1)


def annualized_volatility(returns: pd.Series, periods_per_year: int = 365) -> float:
    """Annualized volatility (standard deviation of returns)."""
    return float(returns.std() * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 365,
) -> float:
    """Annualized Sharpe ratio."""
    ann_ret = annualized_return(returns, periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year)
    if ann_vol == 0:
        return 0.0
    return float((ann_ret - risk_free_rate) / ann_vol)


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 365,
) -> float:
    """Annualized Sortino ratio (downside deviation only)."""
    ann_ret = annualized_return(returns, periods_per_year)
    downside = returns[returns < 0]
    if len(downside) == 0:
        return float("inf") if ann_ret > risk_free_rate else 0.0
    downside_vol = float(downside.std() * np.sqrt(periods_per_year))
    if downside_vol == 0:
        return 0.0
    return float((ann_ret - risk_free_rate) / downside_vol)


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown from a log returns series. Returns a negative number."""
    cumulative = returns.cumsum().apply(np.exp)
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series, periods_per_year: int = 365) -> float:
    """Calmar ratio: annualized return / |max drawdown|."""
    mdd = max_drawdown(returns)
    if mdd == 0:
        return 0.0
    ann_ret = annualized_return(returns, periods_per_year)
    return float(ann_ret / abs(mdd))


def cvar(returns: pd.Series, alpha: float = 0.05) -> float:
    """Historical Conditional Value-at-Risk (Expected Shortfall).

    Returns the mean of the worst alpha% of returns (a negative number).
    """
    if len(returns) == 0:
        return 0.0
    cutoff = returns.quantile(alpha)
    tail = returns[returns <= cutoff]
    if len(tail) == 0:
        return float(cutoff)
    return float(tail.mean())


def omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
    """Omega ratio: probability-weighted gains over losses."""
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns <= threshold]
    if losses.sum() == 0:
        return float("inf") if gains.sum() > 0 else 0.0
    return float(gains.sum() / losses.sum())


def compute_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 365,
) -> dict[str, float]:
    """Compute all portfolio metrics at once."""
    return {
        "annualized_return": annualized_return(returns, periods_per_year),
        "annualized_volatility": annualized_volatility(returns, periods_per_year),
        "sharpe_ratio": sharpe_ratio(returns, risk_free_rate, periods_per_year),
        "sortino_ratio": sortino_ratio(returns, risk_free_rate, periods_per_year),
        "max_drawdown": max_drawdown(returns),
        "calmar_ratio": calmar_ratio(returns, periods_per_year),
        "cvar_95": cvar(returns, 0.05),
        "omega_ratio": omega_ratio(returns),
    }
