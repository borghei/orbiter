"""Stress testing — Monte Carlo, historical scenarios, correlation stress."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class StressResult:
    """Result from a stress test scenario."""

    scenario_name: str
    portfolio_return: float
    portfolio_drawdown: float
    var_95: float
    cvar_95: float
    weights: pd.Series | None = None


# Historical stress scenarios: (start_date, end_date)
HISTORICAL_SCENARIOS: dict[str, tuple[str, str]] = {
    "covid_crash": ("2020-02-20", "2020-03-23"),
    "ftx_collapse": ("2022-11-06", "2022-11-14"),
    "luna_crash": ("2022-05-05", "2022-05-13"),
    "china_ban_2021": ("2021-05-12", "2021-05-23"),
    "bear_market_2022": ("2021-11-10", "2022-06-18"),
}


def monte_carlo_stress(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    n_simulations: int = 10000,
    horizon_days: int = 30,
    distribution: str = "student-t",
    df: float = 5.0,
    seed: int = 42,
) -> dict[str, float]:
    """Monte Carlo VaR/CVaR with configurable distribution.

    Args:
        weights: Portfolio weights.
        mu: Expected daily returns per asset.
        cov: Daily covariance matrix.
        n_simulations: Number of simulation paths.
        horizon_days: Simulation horizon in days.
        distribution: 'normal' or 'student-t'.
        df: Degrees of freedom for Student-t (lower = fatter tails).
        seed: Random seed.

    Returns:
        Dict with var_95, cvar_95, var_99, cvar_99, median_return,
        worst_case, best_case, prob_loss.
    """
    rng = np.random.RandomState(seed)
    n_assets = len(weights)

    # Cholesky decomposition for correlated samples
    L = np.linalg.cholesky(cov)

    terminal_returns = np.zeros(n_simulations)

    for sim in range(n_simulations):
        cumulative = 0.0
        for day in range(horizon_days):
            if distribution == "student-t":
                # Multivariate Student-t via normal/chi-squared mixture
                z = rng.randn(n_assets)
                chi2 = rng.chisquare(df)
                z = z * np.sqrt(df / chi2)
            else:
                z = rng.randn(n_assets)

            daily_returns = mu + L @ z
            port_return = float(weights @ daily_returns)
            cumulative += port_return

        terminal_returns[sim] = cumulative

    terminal_returns.sort()

    var_95_idx = int(0.05 * n_simulations)
    var_99_idx = int(0.01 * n_simulations)

    return {
        "var_95": float(terminal_returns[var_95_idx]),
        "cvar_95": float(terminal_returns[:var_95_idx].mean()) if var_95_idx > 0 else float(terminal_returns[0]),
        "var_99": float(terminal_returns[var_99_idx]),
        "cvar_99": float(terminal_returns[:var_99_idx].mean()) if var_99_idx > 0 else float(terminal_returns[0]),
        "median_return": float(np.median(terminal_returns)),
        "worst_case": float(terminal_returns[0]),
        "best_case": float(terminal_returns[-1]),
        "prob_loss": float(np.mean(terminal_returns < 0)),
        "mean_return": float(terminal_returns.mean()),
        "std_return": float(terminal_returns.std()),
    }


def historical_scenario(
    weights: np.ndarray,
    scenario_returns: pd.DataFrame,
) -> StressResult:
    """Run portfolio through a historical scenario.

    Args:
        weights: Portfolio weights.
        scenario_returns: DataFrame of log returns during the scenario period.

    Returns:
        StressResult with portfolio performance during the scenario.
    """
    port_returns = (scenario_returns * weights).sum(axis=1)

    # Cumulative return
    total_return = float(np.exp(port_returns.sum()) - 1)

    # Drawdown
    cumulative = port_returns.cumsum().apply(np.exp)
    running_max = cumulative.cummax()
    drawdown = ((cumulative - running_max) / running_max).min()

    # VaR/CVaR from the scenario daily returns
    sorted_returns = port_returns.sort_values()
    n = len(sorted_returns)
    var_idx = max(1, int(0.05 * n))
    var_95 = float(sorted_returns.iloc[var_idx - 1]) if n > 0 else 0.0
    cvar_95 = float(sorted_returns.iloc[:var_idx].mean()) if var_idx > 0 else var_95

    return StressResult(
        scenario_name="historical",
        portfolio_return=total_return,
        portfolio_drawdown=float(drawdown),
        var_95=var_95,
        cvar_95=cvar_95,
    )


def correlation_stress(
    weights: np.ndarray,
    cov: np.ndarray,
    stress_factor: float = 1.5,
) -> dict[str, float]:
    """Stress test by increasing correlations toward 1.0.

    Args:
        weights: Portfolio weights.
        cov: Original covariance matrix.
        stress_factor: Factor to increase off-diagonal correlations.
            1.0 = no change, 2.0 = double the correlations (capped at 1.0).

    Returns:
        Dict with original_volatility, stressed_volatility, increase_pct.
    """
    # Convert to correlation matrix
    vols = np.sqrt(np.diag(cov))
    vol_outer = np.outer(vols, vols)
    corr = cov / vol_outer

    # Stress correlations
    np.fill_diagonal(corr, 0)
    stressed_corr = corr * stress_factor
    stressed_corr = np.clip(stressed_corr, -1.0, 1.0)
    np.fill_diagonal(stressed_corr, 1.0)

    # Ensure positive semi-definite via eigenvalue clipping
    eigenvalues, eigenvectors = np.linalg.eigh(stressed_corr)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    stressed_corr = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    # Convert back to covariance
    stressed_cov = stressed_corr * vol_outer

    original_vol = float(np.sqrt(weights @ cov @ weights))
    stressed_vol = float(np.sqrt(weights @ stressed_cov @ weights))

    increase_pct = (stressed_vol - original_vol) / original_vol * 100 if original_vol > 0 else 0

    return {
        "original_volatility": original_vol,
        "stressed_volatility": stressed_vol,
        "increase_pct": increase_pct,
        "stress_factor": stress_factor,
    }
