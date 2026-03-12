"""Rebalancing simulation with configurable triggers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd

from orbiter.costs import FeeSchedule, compute_rebalance_cost
from orbiter.metrics import compute_metrics


class RebalanceTrigger(Enum):
    CALENDAR = "calendar"
    THRESHOLD = "threshold"
    HYBRID = "hybrid"


@dataclass
class RebalanceConfig:
    """Configuration for rebalancing behavior."""

    trigger: RebalanceTrigger = RebalanceTrigger.CALENDAR
    calendar_days: int = 30
    drift_threshold: float = 0.05  # 5% absolute drift from target
    fee_schedule: FeeSchedule | None = None


@dataclass
class RebalanceResult:
    """Results from a rebalancing simulation."""

    portfolio_returns: pd.Series
    rebalance_dates: list
    turnover_history: list[float] = field(default_factory=list)
    cost_history: list[float] = field(default_factory=list)
    weights_history: pd.DataFrame = field(default_factory=pd.DataFrame)
    total_turnover: float = 0.0
    total_cost: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.metrics and len(self.portfolio_returns) > 0:
            self.metrics = compute_metrics(self.portfolio_returns)


def check_drift(
    current_weights: np.ndarray,
    target_weights: np.ndarray,
    threshold: float,
) -> bool:
    """Returns True if any weight has drifted beyond the threshold."""
    return bool(np.max(np.abs(current_weights - target_weights)) > threshold)


def simulate_rebalancing(
    returns: pd.DataFrame,
    target_weights: np.ndarray,
    config: RebalanceConfig = RebalanceConfig(),
    initial_value: float = 10000.0,
    daily_volumes: pd.DataFrame | None = None,
) -> RebalanceResult:
    """Simulate a portfolio with configurable rebalancing.

    Tracks day-by-day portfolio value with weight drift and rebalancing.
    """
    n_days = len(returns)
    fee_schedule = config.fee_schedule or FeeSchedule(maker=0, taker=0, spread_bps=0)

    # Initialize
    current_weights = target_weights.copy()
    portfolio_value = initial_value
    portfolio_returns_list = []
    rebalance_dates = []
    turnover_history = []
    cost_history = []
    weights_records = []
    days_since_rebalance = 0

    for i in range(n_days):
        date = returns.index[i]
        day_returns = returns.iloc[i].values

        # Portfolio return for this day (before rebalancing)
        port_return = float(np.sum(current_weights * day_returns))
        portfolio_value *= np.exp(port_return)

        # Update weights based on asset returns (drift)
        drifted = current_weights * np.exp(day_returns)
        current_weights = drifted / drifted.sum()

        days_since_rebalance += 1

        # Check if we should rebalance
        should_rebalance = False
        if config.trigger == RebalanceTrigger.CALENDAR:
            should_rebalance = days_since_rebalance >= config.calendar_days
        elif config.trigger == RebalanceTrigger.THRESHOLD:
            should_rebalance = check_drift(current_weights, target_weights, config.drift_threshold)
        elif config.trigger == RebalanceTrigger.HYBRID:
            should_rebalance = days_since_rebalance >= config.calendar_days and check_drift(
                current_weights, target_weights, config.drift_threshold
            )

        # Apply rebalancing
        cost = 0.0
        if should_rebalance and i < n_days - 1:  # Don't rebalance on last day
            volumes = None
            if daily_volumes is not None:
                volumes = daily_volumes.iloc[i].values

            cost, turnover = compute_rebalance_cost(
                current_weights,
                target_weights,
                portfolio_value,
                daily_volumes=volumes,
                fee_schedule=fee_schedule,
            )

            portfolio_value -= cost
            current_weights = target_weights.copy()
            days_since_rebalance = 0

            rebalance_dates.append(date)
            turnover_history.append(turnover)
            cost_history.append(cost)

        # Adjust return for cost
        adjusted_return = port_return - (cost / portfolio_value if portfolio_value > 0 else 0)
        portfolio_returns_list.append(adjusted_return)

        weights_records.append(
            {"date": date, **{col: w for col, w in zip(returns.columns, current_weights)}}
        )

    portfolio_returns = pd.Series(portfolio_returns_list, index=returns.index, name="portfolio")
    weights_history = pd.DataFrame(weights_records).set_index("date")

    return RebalanceResult(
        portfolio_returns=portfolio_returns,
        rebalance_dates=rebalance_dates,
        turnover_history=turnover_history,
        cost_history=cost_history,
        weights_history=weights_history,
        total_turnover=sum(turnover_history),
        total_cost=sum(cost_history),
    )
