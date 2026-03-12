"""Walk-forward backtesting engine."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from orbiter.metrics import compute_metrics
from orbiter.optimize import PortfolioOptimizer


@dataclass
class BacktestResult:
    portfolio_returns: pd.Series
    weights_history: pd.DataFrame
    rebalance_dates: list
    metrics: dict[str, float] = field(default_factory=dict)
    strategy: str = ""

    def __post_init__(self):
        if not self.metrics and len(self.portfolio_returns) > 0:
            self.metrics = compute_metrics(self.portfolio_returns)


class WalkForwardBacktest:
    """Rolling walk-forward backtest with periodic rebalancing.

    1. Train on `train_days` of history -> optimize weights
    2. Apply weights to the next `test_days` (out-of-sample)
    3. Slide forward by `test_days`, repeat
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        train_days: int = 180,
        test_days: int = 30,
        strategy: str = "max-sharpe",
        cov_method: str = "ledoit-wolf",
    ):
        self.returns = returns
        self.train_days = train_days
        self.test_days = test_days
        self.strategy = strategy
        self.cov_method = cov_method

    def run(self) -> BacktestResult:
        """Execute the walk-forward backtest."""
        total_days = len(self.returns)
        min_required = self.train_days + self.test_days

        if total_days < min_required:
            raise ValueError(
                f"Need at least {min_required} days of data, got {total_days}."
            )

        oos_returns_list: list[pd.Series] = []
        weights_records: list[dict] = []
        rebalance_dates: list = []

        start = 0
        while start + self.train_days + self.test_days <= total_days:
            train_end = start + self.train_days
            test_end = min(train_end + self.test_days, total_days)

            train_data = self.returns.iloc[start:train_end]
            test_data = self.returns.iloc[train_end:test_end]

            optimizer = PortfolioOptimizer(
                train_data,
                cov_method=self.cov_method,
            )
            result = optimizer.optimize(self.strategy)
            weights = result.weights.values

            oos_period_returns = (test_data * weights).sum(axis=1)
            oos_returns_list.append(oos_period_returns)

            rebalance_date = self.returns.index[train_end]
            rebalance_dates.append(rebalance_date)
            record = {"date": rebalance_date}
            record.update(result.weights.to_dict())
            weights_records.append(record)

            start += self.test_days

        portfolio_returns = pd.concat(oos_returns_list)
        weights_history = pd.DataFrame(weights_records).set_index("date")

        return BacktestResult(
            portfolio_returns=portfolio_returns,
            weights_history=weights_history,
            rebalance_dates=rebalance_dates,
            strategy=self.strategy,
        )
