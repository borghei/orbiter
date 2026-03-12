"""Transaction cost modeling — fees, slippage, and rebalance costs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FeeSchedule:
    """Exchange fee schedule."""

    maker: float = 0.001  # 10 bps
    taker: float = 0.001  # 10 bps
    spread_bps: float = 1.0  # typical bid-ask spread in basis points

    @property
    def effective_fee(self) -> float:
        """Average of maker and taker fees."""
        return (self.maker + self.taker) / 2


def estimate_slippage(
    trade_value_usd: float,
    daily_volume_usd: float,
    impact_coeff: float = 0.1,
    impact_exp: float = 0.5,
) -> float:
    """Square-root market impact model.

    slippage = impact_coeff * (trade_value / daily_volume) ^ impact_exp

    Based on the Almgren-Chriss market impact model, simplified.
    Returns slippage as a fraction (e.g., 0.002 = 0.2%).
    """
    if daily_volume_usd <= 0 or trade_value_usd <= 0:
        return 0.0
    participation = trade_value_usd / daily_volume_usd
    return impact_coeff * (participation**impact_exp)


def compute_rebalance_cost(
    old_weights: np.ndarray,
    new_weights: np.ndarray,
    portfolio_value: float,
    daily_volumes: np.ndarray | None = None,
    fee_schedule: FeeSchedule = FeeSchedule(),
) -> tuple[float, float]:
    """Compute total cost of rebalancing.

    Args:
        old_weights: Current portfolio weights.
        new_weights: Target portfolio weights.
        portfolio_value: Current portfolio value in USD.
        daily_volumes: Daily volume in USD per asset (for slippage).
        fee_schedule: Fee schedule.

    Returns:
        Tuple of (total_cost_usd, turnover).
        Turnover is the sum of absolute weight changes (0 to 2).
    """
    weight_changes = np.abs(new_weights - old_weights)
    turnover = float(np.sum(weight_changes))
    trade_values = weight_changes * portfolio_value

    # Flat fees
    total_fee = float(np.sum(trade_values * fee_schedule.effective_fee))

    # Spread cost
    spread_cost = float(np.sum(trade_values * fee_schedule.spread_bps / 10000 / 2))

    # Slippage
    slippage_cost = 0.0
    if daily_volumes is not None:
        for i in range(len(weight_changes)):
            if trade_values[i] > 0 and daily_volumes[i] > 0:
                slip = estimate_slippage(trade_values[i], daily_volumes[i])
                slippage_cost += trade_values[i] * slip

    total_cost = total_fee + spread_cost + slippage_cost
    return total_cost, turnover
