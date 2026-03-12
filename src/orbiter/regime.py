"""Regime detection using Hidden Markov Models."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
import pandas as pd


class Regime(IntEnum):
    BEAR = 0
    SIDEWAYS = 1
    BULL = 2


# Default mapping: which optimization strategy to use in each regime
REGIME_STRATEGY_MAP: dict[Regime, str] = {
    Regime.BULL: "max-sharpe",
    Regime.SIDEWAYS: "risk-parity",
    Regime.BEAR: "min-vol",
}


@dataclass
class RegimeModel:
    """Gaussian HMM for market regime detection.

    Fits a Hidden Markov Model on market returns (e.g., BTC or portfolio)
    to identify bull, sideways, and bear regimes.
    """

    n_regimes: int = 3
    model: object = field(default=None, repr=False)
    strategy_map: dict[Regime, str] = field(default_factory=lambda: dict(REGIME_STRATEGY_MAP))

    def fit(self, market_returns: pd.Series) -> RegimeModel:
        """Fit the HMM on market returns."""
        from hmmlearn.hmm import GaussianHMM

        X = market_returns.values.reshape(-1, 1)

        hmm = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=200,
            random_state=42,
        )
        hmm.fit(X)
        self.model = hmm

        # Re-label states so that BEAR=lowest mean, BULL=highest mean
        means = hmm.means_.flatten()
        order = np.argsort(means)
        self._label_map = {old: new for new, old in enumerate(order)}

        return self

    def predict(self, market_returns: pd.Series) -> pd.Series:
        """Predict regime for each time step.

        Returns Series of Regime values aligned to the input index.
        """
        if self.model is None:
            raise RuntimeError("Must call fit() before predict().")

        X = market_returns.values.reshape(-1, 1)
        raw_states = self.model.predict(X)
        mapped = pd.Series(
            [Regime(self._label_map[s]) for s in raw_states],
            index=market_returns.index,
            name="regime",
        )
        return mapped

    def current_regime(self, market_returns: pd.Series) -> Regime:
        """Predict the latest regime."""
        regimes = self.predict(market_returns)
        return Regime(int(regimes.iloc[-1]))

    def get_strategy(self, regime: Regime) -> str:
        """Get the optimization strategy for a given regime."""
        return self.strategy_map[regime]
