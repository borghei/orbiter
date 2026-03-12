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


@dataclass
class SentimentRegimeModel:
    """Multi-signal regime detection fusing price action with sentiment.

    Extends RegimeModel by adding Fear & Greed, funding rates, and exchange
    net flow as additional observation dimensions for the HMM.
    """

    n_regimes: int = 3
    model: object = field(default=None, repr=False)
    strategy_map: dict[Regime, str] = field(default_factory=lambda: dict(REGIME_STRATEGY_MAP))
    _label_map: dict[int, int] = field(default_factory=dict, repr=False)

    def fit(
        self,
        market_returns: pd.Series,
        sentiment_series: pd.DataFrame | None = None,
    ) -> SentimentRegimeModel:
        """Fit HMM on returns + sentiment features.

        Args:
            market_returns: Daily market returns.
            sentiment_series: DataFrame with columns like fear_greed_normalized,
                funding_rate_mean, etc. Index must align with market_returns.
                If None, falls back to price-only regime detection.
        """
        from hmmlearn.hmm import GaussianHMM

        returns_col = market_returns.values.reshape(-1, 1)

        if sentiment_series is not None and len(sentiment_series) > 0:
            # Align indices
            common_idx = market_returns.index.intersection(sentiment_series.index)
            if len(common_idx) > 30:
                returns_aligned = market_returns.loc[common_idx].values.reshape(-1, 1)
                sent_aligned = sentiment_series.loc[common_idx].values
                X = np.hstack([returns_aligned, sent_aligned])
                self._n_features = X.shape[1]
            else:
                X = returns_col
                self._n_features = 1
        else:
            X = returns_col
            self._n_features = 1

        hmm = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=200,
            random_state=42,
        )
        hmm.fit(X)
        self.model = hmm

        # Re-label by the first dimension (returns) mean
        means = hmm.means_[:, 0]
        order = np.argsort(means)
        self._label_map = {old: new for new, old in enumerate(order)}

        return self

    def predict(
        self,
        market_returns: pd.Series,
        sentiment_series: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Predict regime for each time step using returns + sentiment."""
        if self.model is None:
            raise RuntimeError("Must call fit() before predict().")

        returns_col = market_returns.values.reshape(-1, 1)

        if self._n_features > 1 and sentiment_series is not None:
            common_idx = market_returns.index.intersection(sentiment_series.index)
            if len(common_idx) == len(market_returns):
                sent_aligned = sentiment_series.loc[common_idx].values
                X = np.hstack([returns_col, sent_aligned])
            else:
                # Pad missing sentiment with zeros
                X = np.hstack([
                    returns_col,
                    np.zeros((len(market_returns), self._n_features - 1)),
                ])
        else:
            X = returns_col
            if self._n_features > 1:
                X = np.hstack([
                    returns_col,
                    np.zeros((len(market_returns), self._n_features - 1)),
                ])

        raw_states = self.model.predict(X)
        mapped = pd.Series(
            [Regime(self._label_map[s]) for s in raw_states],
            index=market_returns.index,
            name="regime",
        )
        return mapped

    def current_regime(
        self,
        market_returns: pd.Series,
        sentiment_series: pd.DataFrame | None = None,
    ) -> Regime:
        """Predict the latest regime."""
        regimes = self.predict(market_returns, sentiment_series)
        return Regime(int(regimes.iloc[-1]))

    def get_strategy(self, regime: Regime) -> str:
        """Get the optimization strategy for a given regime."""
        return self.strategy_map[regime]
