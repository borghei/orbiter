"""Crypto factor model — market, momentum, size, liquidity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


CRYPTO_FACTORS = ["market", "momentum", "size", "liquidity"]


@dataclass
class FactorExposures:
    """Factor model results."""

    loadings: pd.DataFrame  # assets x factors
    factor_returns: pd.DataFrame  # dates x factors
    r_squared: pd.Series  # per-asset R^2
    residual_returns: pd.DataFrame  # dates x assets


class CryptoFactorModel:
    """Cross-sectional factor model for crypto assets.

    Factors:
    - market: equal-weight (or cap-weighted) market return
    - momentum: long recent winners, short recent losers
    - size: small minus big (by market cap)
    - liquidity: high volume minus low volume
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        market_caps: pd.Series | None = None,
        volumes: pd.DataFrame | None = None,
        momentum_lookback: int = 30,
    ):
        self.returns = returns
        self.market_caps = market_caps
        self.volumes = volumes
        self.momentum_lookback = momentum_lookback
        self.asset_names = list(returns.columns)
        self.n_assets = len(self.asset_names)
        self._exposures: FactorExposures | None = None

    def _compute_market_factor(self) -> pd.Series:
        """Market factor: equal-weight or cap-weighted market return."""
        if self.market_caps is not None:
            # Cap-weighted
            caps = self.market_caps.reindex(self.asset_names)
            caps = caps.fillna(caps.mean())
            cap_weights = caps / caps.sum()
            return (self.returns * cap_weights.values).sum(axis=1)
        return self.returns.mean(axis=1)

    def _compute_momentum_factor(self) -> pd.Series:
        """Momentum: long winners, short losers over lookback period."""
        rolling_ret = self.returns.rolling(self.momentum_lookback).sum()
        n = self.n_assets

        def _mom_return(row):
            if row.isna().any():
                return 0.0
            ranked = row.rank()
            top_q = ranked >= ranked.quantile(0.75)
            bot_q = ranked <= ranked.quantile(0.25)
            if top_q.sum() == 0 or bot_q.sum() == 0:
                return 0.0
            return row[top_q].mean() - row[bot_q].mean()

        # Compute daily cross-sectional momentum returns
        momentum = pd.Series(index=self.returns.index, dtype=float)
        for i in range(len(self.returns)):
            if i < self.momentum_lookback:
                momentum.iloc[i] = 0.0
            else:
                lookback_rets = self.returns.iloc[i - self.momentum_lookback : i].sum()
                ranked = lookback_rets.rank()
                q75 = ranked.quantile(0.75)
                q25 = ranked.quantile(0.25)
                winners = ranked >= q75
                losers = ranked <= q25
                if winners.sum() > 0 and losers.sum() > 0:
                    momentum.iloc[i] = (
                        self.returns.iloc[i][winners].mean()
                        - self.returns.iloc[i][losers].mean()
                    )
                else:
                    momentum.iloc[i] = 0.0

        return momentum

    def _compute_size_factor(self) -> pd.Series:
        """Size factor: small minus big (by market cap)."""
        if self.market_caps is None:
            return pd.Series(0.0, index=self.returns.index)

        caps = self.market_caps.reindex(self.asset_names)
        median_cap = caps.median()
        small = caps <= median_cap
        big = caps > median_cap

        smb = pd.Series(index=self.returns.index, dtype=float)
        for i in range(len(self.returns)):
            small_ret = self.returns.iloc[i][small].mean() if small.sum() > 0 else 0
            big_ret = self.returns.iloc[i][big].mean() if big.sum() > 0 else 0
            smb.iloc[i] = small_ret - big_ret

        return smb

    def _compute_liquidity_factor(self) -> pd.Series:
        """Liquidity factor: high volume minus low volume."""
        if self.volumes is None:
            return pd.Series(0.0, index=self.returns.index)

        avg_vol = self.volumes.mean()
        median_vol = avg_vol.median()
        high_liq = avg_vol > median_vol
        low_liq = avg_vol <= median_vol

        liq = pd.Series(index=self.returns.index, dtype=float)
        for i in range(len(self.returns)):
            high_ret = self.returns.iloc[i][high_liq].mean() if high_liq.sum() > 0 else 0
            low_ret = self.returns.iloc[i][low_liq].mean() if low_liq.sum() > 0 else 0
            liq.iloc[i] = high_ret - low_ret

        return liq

    def fit(self) -> FactorExposures:
        """Run time-series regressions to get factor loadings per asset."""
        # Build factor returns matrix
        factors = pd.DataFrame({
            "market": self._compute_market_factor(),
            "momentum": self._compute_momentum_factor(),
            "size": self._compute_size_factor(),
            "liquidity": self._compute_liquidity_factor(),
        })

        # Drop initial NaN rows
        valid = factors.notna().all(axis=1) & self.returns.notna().all(axis=1)
        factors = factors[valid]
        returns = self.returns[valid]

        # Regress each asset on factors
        X = factors.values
        X_with_const = np.column_stack([np.ones(len(X)), X])

        loadings_list = []
        r_squared_list = []
        residuals = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)

        for asset in returns.columns:
            y = returns[asset].values
            # OLS: beta = (X'X)^-1 X'y
            beta, res, rank, sv = np.linalg.lstsq(X_with_const, y, rcond=None)

            loadings_list.append(beta[1:])  # Exclude intercept

            y_hat = X_with_const @ beta
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            r_squared_list.append(r2)
            residuals[asset] = y - y_hat

        loadings = pd.DataFrame(
            loadings_list,
            index=returns.columns,
            columns=["market", "momentum", "size", "liquidity"],
        )
        r_squared = pd.Series(r_squared_list, index=returns.columns, name="r_squared")

        self._exposures = FactorExposures(
            loadings=loadings,
            factor_returns=factors,
            r_squared=r_squared,
            residual_returns=residuals,
        )
        return self._exposures

    def expected_returns(
        self,
        factor_risk_premia: dict[str, float] | None = None,
    ) -> np.ndarray:
        """Factor-implied expected returns: E[r] = B @ lambda.

        Args:
            factor_risk_premia: Expected return per factor. If None, uses
                historical mean factor returns (annualized).
        """
        if self._exposures is None:
            self.fit()

        if factor_risk_premia is None:
            # Use historical means, annualized
            premia = self._exposures.factor_returns.mean().values * 365
        else:
            premia = np.array([
                factor_risk_premia.get(f, 0.0) for f in CRYPTO_FACTORS
            ])

        return self._exposures.loadings.values @ premia

    def factor_covariance(self) -> np.ndarray:
        """Factor model covariance: B @ F @ B' + D.

        Where B = loadings, F = factor covariance, D = diagonal residual variance.
        Annualized.
        """
        if self._exposures is None:
            self.fit()

        B = self._exposures.loadings.values
        F = self._exposures.factor_returns.cov().values * 365
        D = np.diag(self._exposures.residual_returns.var().values * 365)

        return B @ F @ B.T + D
