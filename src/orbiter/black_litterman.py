"""Black-Litterman model for crypto portfolio optimization with AI-generated views."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from orbiter.covariance import get_covariance

# Rough market cap ratios for top coins when real data unavailable.
MARKET_CAPS_FALLBACK: dict[str, float] = {
    "BTC": 0.50,
    "ETH": 0.18,
    "BNB": 0.04,
    "SOL": 0.04,
    "XRP": 0.03,
    "ADA": 0.02,
    "AVAX": 0.01,
    "DOT": 0.01,
}


@dataclass
class View:
    """A single investor view on asset returns.

    Args:
        asset: Asset ticker for absolute views (e.g. "SOL"), or tuple of
            (long, short) for relative views (e.g. ("SOL", "ETH")).
        return_view: Expected excess return, annualized.
        confidence: Confidence level 0-1, where 1.0 = very confident.
        source: Origin of the view.
    """

    asset: str | tuple[str, str]
    return_view: float
    confidence: float
    source: str = "manual"


@dataclass
class BLResult:
    """Result of Black-Litterman optimization."""

    posterior_returns: pd.Series
    posterior_covariance: pd.DataFrame
    weights: pd.Series
    views_used: list[View]
    market_implied_returns: pd.Series


class BlackLitterman:
    """Black-Litterman model (He & Litterman 1999) for combining market
    equilibrium with investor views.

    Args:
        returns: DataFrame of log returns (assets as columns).
        market_caps: Market cap per asset for implied weights. If None,
            falls back to MARKET_CAPS_FALLBACK or equal weights.
        risk_aversion: Risk aversion coefficient (delta).
        tau: Scalar reflecting uncertainty in the prior (typically 0.01-0.1).
        risk_free_rate: Annualized risk-free rate.
        periods_per_year: Annualization factor (365 for crypto).
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        market_caps: pd.Series | None = None,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        risk_free_rate: float = 0.0,
        periods_per_year: int = 365,
    ):
        self.returns = returns
        self.asset_names: list[str] = list(returns.columns)
        self.n_assets: int = returns.shape[1]
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

        self.cov_matrix: np.ndarray = get_covariance(
            returns, method="ledoit-wolf", periods_per_year=periods_per_year
        )

        self.market_weights: np.ndarray = self._compute_market_weights(market_caps)

        # Market-implied equilibrium returns: pi = delta * Sigma @ w_mkt
        self.pi: np.ndarray = self.risk_aversion * self.cov_matrix @ self.market_weights

    def _compute_market_weights(self, market_caps: pd.Series | None) -> np.ndarray:
        """Derive market-cap weights, falling back to defaults or equal."""
        if market_caps is not None:
            caps = market_caps.reindex(self.asset_names).fillna(0.0).values
            total = caps.sum()
            if total > 0:
                return caps / total
            return np.ones(self.n_assets) / self.n_assets

        # Try fallback caps
        caps = np.array([MARKET_CAPS_FALLBACK.get(a, 0.0) for a in self.asset_names])
        total = caps.sum()
        if total > 0:
            # Distribute remainder equally among unknown assets
            known_total = total
            unknown = [i for i, a in enumerate(self.asset_names) if a not in MARKET_CAPS_FALLBACK]
            if unknown:
                remainder = max(0.0, 1.0 - known_total)
                per_unknown = remainder / len(unknown)
                for i in unknown:
                    caps[i] = per_unknown
            return caps / caps.sum()

        return np.ones(self.n_assets) / self.n_assets

    def implied_returns(self) -> pd.Series:
        """Market-implied equilibrium returns (pi)."""
        return pd.Series(self.pi, index=self.asset_names, name="implied_return")

    def _build_views_matrices(
        self, views: list[View]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build P (pick), Q (return), and Omega (uncertainty) matrices."""
        n_views = len(views)
        P = np.zeros((n_views, self.n_assets))
        Q = np.zeros(n_views)

        for k, view in enumerate(views):
            if isinstance(view.asset, tuple):
                # Relative view: long first, short second
                long_asset, short_asset = view.asset
                i_long = self.asset_names.index(long_asset)
                i_short = self.asset_names.index(short_asset)
                P[k, i_long] = 1.0
                P[k, i_short] = -1.0
            else:
                # Absolute view
                i = self.asset_names.index(view.asset)
                P[k, i] = 1.0
            Q[k] = view.return_view

        # Omega: diagonal uncertainty matrix
        # omega_i = (1/confidence_i - 1) * tau * (P @ Sigma @ P.T)_ii
        tau_sigma = self.tau * self.cov_matrix
        p_sigma_pt = P @ tau_sigma @ P.T
        omega_diag = np.zeros(n_views)
        for k, view in enumerate(views):
            confidence = np.clip(view.confidence, 1e-6, 1.0 - 1e-6)
            omega_diag[k] = (1.0 / confidence - 1.0) * p_sigma_pt[k, k]

        Omega = np.diag(omega_diag)
        return P, Q, Omega

    def posterior(self, views: list[View]) -> BLResult:
        """Compute Black-Litterman posterior given investor views.

        Args:
            views: List of View objects expressing return expectations.

        Returns:
            BLResult with posterior returns, covariance, and optimal weights.
        """
        if not views:
            # No views — return equilibrium
            weights = self.market_weights.copy()
            return BLResult(
                posterior_returns=pd.Series(self.pi, index=self.asset_names),
                posterior_covariance=pd.DataFrame(
                    self.cov_matrix, index=self.asset_names, columns=self.asset_names
                ),
                weights=pd.Series(weights, index=self.asset_names),
                views_used=[],
                market_implied_returns=self.implied_returns(),
            )

        P, Q, Omega = self._build_views_matrices(views)
        tau_sigma = self.tau * self.cov_matrix

        # M = P @ tau_sigma @ P.T + Omega
        M = P @ tau_sigma @ P.T + Omega

        # Posterior returns: mu_bl = pi + tau*Sigma @ P.T @ inv(M) @ (Q - P @ pi)
        M_inv_residual = np.linalg.solve(M, Q - P @ self.pi)
        posterior_mu = self.pi + tau_sigma @ P.T @ M_inv_residual

        # Posterior covariance:
        # Sigma_bl = (1 + tau)*Sigma - tau^2 * Sigma @ P.T @ inv(M) @ P @ Sigma
        M_inv_P_sigma = np.linalg.solve(M, P @ self.cov_matrix)
        posterior_cov = (
            (1.0 + self.tau) * self.cov_matrix
            - self.tau**2 * self.cov_matrix @ P.T @ M_inv_P_sigma
        )

        # Optimal weights: w* = inv(delta * posterior_cov) @ posterior_mu
        raw_weights = np.linalg.solve(
            self.risk_aversion * posterior_cov, posterior_mu
        )

        # Long only: clip and normalize
        raw_weights = np.clip(raw_weights, 0.0, None)
        total = raw_weights.sum()
        if total > 0:
            weights = raw_weights / total
        else:
            weights = np.ones(self.n_assets) / self.n_assets

        weights = np.clip(weights, 0.0, 1.0)
        weights /= weights.sum()

        return BLResult(
            posterior_returns=pd.Series(posterior_mu, index=self.asset_names),
            posterior_covariance=pd.DataFrame(
                posterior_cov, index=self.asset_names, columns=self.asset_names
            ),
            weights=pd.Series(weights, index=self.asset_names),
            views_used=views,
            market_implied_returns=self.implied_returns(),
        )

    def optimize(self, views: list[View]) -> BLResult:
        """Alias for posterior()."""
        return self.posterior(views)


def parse_ai_views(response: str, assets: list[str]) -> list[View]:
    """Parse AI-generated JSON response into View objects.

    Expects JSON array with objects like:
        {"asset": "SOL", "return": 0.15, "confidence": 0.7, "reasoning": "..."}
    or for relative views:
        {"asset": ["SOL", "ETH"], "return": 0.05, "confidence": 0.6}

    Strips markdown code fences. Returns empty list on parse failure.
    """
    try:
        # Strip markdown code fences
        text = response.strip()
        text = re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
        text = text.strip()

        data = json.loads(text)
        if not isinstance(data, list):
            return []

        views: list[View] = []
        for item in data:
            asset_raw = item.get("asset")
            ret = item.get("return")
            conf = item.get("confidence")

            if asset_raw is None or ret is None or conf is None:
                continue

            # Determine asset type
            if isinstance(asset_raw, list) and len(asset_raw) == 2:
                long, short = str(asset_raw[0]), str(asset_raw[1])
                if long not in assets or short not in assets:
                    continue
                asset: str | tuple[str, str] = (long, short)
            elif isinstance(asset_raw, str):
                if asset_raw not in assets:
                    continue
                asset = asset_raw
            else:
                continue

            conf = float(np.clip(float(conf), 0.01, 0.99))

            views.append(
                View(
                    asset=asset,
                    return_view=float(ret),
                    confidence=conf,
                    source="ai",
                )
            )

        return views

    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return []
