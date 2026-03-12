"""Portfolio optimization engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform

from orbiter.covariance import get_covariance
from orbiter.metrics import compute_metrics


@dataclass
class OptimizationResult:
    weights: pd.Series
    metrics: dict[str, float]
    strategy: str


class PortfolioOptimizer:
    """Optimizes crypto portfolio allocations using multiple strategies."""

    STRATEGIES = [
        "max-sharpe",
        "min-vol",
        "min-cvar",
        "risk-parity",
        "hrp",
        "regime-aware",
        "factor-max-sharpe",
    ]

    def __init__(
        self,
        returns: pd.DataFrame,
        cov_method: str = "ledoit-wolf",
        risk_free_rate: float = 0.0,
        periods_per_year: int = 365,
        factor_model: object | None = None,
    ):
        self.returns = returns
        self.n_assets = returns.shape[1]
        self.asset_names = list(returns.columns)
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        self.factor_model = factor_model

        self.expected_returns = returns.mean().values * periods_per_year
        self.cov_matrix = get_covariance(
            returns, method=cov_method, periods_per_year=periods_per_year
        )

    def _portfolio_return(self, weights: np.ndarray) -> float:
        return float(weights @ self.expected_returns)

    def _portfolio_volatility(self, weights: np.ndarray) -> float:
        return float(np.sqrt(weights @ self.cov_matrix @ weights))

    def _portfolio_sharpe(self, weights: np.ndarray) -> float:
        ret = self._portfolio_return(weights)
        vol = self._portfolio_volatility(weights)
        if vol == 0:
            return 0.0
        return (ret - self.risk_free_rate) / vol

    def _portfolio_returns_series(self, weights: np.ndarray) -> pd.Series:
        return (self.returns * weights).sum(axis=1)

    def _base_constraints(self):
        return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    def _base_bounds(self):
        return [(0.0, 1.0)] * self.n_assets

    def _initial_weights(self):
        return np.ones(self.n_assets) / self.n_assets

    def _make_result(self, weights: np.ndarray, strategy: str) -> OptimizationResult:
        port_returns = self._portfolio_returns_series(weights)
        metrics = compute_metrics(port_returns, self.risk_free_rate, self.periods_per_year)
        return OptimizationResult(
            weights=pd.Series(weights, index=self.asset_names),
            metrics=metrics,
            strategy=strategy,
        )

    def max_sharpe(self) -> OptimizationResult:
        """Maximize the Sharpe ratio (risk-adjusted return)."""
        result = minimize(
            lambda w: -self._portfolio_sharpe(w),
            self._initial_weights(),
            method="SLSQP",
            bounds=self._base_bounds(),
            constraints=self._base_constraints(),
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        return self._make_result(result.x, "max-sharpe")

    def min_volatility(self) -> OptimizationResult:
        """Minimize portfolio volatility."""
        result = minimize(
            lambda w: self._portfolio_volatility(w),
            self._initial_weights(),
            method="SLSQP",
            bounds=self._base_bounds(),
            constraints=self._base_constraints(),
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        return self._make_result(result.x, "min-vol")

    def min_cvar(self, alpha: float = 0.05) -> OptimizationResult:
        """Minimize Conditional Value-at-Risk (Expected Shortfall)."""
        from orbiter.metrics import cvar as compute_cvar

        def objective(weights):
            port_returns = self._portfolio_returns_series(weights)
            return -compute_cvar(port_returns, alpha)

        result = minimize(
            objective,
            self._initial_weights(),
            method="SLSQP",
            bounds=self._base_bounds(),
            constraints=self._base_constraints(),
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        return self._make_result(result.x, "min-cvar")

    def risk_parity(self) -> OptimizationResult:
        """Equal risk contribution — each asset contributes same risk to portfolio."""

        def objective(weights):
            port_vol = self._portfolio_volatility(weights)
            if port_vol == 0:
                return 0.0
            marginal = self.cov_matrix @ weights
            risk_contrib = weights * marginal / port_vol
            target = port_vol / self.n_assets
            return float(np.sum((risk_contrib - target) ** 2))

        result = minimize(
            objective,
            self._initial_weights(),
            method="SLSQP",
            bounds=self._base_bounds(),
            constraints=self._base_constraints(),
            options={"maxiter": 1000, "ftol": 1e-15},
        )
        return self._make_result(result.x, "risk-parity")

    def hrp(self) -> OptimizationResult:
        """Hierarchical Risk Parity (Lopez de Prado).

        Uses correlation structure only — does not rely on expected returns.
        More robust for assets with unreliable return forecasts.
        """
        corr = self.returns.corr().values
        # Correlation distance
        dist = np.sqrt((1 - corr) / 2)
        np.fill_diagonal(dist, 0)
        condensed = squareform(dist, checks=False)
        link = linkage(condensed, method="single")
        order = self._quasi_diag(link)
        weights = self._recursive_bisection(order)
        return self._make_result(weights, "hrp")

    def _quasi_diag(self, link: np.ndarray) -> list[int]:
        """Reorder assets to place correlated ones adjacent (recursive)."""
        link = link.astype(int)

        def _recurse(node: int) -> list[int]:
            if node < self.n_assets:
                return [node]
            row = node - self.n_assets
            left = int(link[row, 0])
            right = int(link[row, 1])
            return _recurse(left) + _recurse(right)

        root = self.n_assets + len(link) - 1
        return _recurse(root)

    def _recursive_bisection(self, order: list[int]) -> np.ndarray:
        """Allocate weights by recursive bisection of sorted correlation matrix."""
        weights = np.ones(self.n_assets)
        clusters = [order]

        while clusters:
            new_clusters = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue
                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                left_var = self._cluster_var(left)
                right_var = self._cluster_var(right)
                alloc = 1 - left_var / (left_var + right_var)

                for i in left:
                    weights[i] *= alloc
                for i in right:
                    weights[i] *= 1 - alloc

                new_clusters.extend([left, right])
            clusters = new_clusters

        weights /= weights.sum()
        return weights

    def _cluster_var(self, indices: list[int]) -> float:
        """Variance of an inverse-variance weighted cluster."""
        cov_sub = self.cov_matrix[np.ix_(indices, indices)]
        ivp = 1.0 / np.diag(cov_sub)
        ivp /= ivp.sum()
        return float(ivp @ cov_sub @ ivp)

    def efficient_frontier(self, n_points: int = 50) -> pd.DataFrame:
        """Compute the efficient frontier.

        Returns DataFrame with columns: return, volatility, sharpe, and one
        column per asset for weights.
        """
        min_ret = self.expected_returns.min()
        max_ret = self.expected_returns.max()
        target_returns = np.linspace(min_ret, max_ret, n_points)

        results = []
        for target in target_returns:
            constraints = [
                self._base_constraints(),
                {"type": "eq", "fun": lambda w, t=target: self._portfolio_return(w) - t},
            ]
            res = minimize(
                lambda w: self._portfolio_volatility(w),
                self._initial_weights(),
                method="SLSQP",
                bounds=self._base_bounds(),
                constraints=constraints,
                options={"maxiter": 1000, "ftol": 1e-12},
            )
            if res.success:
                ret = self._portfolio_return(res.x)
                vol = self._portfolio_volatility(res.x)
                sharpe = (ret - self.risk_free_rate) / vol if vol > 0 else 0
                row = {"return": ret, "volatility": vol, "sharpe": sharpe}
                for name, w in zip(self.asset_names, res.x):
                    row[name] = w
                results.append(row)

        return pd.DataFrame(results)

    def regime_aware(self) -> OptimizationResult:
        """Regime-aware optimization — detects market regime and picks strategy."""
        from orbiter.regime import RegimeModel

        market_returns = self.returns.mean(axis=1)
        model = RegimeModel(n_regimes=3)
        model.fit(market_returns)
        regime = model.current_regime(market_returns)
        sub_strategy = model.get_strategy(regime)

        result = self.optimize(sub_strategy)
        return OptimizationResult(
            weights=result.weights,
            metrics=result.metrics,
            strategy=f"regime-aware ({sub_strategy})",
        )

    def factor_max_sharpe(self) -> OptimizationResult:
        """Max Sharpe using factor-model implied expected returns."""
        if self.factor_model is None:
            raise ValueError(
                "factor_model must be provided for factor-max-sharpe strategy. "
                "Pass a CryptoFactorModel to PortfolioOptimizer."
            )
        # Override expected returns with factor-implied
        original_er = self.expected_returns.copy()
        self.expected_returns = self.factor_model.expected_returns()
        result = self.max_sharpe()
        self.expected_returns = original_er
        return OptimizationResult(
            weights=result.weights,
            metrics=result.metrics,
            strategy="factor-max-sharpe",
        )

    def stress_test(
        self,
        weights: np.ndarray | None = None,
        n_simulations: int = 10000,
        horizon_days: int = 30,
        distribution: str = "student-t",
        df: float = 5.0,
    ) -> dict[str, float]:
        """Run Monte Carlo stress test on the portfolio."""
        from orbiter.stress import monte_carlo_stress

        if weights is None:
            weights = self._initial_weights()
        mu = self.returns.mean().values
        cov = self.returns.cov().values
        return monte_carlo_stress(
            weights,
            mu,
            cov,
            n_simulations=n_simulations,
            horizon_days=horizon_days,
            distribution=distribution,
            df=df,
        )

    def optimize(self, strategy: str = "max-sharpe") -> OptimizationResult:
        """Run optimization with the named strategy."""
        dispatch = {
            "max-sharpe": self.max_sharpe,
            "min-vol": self.min_volatility,
            "min-cvar": self.min_cvar,
            "risk-parity": self.risk_parity,
            "hrp": self.hrp,
            "regime-aware": self.regime_aware,
            "factor-max-sharpe": self.factor_max_sharpe,
        }
        if strategy not in dispatch:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Choose from: {list(dispatch.keys())}"
            )
        return dispatch[strategy]()
