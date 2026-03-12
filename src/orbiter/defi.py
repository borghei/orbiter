"""DeFi yield data — staking/lending yields for yield-adjusted optimization."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DEFI_SYMBOL_MAP: dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "SOL": "solana",
    "AVAX": "avalanche-2",
    "BNB": "binancecoin",
    "ADA": "cardano",
    "DOT": "polkadot",
    "MATIC": "polygon",
    "ATOM": "cosmos",
    "NEAR": "near",
    "ARB": "arbitrum",
    "OP": "optimism",
    "SUI": "sui",
    "APT": "aptos",
    "SEI": "sei-network",
    "TIA": "celestia",
    "INJ": "injective-protocol",
    "LINK": "chainlink",
    "UNI": "uniswap",
    "AAVE": "aave",
}

DEFILLAMA_POOLS_URL = "https://yields.llama.fi/pools"

# Conservative manual staking APYs (as decimals).
MANUAL_STAKING_APYS: dict[str, float] = {
    "ETH": 0.035,
    "BTC": 0.005,
    "SOL": 0.065,
    "ADA": 0.030,
    "DOT": 0.110,
    "ATOM": 0.150,
    "AVAX": 0.080,
    "BNB": 0.020,
    "NEAR": 0.050,
}


@dataclass
class YieldInfo:
    """Yield data for a single asset."""

    symbol: str
    staking_apy: float = 0.0
    lending_apy: float = 0.0
    best_yield: float = 0.0
    protocol: str = ""
    source: str = "manual"


class YieldCollector:
    """Collects DeFi yield data from DeFiLlama and manual fallbacks."""

    def __init__(self) -> None:
        pass

    def get_staking_yields(self, symbols: list[str]) -> dict[str, YieldInfo]:
        """Fetch yields from DeFiLlama pools API."""
        try:
            resp = requests.get(DEFILLAMA_POOLS_URL, timeout=15)
            resp.raise_for_status()
            pools = resp.json().get("data", [])
        except (requests.RequestException, ValueError) as exc:
            logger.warning("DeFiLlama pools request failed: %s", exc)
            return {}

        symbol_set = {s.upper() for s in symbols}
        results: dict[str, YieldInfo] = {}

        # Group matching pools by symbol.
        candidates: dict[str, list[dict]] = {s: [] for s in symbol_set}
        for pool in pools:
            pool_symbol = (pool.get("symbol") or "").upper()
            pool_category = (pool.get("category") or "").lower()
            if pool_symbol in symbol_set and pool_category in ("staking", "lending"):
                candidates[pool_symbol].append(pool)

        for sym in symbol_set:
            sym_pools = candidates.get(sym, [])
            if not sym_pools:
                continue

            # Sort by TVL descending, pick top per category.
            sym_pools.sort(key=lambda p: p.get("tvlUsd", 0), reverse=True)

            staking_apy = 0.0
            lending_apy = 0.0
            best_protocol = ""
            best_apy = 0.0

            for pool in sym_pools:
                cat = (pool.get("category") or "").lower()
                apy = (pool.get("apy") or 0.0) / 100.0  # percentage -> decimal
                protocol = pool.get("project", "")

                if cat == "staking" and staking_apy == 0.0:
                    staking_apy = apy
                    if apy > best_apy:
                        best_apy = apy
                        best_protocol = protocol
                elif cat == "lending" and lending_apy == 0.0:
                    lending_apy = apy
                    if apy > best_apy:
                        best_apy = apy
                        best_protocol = protocol

                # Stop once we have both.
                if staking_apy > 0 and lending_apy > 0:
                    break

            results[sym] = YieldInfo(
                symbol=sym,
                staking_apy=staking_apy,
                lending_apy=lending_apy,
                best_yield=max(staking_apy, lending_apy),
                protocol=best_protocol,
                source="defillama",
            )

        return results

    def get_manual_yields(self) -> dict[str, YieldInfo]:
        """Fallback conservative yield estimates."""
        results: dict[str, YieldInfo] = {}
        for sym, apy in MANUAL_STAKING_APYS.items():
            results[sym] = YieldInfo(
                symbol=sym,
                staking_apy=apy,
                lending_apy=0.0,
                best_yield=apy,
                protocol="manual",
                source="manual",
            )
        return results

    def collect(
        self, symbols: list[str], use_live: bool = True
    ) -> dict[str, YieldInfo]:
        """Collect yields for all symbols, with API + manual fallback."""
        upper_symbols = [s.upper() for s in symbols]
        results: dict[str, YieldInfo] = {}

        if use_live:
            try:
                results = self.get_staking_yields(upper_symbols)
            except Exception as exc:
                logger.warning("Live yield fetch failed: %s", exc)

        # Fill missing from manual.
        manual = self.get_manual_yields()
        for sym in upper_symbols:
            if sym not in results:
                if sym in manual:
                    results[sym] = manual[sym]
                else:
                    results[sym] = YieldInfo(symbol=sym)

        return results


def adjust_expected_returns(
    mu: pd.Series,
    yields: dict[str, YieldInfo],
    weight: float = 1.0,
) -> pd.Series:
    """Add daily DeFi yield to expected returns.

    Args:
        mu: Daily expected returns per asset.
        yields: Yield info per symbol.
        weight: 0-1, how much yield influences optimization.
    """
    adjusted = mu.copy()
    for sym in adjusted.index:
        sym_upper = str(sym).upper()
        if sym_upper in yields:
            daily_yield = yields[sym_upper].best_yield / 365.0
            adjusted[sym] = adjusted[sym] + daily_yield * weight
    return adjusted


def yield_risk_adjustment(yields: dict[str, YieldInfo]) -> dict[str, float]:
    """Yield reliability score per symbol (0-1).

    Higher APY implies more smart contract risk.
    Staking: score = max(0, 1.0 - apy * 0.5)
    Lending: score = max(0, 1.0 - apy * 1.0)
    Returns the lower (more conservative) of the two scores.
    """
    scores: dict[str, float] = {}
    for sym, info in yields.items():
        staking_score = max(0.0, 1.0 - info.staking_apy * 0.5)
        lending_score = max(0.0, 1.0 - info.lending_apy * 1.0)

        if info.staking_apy > 0 and info.lending_apy > 0:
            scores[sym] = min(staking_score, lending_score)
        elif info.staking_apy > 0:
            scores[sym] = staking_score
        elif info.lending_apy > 0:
            scores[sym] = lending_score
        else:
            scores[sym] = 1.0  # no yield exposure = no smart contract risk

    return scores
