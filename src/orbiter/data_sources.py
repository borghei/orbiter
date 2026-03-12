"""Additional data sources — CoinGecko, on-chain metrics."""

from __future__ import annotations

import time

import pandas as pd
import requests


class CoinGeckoClient:
    """CoinGecko free API client (no key needed, rate-limited)."""

    BASE_URL = "https://api.coingecko.com/api/v3"

    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})
        self._symbol_id_cache: dict[str, str] | None = None

    def _get(self, endpoint: str, params: dict | None = None) -> dict | list:
        url = f"{self.BASE_URL}{endpoint}"
        resp = self._session.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            time.sleep(60)
            resp = self._session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def top_coins(
        self,
        n: int = 50,
        min_mcap_usd: float = 1e8,
    ) -> pd.DataFrame:
        """Get top coins by market cap.

        Returns DataFrame with columns: symbol, name, market_cap, volume_24h,
        price, rank.
        """
        per_page = min(n, 250)
        pages_needed = (n + per_page - 1) // per_page
        all_coins = []

        for page in range(1, pages_needed + 1):
            data = self._get(
                "/coins/markets",
                params={
                    "vs_currency": "usd",
                    "order": "market_cap_desc",
                    "per_page": per_page,
                    "page": page,
                    "sparkline": "false",
                },
            )
            all_coins.extend(data)
            if len(data) < per_page:
                break
            if page < pages_needed:
                time.sleep(1.5)

        df = pd.DataFrame(all_coins)
        df = df.rename(
            columns={
                "market_cap_rank": "rank",
                "current_price": "price",
                "total_volume": "volume_24h",
            }
        )
        df["symbol"] = df["symbol"].str.upper()
        df = df[df["market_cap"] >= min_mcap_usd]

        cols = ["symbol", "name", "market_cap", "volume_24h", "price", "rank"]
        return df[cols].head(n).reset_index(drop=True)

    def get_market_caps(self, symbols: list[str]) -> pd.Series:
        """Get current market caps for given symbols."""
        top = self.top_coins(n=250, min_mcap_usd=0)
        top = top.set_index("symbol")
        symbols_upper = [s.upper() for s in symbols]
        result = {}
        for s in symbols_upper:
            if s in top.index:
                result[s] = top.loc[s, "market_cap"]
                if isinstance(result[s], pd.Series):
                    result[s] = result[s].iloc[0]
            else:
                result[s] = None
        return pd.Series(result, name="market_cap")

    def _get_coin_id(self, symbol: str) -> str | None:
        """Resolve symbol to CoinGecko coin ID."""
        if self._symbol_id_cache is None:
            coins_list = self._get("/coins/list")
            self._symbol_id_cache = {}
            for coin in coins_list:
                sym = coin["symbol"].upper()
                # Prefer the most well-known coin for each symbol
                if sym not in self._symbol_id_cache:
                    self._symbol_id_cache[sym] = coin["id"]
        return self._symbol_id_cache.get(symbol.upper())

    def get_historical_market_caps(self, symbol: str, days: int = 365) -> pd.Series | None:
        """Historical daily market cap for a single coin."""
        coin_id = self._get_coin_id(symbol)
        if coin_id is None:
            return None

        data = self._get(
            f"/coins/{coin_id}/market_chart",
            params={"vs_currency": "usd", "days": days, "interval": "daily"},
        )
        if "market_caps" not in data:
            return None

        mc = data["market_caps"]
        df = pd.DataFrame(mc, columns=["timestamp", "market_cap"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df.set_index("timestamp")["market_cap"]


class OnChainMetrics:
    """On-chain data from free public APIs.

    Currently supports BTC (blockchain.com) and ETH (etherscan).
    Returns None for unsupported assets.
    """

    SUPPORTED = {"BTC", "ETH"}

    def get_active_addresses(self, symbol: str, days: int = 90) -> pd.Series | None:
        """Daily active addresses."""
        symbol = symbol.upper()
        if symbol not in self.SUPPORTED:
            return None

        if symbol == "BTC":
            return self._btc_active_addresses(days)
        return None  # ETH requires API key

    def _btc_active_addresses(self, days: int) -> pd.Series | None:
        try:
            url = "https://api.blockchain.info/charts/n-unique-addresses"
            resp = requests.get(
                url,
                params={"timespan": f"{days}days", "format": "json"},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            values = data.get("values", [])
            if not values:
                return None
            df = pd.DataFrame(values)
            df["x"] = pd.to_datetime(df["x"], unit="s", utc=True)
            return df.set_index("x")["y"].rename("active_addresses")
        except Exception:
            return None

    def get_nvt_ratio(self, symbol: str, days: int = 90) -> pd.Series | None:
        """Network Value to Transactions ratio (BTC only)."""
        if symbol.upper() != "BTC":
            return None

        try:
            url = "https://api.blockchain.info/charts/nvt"
            resp = requests.get(
                url,
                params={"timespan": f"{days}days", "format": "json"},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            values = data.get("values", [])
            if not values:
                return None
            df = pd.DataFrame(values)
            df["x"] = pd.to_datetime(df["x"], unit="s", utc=True)
            return df.set_index("x")["y"].rename("nvt_ratio")
        except Exception:
            return None
