"""Data layer — fetch crypto OHLCV data via ccxt."""

from __future__ import annotations

import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd


class PriceLoader:
    """Fetches cryptocurrency price data from exchanges via ccxt."""

    def __init__(self, exchange: str = "binance"):
        self._exchange_id = exchange
        self._exchange = None
        self._cache: dict[str, pd.DataFrame] = {}

    def _get_exchange(self):
        if self._exchange is None:
            import ccxt

            exchange_class = getattr(ccxt, self._exchange_id)
            self._exchange = exchange_class({"enableRateLimit": True})
        return self._exchange

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """Convert user-friendly symbol to exchange pair. BTC -> BTC/USDT."""
        symbol = symbol.upper().strip()
        if "/" in symbol:
            return symbol
        return f"{symbol}/USDT"

    def fetch_ohlcv(
        self,
        symbols: list[str],
        timeframe: str = "1d",
        days: int = 365,
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV data for multiple symbols.

        Returns dict mapping symbol name to DataFrame with columns:
        [timestamp, open, high, low, close, volume].
        """
        exchange = self._get_exchange()
        since = int(
            (datetime.now(timezone.utc).timestamp() - days * 86400) * 1000
        )
        result = {}

        for symbol in symbols:
            pair = self._normalize_symbol(symbol)
            cache_key = f"{pair}:{timeframe}:{days}"

            if cache_key in self._cache:
                result[symbol.upper()] = self._cache[cache_key]
                continue

            all_candles = []
            fetch_since = since

            while True:
                candles = exchange.fetch_ohlcv(
                    pair, timeframe=timeframe, since=fetch_since, limit=1000
                )
                if not candles:
                    break
                all_candles.extend(candles)
                fetch_since = candles[-1][0] + 1
                if len(candles) < 1000:
                    break
                time.sleep(0.1)

            if not all_candles:
                raise ValueError(f"No data returned for {pair}")

            df = pd.DataFrame(
                all_candles,
                columns=["timestamp", "open", "high", "low", "close", "volume"],
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df = df.set_index("timestamp").sort_index()
            df = df[~df.index.duplicated(keep="last")]

            self._cache[cache_key] = df
            result[symbol.upper()] = df

        return result

    def get_close_prices(
        self,
        symbols: list[str],
        timeframe: str = "1d",
        days: int = 365,
    ) -> pd.DataFrame:
        """Get close prices as a DataFrame with one column per asset."""
        ohlcv = self.fetch_ohlcv(symbols, timeframe, days)
        closes = {}
        for symbol, df in ohlcv.items():
            closes[symbol] = df["close"]

        prices = pd.DataFrame(closes)
        prices = prices.dropna()

        if len(prices) < 30:
            raise ValueError(
                f"Only {len(prices)} overlapping data points. Need at least 30."
            )

        dropped = sum(len(df) for df in ohlcv.values()) / len(ohlcv) - len(prices)
        if dropped > 0.1 * len(prices):
            import warnings

            warnings.warn(
                f"Dropped {dropped:.0f} rows (~{dropped/len(prices)*100:.0f}%) "
                "to align dates across assets.",
                stacklevel=2,
            )

        return prices

    def get_returns(
        self,
        symbols: list[str],
        timeframe: str = "1d",
        days: int = 365,
    ) -> pd.DataFrame:
        """Get log returns for multiple symbols."""
        prices = self.get_close_prices(symbols, timeframe, days)
        returns = np.log(prices / prices.shift(1)).dropna()
        return returns

    def get_volumes(
        self,
        symbols: list[str],
        timeframe: str = "1d",
        days: int = 365,
    ) -> pd.DataFrame:
        """Get daily USD volume for multiple symbols (close * volume)."""
        ohlcv = self.fetch_ohlcv(symbols, timeframe, days)
        volumes = {}
        for symbol, df in ohlcv.items():
            volumes[symbol] = df["close"] * df["volume"]
        result = pd.DataFrame(volumes).dropna()
        return result
