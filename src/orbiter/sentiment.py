"""Sentiment data collection for crypto portfolio optimization."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import requests

logger = logging.getLogger(__name__)

SENTIMENT_THRESHOLDS: dict[str, int] = {
    "extreme_fear": 20,
    "fear": 40,
    "neutral": 60,
    "greed": 80,
    "extreme_greed": 100,
}

FEAR_GREED_API = "https://api.alternative.me/fng/"
BLOCKCHAIN_COM_API = "https://api.blockchain.info/charts/exchange-trade-volume"

# Funding rate above this (absolute) signals crowded positioning
FUNDING_EXTREME_THRESHOLD: float = 0.001

# Default timeout for HTTP requests (seconds)
REQUEST_TIMEOUT: int = 10


@dataclass
class SentimentData:
    """Aggregated sentiment signals for regime detection."""

    fear_greed_index: int
    fear_greed_label: str
    funding_rates: dict[str, float]
    exchange_net_flow: float | None
    timestamp: str


def _label_from_value(value: int) -> str:
    """Map fear & greed index value to human-readable label."""
    if value <= SENTIMENT_THRESHOLDS["extreme_fear"]:
        return "Extreme Fear"
    if value <= SENTIMENT_THRESHOLDS["fear"]:
        return "Fear"
    if value <= SENTIMENT_THRESHOLDS["neutral"]:
        return "Neutral"
    if value <= SENTIMENT_THRESHOLDS["greed"]:
        return "Greed"
    return "Extreme Greed"


class SentimentCollector:
    """Collects sentiment signals from public APIs and exchanges."""

    def __init__(self, exchange: str = "binance") -> None:
        self._exchange_id = exchange
        self._exchange = None

    def _get_exchange(self):
        if self._exchange is None:
            import ccxt

            exchange_class = getattr(ccxt, self._exchange_id)
            self._exchange = exchange_class({"enableRateLimit": True})
        return self._exchange

    def get_fear_greed(self) -> dict[str, int | str]:
        """Fetch the latest Fear & Greed Index from alternative.me."""
        try:
            resp = requests.get(FEAR_GREED_API, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            entry = resp.json()["data"][0]
            value = int(entry["value"])
            return {
                "value": value,
                "label": _label_from_value(value),
                "timestamp": entry.get("timestamp", ""),
            }
        except Exception:
            logger.warning("Failed to fetch Fear & Greed index, using default")
            return {"value": 50, "label": "Neutral", "timestamp": ""}

    @staticmethod
    def _normalize_perp_symbol(symbol: str) -> str:
        """Convert user-friendly symbol to perpetual pair. BTC -> BTC/USDT:USDT."""
        symbol = symbol.upper().strip()
        if ":" in symbol:
            return symbol
        base = symbol.split("/")[0]
        return f"{base}/USDT:USDT"

    def get_funding_rates(self, symbols: list[str]) -> dict[str, float]:
        """Fetch perpetual funding rates via ccxt."""
        rates: dict[str, float] = {}
        try:
            exchange = self._get_exchange()
            for symbol in symbols:
                perp = self._normalize_perp_symbol(symbol)
                try:
                    result = exchange.fetch_funding_rate(perp)
                    rate = result.get("fundingRate")
                    if rate is not None:
                        rates[symbol.upper().strip()] = float(rate)
                except Exception:
                    logger.debug("Funding rate unavailable for %s", symbol)
        except Exception:
            logger.warning("Failed to fetch funding rates")
        return rates

    def get_exchange_net_flow(self) -> float | None:
        """Fetch BTC exchange net flow from blockchain.com."""
        try:
            resp = requests.get(
                BLOCKCHAIN_COM_API,
                params={"timespan": "1days", "format": "json"},
                timeout=REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            values = data.get("values", [])
            if values:
                return float(values[-1]["y"])
        except Exception:
            logger.warning("Failed to fetch exchange net flow")
        return None

    def collect(self, symbols: list[str]) -> SentimentData:
        """Collect all sentiment signals into a single snapshot."""
        fg = self.get_fear_greed()
        funding = self.get_funding_rates(symbols)
        net_flow = self.get_exchange_net_flow()

        return SentimentData(
            fear_greed_index=fg["value"],
            fear_greed_label=fg["label"],
            funding_rates=funding,
            exchange_net_flow=net_flow,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )


def sentiment_features(data: SentimentData) -> dict[str, float]:
    """Convert SentimentData into numeric features for the HMM."""
    # Fear & Greed normalized to 0-1
    fear_greed_normalized = data.fear_greed_index / 100.0

    # Funding rate aggregates
    if data.funding_rates:
        rates = list(data.funding_rates.values())
        funding_rate_mean = float(np.mean(rates))
    else:
        funding_rate_mean = 0.0

    funding_rate_extreme = 1.0 if abs(funding_rate_mean) > FUNDING_EXTREME_THRESHOLD else 0.0

    # Net flow signal
    if data.exchange_net_flow is None:
        net_flow_signal = 0.0
    elif data.exchange_net_flow > 0:
        net_flow_signal = 1.0
    else:
        net_flow_signal = -1.0

    return {
        "fear_greed_normalized": fear_greed_normalized,
        "funding_rate_mean": funding_rate_mean,
        "funding_rate_extreme": funding_rate_extreme,
        "net_flow_signal": net_flow_signal,
    }
