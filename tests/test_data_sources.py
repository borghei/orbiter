"""Tests for data sources (mocked HTTP calls)."""

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from orbiter.data_sources import CoinGeckoClient, OnChainMetrics


@pytest.fixture
def mock_top_coins_response():
    return [
        {
            "id": "bitcoin",
            "symbol": "btc",
            "name": "Bitcoin",
            "market_cap": 1_000_000_000_000,
            "market_cap_rank": 1,
            "current_price": 50000,
            "total_volume": 30_000_000_000,
        },
        {
            "id": "ethereum",
            "symbol": "eth",
            "name": "Ethereum",
            "market_cap": 400_000_000_000,
            "market_cap_rank": 2,
            "current_price": 3000,
            "total_volume": 15_000_000_000,
        },
    ]


def test_top_coins_shape(mock_top_coins_response):
    client = CoinGeckoClient()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_top_coins_response

    with patch.object(client._session, "get", return_value=mock_resp):
        df = client.top_coins(n=10)

    assert len(df) == 2
    assert "symbol" in df.columns
    assert "market_cap" in df.columns
    assert df.iloc[0]["symbol"] == "BTC"


def test_get_market_caps(mock_top_coins_response):
    client = CoinGeckoClient()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_top_coins_response

    with patch.object(client._session, "get", return_value=mock_resp):
        caps = client.get_market_caps(["BTC", "ETH"])

    assert "BTC" in caps.index
    assert caps["BTC"] == 1_000_000_000_000


def test_onchain_unsupported_returns_none():
    on_chain = OnChainMetrics()
    assert on_chain.get_active_addresses("DOGE") is None
    assert on_chain.get_nvt_ratio("SOL") is None


def test_onchain_supported_symbols():
    assert "BTC" in OnChainMetrics.SUPPORTED
    assert "ETH" in OnChainMetrics.SUPPORTED
