"""Tests for sentiment data collection."""

from unittest.mock import MagicMock, patch

import pytest

from orbiter.sentiment import (
    SentimentCollector,
    SentimentData,
    _label_from_value,
    sentiment_features,
)


@pytest.fixture
def sample_sentiment():
    return SentimentData(
        fear_greed_index=45,
        fear_greed_label="Fear",
        funding_rates={"BTC": 0.0001, "ETH": -0.0002},
        exchange_net_flow=500.0,
        timestamp="2024-06-01T00:00:00+00:00",
    )


class TestSentimentData:
    def test_creation(self, sample_sentiment):
        assert sample_sentiment.fear_greed_index == 45
        assert sample_sentiment.fear_greed_label == "Fear"
        assert len(sample_sentiment.funding_rates) == 2
        assert sample_sentiment.exchange_net_flow == 500.0

    def test_none_exchange_net_flow(self):
        data = SentimentData(
            fear_greed_index=50,
            fear_greed_label="Neutral",
            funding_rates={},
            exchange_net_flow=None,
            timestamp="",
        )
        assert data.exchange_net_flow is None


class TestSentimentFeatures:
    def test_normal_data(self, sample_sentiment):
        features = sentiment_features(sample_sentiment)
        assert "fear_greed_normalized" in features
        assert "funding_rate_mean" in features
        assert "funding_rate_extreme" in features
        assert "net_flow_signal" in features
        assert features["fear_greed_normalized"] == pytest.approx(0.45)

    def test_none_exchange_net_flow(self):
        data = SentimentData(
            fear_greed_index=50,
            fear_greed_label="Neutral",
            funding_rates={"BTC": 0.0001},
            exchange_net_flow=None,
            timestamp="",
        )
        features = sentiment_features(data)
        assert features["net_flow_signal"] == 0.0

    def test_empty_funding_rates(self):
        data = SentimentData(
            fear_greed_index=30,
            fear_greed_label="Fear",
            funding_rates={},
            exchange_net_flow=-100.0,
            timestamp="",
        )
        features = sentiment_features(data)
        assert features["funding_rate_mean"] == 0.0
        assert features["funding_rate_extreme"] == 0.0
        assert features["net_flow_signal"] == -1.0

    def test_positive_net_flow(self, sample_sentiment):
        features = sentiment_features(sample_sentiment)
        assert features["net_flow_signal"] == 1.0


class TestLabelFromValue:
    def test_extreme_fear(self):
        assert _label_from_value(10) == "Extreme Fear"
        assert _label_from_value(20) == "Extreme Fear"

    def test_fear(self):
        assert _label_from_value(21) == "Fear"
        assert _label_from_value(40) == "Fear"

    def test_neutral(self):
        assert _label_from_value(41) == "Neutral"
        assert _label_from_value(60) == "Neutral"

    def test_greed(self):
        assert _label_from_value(61) == "Greed"
        assert _label_from_value(80) == "Greed"

    def test_extreme_greed(self):
        assert _label_from_value(81) == "Extreme Greed"
        assert _label_from_value(100) == "Extreme Greed"


class TestGetFearGreed:
    @patch("orbiter.sentiment.requests.get")
    def test_successful_response(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [{"value": "72", "timestamp": "1717200000"}]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        collector = SentimentCollector()
        result = collector.get_fear_greed()
        assert result["value"] == 72
        assert result["label"] == "Greed"

    @patch("orbiter.sentiment.requests.get")
    def test_failed_response_returns_default(self, mock_get):
        mock_get.side_effect = Exception("connection error")

        collector = SentimentCollector()
        result = collector.get_fear_greed()
        assert result["value"] == 50
        assert result["label"] == "Neutral"
