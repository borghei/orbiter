"""Tests for sentiment-enhanced regime detection."""

import numpy as np
import pandas as pd
import pytest

from orbiter.regime import REGIME_STRATEGY_MAP, Regime, SentimentRegimeModel


@pytest.fixture
def regime_returns():
    """Synthetic data with clear bull/bear/sideways regimes."""
    rng = np.random.RandomState(42)
    dates = pd.date_range("2023-01-01", periods=300, freq="D", tz="UTC")

    bear = rng.randn(100) * 0.04 - 0.01
    sideways = rng.randn(100) * 0.01
    bull = rng.randn(100) * 0.03 + 0.01

    returns = np.concatenate([bear, sideways, bull])
    return pd.Series(returns, index=dates, name="market")


@pytest.fixture
def sentiment_df(regime_returns):
    """Synthetic sentiment features aligned to regime_returns."""
    rng = np.random.RandomState(99)
    n = len(regime_returns)
    data = {
        "fear_greed_normalized": rng.uniform(0.1, 0.9, n),
        "funding_rate_mean": rng.normal(0.0, 0.0005, n),
        "funding_rate_extreme": (rng.uniform(0, 1, n) > 0.8).astype(float),
        "net_flow_signal": rng.choice([-1.0, 0.0, 1.0], n),
    }
    return pd.DataFrame(data, index=regime_returns.index)


class TestSentimentRegimeFitReturnsOnly:
    def test_fit_returns_only(self, regime_returns):
        model = SentimentRegimeModel(n_regimes=3)
        model.fit(regime_returns)
        assert model.model is not None

    def test_predict_shape(self, regime_returns):
        model = SentimentRegimeModel(n_regimes=3)
        model.fit(regime_returns)
        predictions = model.predict(regime_returns)
        assert len(predictions) == len(regime_returns)

    def test_predict_returns_regime_enum(self, regime_returns):
        model = SentimentRegimeModel(n_regimes=3)
        model.fit(regime_returns)
        predictions = model.predict(regime_returns)
        assert set(predictions.unique()).issubset({Regime.BEAR, Regime.SIDEWAYS, Regime.BULL})


class TestSentimentRegimeFitWithSentiment:
    def test_fit_with_sentiment(self, regime_returns, sentiment_df):
        model = SentimentRegimeModel(n_regimes=3)
        model.fit(regime_returns, sentiment_series=sentiment_df)
        assert model.model is not None
        assert model._n_features > 1

    def test_predict_with_sentiment(self, regime_returns, sentiment_df):
        model = SentimentRegimeModel(n_regimes=3)
        model.fit(regime_returns, sentiment_series=sentiment_df)
        predictions = model.predict(regime_returns, sentiment_series=sentiment_df)
        assert len(predictions) == len(regime_returns)
        assert set(predictions.unique()).issubset({Regime.BEAR, Regime.SIDEWAYS, Regime.BULL})

    def test_multiple_regimes_detected(self, regime_returns, sentiment_df):
        model = SentimentRegimeModel(n_regimes=3)
        model.fit(regime_returns, sentiment_series=sentiment_df)
        predictions = model.predict(regime_returns, sentiment_series=sentiment_df)
        assert len(predictions.unique()) >= 2


class TestCurrentRegime:
    def test_returns_valid_regime(self, regime_returns):
        model = SentimentRegimeModel(n_regimes=3)
        model.fit(regime_returns)
        current = model.current_regime(regime_returns)
        assert isinstance(current, Regime)
        assert current in {Regime.BEAR, Regime.SIDEWAYS, Regime.BULL}

    def test_with_sentiment(self, regime_returns, sentiment_df):
        model = SentimentRegimeModel(n_regimes=3)
        model.fit(regime_returns, sentiment_series=sentiment_df)
        current = model.current_regime(regime_returns, sentiment_series=sentiment_df)
        assert isinstance(current, Regime)


class TestGetStrategy:
    def test_returns_valid_strategy(self, regime_returns):
        model = SentimentRegimeModel(n_regimes=3)
        model.fit(regime_returns)
        for regime in Regime:
            strategy = model.get_strategy(regime)
            assert strategy in ["max-sharpe", "min-vol", "risk-parity", "hrp"]

    def test_strategy_map_complete(self):
        for regime in Regime:
            assert regime in REGIME_STRATEGY_MAP


class TestPredictWithoutFit:
    def test_raises_runtime_error(self):
        model = SentimentRegimeModel()
        data = pd.Series([0.01] * 100)
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(data)
