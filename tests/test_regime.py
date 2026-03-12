"""Tests for regime detection."""

import numpy as np
import pandas as pd
import pytest

from orbiter.regime import REGIME_STRATEGY_MAP, Regime, RegimeModel


@pytest.fixture
def regime_data():
    """Synthetic data with clear bull/bear/sideways regimes."""
    rng = np.random.RandomState(42)
    dates = pd.date_range("2023-01-01", periods=300, freq="D", tz="UTC")

    # Bear: negative drift, high vol
    bear = rng.randn(100) * 0.04 - 0.01
    # Sideways: no drift, low vol
    sideways = rng.randn(100) * 0.01
    # Bull: positive drift, moderate vol
    bull = rng.randn(100) * 0.03 + 0.01

    returns = np.concatenate([bear, sideways, bull])
    return pd.Series(returns, index=dates, name="market")


def test_fit_predict_shape(regime_data):
    model = RegimeModel(n_regimes=3)
    model.fit(regime_data)
    predictions = model.predict(regime_data)
    assert len(predictions) == len(regime_data)


def test_predictions_are_regimes(regime_data):
    model = RegimeModel(n_regimes=3)
    model.fit(regime_data)
    predictions = model.predict(regime_data)
    assert set(predictions.unique()).issubset({Regime.BEAR, Regime.SIDEWAYS, Regime.BULL})


def test_multiple_regimes_detected(regime_data):
    model = RegimeModel(n_regimes=3)
    model.fit(regime_data)
    predictions = model.predict(regime_data)
    assert len(predictions.unique()) >= 2


def test_current_regime_returns_enum(regime_data):
    model = RegimeModel(n_regimes=3)
    model.fit(regime_data)
    current = model.current_regime(regime_data)
    assert isinstance(current, Regime)


def test_strategy_mapping():
    for regime in Regime:
        assert regime in REGIME_STRATEGY_MAP
        assert REGIME_STRATEGY_MAP[regime] in ["max-sharpe", "min-vol", "risk-parity", "hrp"]


def test_predict_without_fit():
    model = RegimeModel()
    data = pd.Series([0.01] * 100)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(data)


def test_get_strategy(regime_data):
    model = RegimeModel()
    model.fit(regime_data)
    regime = model.current_regime(regime_data)
    strategy = model.get_strategy(regime)
    assert strategy in ["max-sharpe", "min-vol", "risk-parity"]
