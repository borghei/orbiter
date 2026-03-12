"""Tests for crypto factor model."""

import numpy as np
import pandas as pd
import pytest

from orbiter.factors import CryptoFactorModel, FactorExposures


@pytest.fixture
def factor_data(sample_returns):
    """Returns, market caps, and volumes for factor model testing."""
    caps = pd.Series(
        [800e9, 300e9, 50e9, 10e9],
        index=sample_returns.columns,
        name="market_cap",
    )
    volumes = pd.DataFrame(
        np.random.RandomState(99).uniform(1e6, 1e9, size=sample_returns.shape),
        index=sample_returns.index,
        columns=sample_returns.columns,
    )
    return sample_returns, caps, volumes


def test_fit_returns_exposures(factor_data):
    returns, caps, volumes = factor_data
    model = CryptoFactorModel(returns, market_caps=caps, volumes=volumes)
    exposures = model.fit()
    assert isinstance(exposures, FactorExposures)


def test_loadings_shape(factor_data):
    returns, caps, volumes = factor_data
    model = CryptoFactorModel(returns, market_caps=caps, volumes=volumes)
    exposures = model.fit()
    assert exposures.loadings.shape == (4, 4)
    assert list(exposures.loadings.columns) == ["market", "momentum", "size", "liquidity"]


def test_r_squared_range(factor_data):
    returns, caps, volumes = factor_data
    model = CryptoFactorModel(returns, market_caps=caps, volumes=volumes)
    exposures = model.fit()
    for asset, r2 in exposures.r_squared.items():
        assert -0.5 <= r2 <= 1.0, f"R^2 for {asset} = {r2}"


def test_expected_returns_shape(factor_data):
    returns, caps, volumes = factor_data
    model = CryptoFactorModel(returns, market_caps=caps, volumes=volumes)
    model.fit()
    er = model.expected_returns()
    assert len(er) == 4


def test_factor_covariance_psd(factor_data):
    returns, caps, volumes = factor_data
    model = CryptoFactorModel(returns, market_caps=caps, volumes=volumes)
    model.fit()
    cov = model.factor_covariance()
    assert cov.shape == (4, 4)
    eigenvalues = np.linalg.eigvalsh(cov)
    assert np.all(eigenvalues >= -1e-8)


def test_works_without_caps_and_volumes(sample_returns):
    """Factor model should work with only returns (market + momentum only)."""
    model = CryptoFactorModel(sample_returns)
    exposures = model.fit()
    assert exposures.loadings.shape[0] == 4
    er = model.expected_returns()
    assert len(er) == 4


def test_custom_risk_premia(factor_data):
    returns, caps, volumes = factor_data
    model = CryptoFactorModel(returns, market_caps=caps, volumes=volumes)
    model.fit()
    premia = {"market": 0.10, "momentum": 0.05, "size": 0.02, "liquidity": 0.01}
    er = model.expected_returns(factor_risk_premia=premia)
    assert len(er) == 4
    assert not np.all(er == 0)
