"""Tests for covariance estimation."""

import numpy as np
import pytest
from orbiter.covariance import (
    exponential_covariance,
    get_covariance,
    ledoit_wolf,
    sample_covariance,
)


def test_sample_covariance_shape(sample_returns):
    cov = sample_covariance(sample_returns)
    assert cov.shape == (4, 4)


def test_sample_covariance_symmetric(sample_returns):
    cov = sample_covariance(sample_returns)
    np.testing.assert_array_almost_equal(cov, cov.T)


def test_sample_covariance_psd(sample_returns):
    cov = sample_covariance(sample_returns)
    eigenvalues = np.linalg.eigvalsh(cov)
    assert np.all(eigenvalues >= -1e-10)


def test_ledoit_wolf_shape(sample_returns):
    cov = ledoit_wolf(sample_returns)
    assert cov.shape == (4, 4)


def test_ledoit_wolf_symmetric(sample_returns):
    cov = ledoit_wolf(sample_returns)
    np.testing.assert_array_almost_equal(cov, cov.T)


def test_ledoit_wolf_shrinkage(sample_returns):
    """Ledoit-Wolf should produce less extreme eigenvalues than sample."""
    sample = sample_covariance(sample_returns)
    lw = ledoit_wolf(sample_returns)

    sample_eig = np.linalg.eigvalsh(sample)
    lw_eig = np.linalg.eigvalsh(lw)

    # Shrinkage reduces spread of eigenvalues
    assert np.std(lw_eig) <= np.std(sample_eig) + 1e-10


def test_exponential_covariance_shape(sample_returns):
    cov = exponential_covariance(sample_returns)
    assert cov.shape == (4, 4)


def test_get_covariance_dispatcher(sample_returns):
    for method in ["sample", "ledoit-wolf", "exponential"]:
        cov = get_covariance(sample_returns, method=method)
        assert cov.shape == (4, 4)


def test_get_covariance_invalid_method(sample_returns):
    with pytest.raises(ValueError, match="Unknown method"):
        get_covariance(sample_returns, method="magic")
