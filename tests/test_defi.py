"""Tests for DeFi yield data."""

import pandas as pd
import pytest

from orbiter.defi import (
    MANUAL_STAKING_APYS,
    YieldCollector,
    YieldInfo,
    adjust_expected_returns,
    yield_risk_adjustment,
)


@pytest.fixture
def sample_yields():
    return {
        "ETH": YieldInfo(symbol="ETH", staking_apy=0.035, best_yield=0.035),
        "SOL": YieldInfo(symbol="SOL", staking_apy=0.065, best_yield=0.065),
        "BTC": YieldInfo(symbol="BTC", staking_apy=0.005, best_yield=0.005),
        "AVAX": YieldInfo(symbol="AVAX", staking_apy=0.080, best_yield=0.080),
    }


@pytest.fixture
def sample_mu():
    return pd.Series(
        [0.001, 0.0015, 0.002, 0.0012],
        index=["BTC", "ETH", "SOL", "AVAX"],
    )


class TestYieldInfo:
    def test_creation(self):
        info = YieldInfo(symbol="ETH", staking_apy=0.035, lending_apy=0.02)
        assert info.symbol == "ETH"
        assert info.staking_apy == 0.035
        assert info.lending_apy == 0.02
        assert info.best_yield == 0.0
        assert info.source == "manual"

    def test_defaults(self):
        info = YieldInfo(symbol="BTC")
        assert info.staking_apy == 0.0
        assert info.lending_apy == 0.0
        assert info.best_yield == 0.0
        assert info.protocol == ""


class TestGetManualYields:
    def test_returns_known_values(self):
        collector = YieldCollector()
        manual = collector.get_manual_yields()
        assert "ETH" in manual
        assert manual["ETH"].staking_apy == MANUAL_STAKING_APYS["ETH"]
        assert manual["SOL"].staking_apy == MANUAL_STAKING_APYS["SOL"]
        assert manual["ETH"].source == "manual"

    def test_all_manual_apys_present(self):
        collector = YieldCollector()
        manual = collector.get_manual_yields()
        for sym in MANUAL_STAKING_APYS:
            assert sym in manual


class TestAdjustExpectedReturns:
    def test_adds_daily_yield(self, sample_mu, sample_yields):
        adjusted = adjust_expected_returns(sample_mu, sample_yields, weight=1.0)
        for sym in sample_mu.index:
            expected_bump = sample_yields[sym].best_yield / 365.0
            assert adjusted[sym] == pytest.approx(
                sample_mu[sym] + expected_bump, abs=1e-10
            )

    def test_weight_zero_returns_original(self, sample_mu, sample_yields):
        adjusted = adjust_expected_returns(sample_mu, sample_yields, weight=0.0)
        pd.testing.assert_series_equal(adjusted, sample_mu)

    def test_partial_weight(self, sample_mu, sample_yields):
        adjusted = adjust_expected_returns(sample_mu, sample_yields, weight=0.5)
        for sym in sample_mu.index:
            expected_bump = sample_yields[sym].best_yield / 365.0 * 0.5
            assert adjusted[sym] == pytest.approx(
                sample_mu[sym] + expected_bump, abs=1e-10
            )


class TestYieldRiskAdjustment:
    def test_scores_decrease_with_higher_apy(self):
        yields_low = {"ETH": YieldInfo(symbol="ETH", staking_apy=0.03, best_yield=0.03)}
        yields_high = {"ETH": YieldInfo(symbol="ETH", staking_apy=0.50, best_yield=0.50)}
        scores_low = yield_risk_adjustment(yields_low)
        scores_high = yield_risk_adjustment(yields_high)
        assert scores_low["ETH"] > scores_high["ETH"]

    def test_no_yield_returns_one(self):
        yields = {"BTC": YieldInfo(symbol="BTC")}
        scores = yield_risk_adjustment(yields)
        assert scores["BTC"] == 1.0

    def test_scores_in_range(self, sample_yields):
        scores = yield_risk_adjustment(sample_yields)
        for sym, score in scores.items():
            assert 0.0 <= score <= 1.0, f"Score for {sym} out of range: {score}"


class TestCollect:
    def test_fallback_to_manual(self):
        collector = YieldCollector()
        results = collector.collect(["ETH", "SOL"], use_live=False)
        assert "ETH" in results
        assert "SOL" in results
        assert results["ETH"].source == "manual"

    def test_unknown_symbol_gets_empty_yield(self):
        collector = YieldCollector()
        results = collector.collect(["FAKECOIN"], use_live=False)
        assert "FAKECOIN" in results
        assert results["FAKECOIN"].best_yield == 0.0
