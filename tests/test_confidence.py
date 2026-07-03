from __future__ import annotations

from src.confidence import calibration_status_label, model_market_benchmark_status, score_forecast_confidence


def test_extreme_favorite_without_production_calibration_cannot_be_high() -> None:
    result = score_forecast_confidence(
        calibration_sample_size=8,
        calibration_status="Conservative shrinkage",
        market_join_status="joined_price",
        favorite_probability=0.88,
        draw_probability=0.07,
    )

    assert result["confidence_label"] != "High"
    assert result["confidence_score"] <= 60
    assert "extreme_favorite_without_calibration" in result["confidence_flags"]


def test_missing_market_caps_confidence() -> None:
    result = score_forecast_confidence(
        calibration_sample_size=80,
        calibration_status="Calibrated, production-eligible",
        market_join_status="no_event_resolved",
        favorite_probability=0.55,
        draw_probability=0.25,
    )

    assert result["confidence_label"] != "High"
    assert result["confidence_score"] <= 60
    assert "market_not_joined" in result["confidence_flags"]


def test_large_behavior_disagreement_downgrades_to_low() -> None:
    result = score_forecast_confidence(
        calibration_sample_size=80,
        calibration_status="Calibrated, production-eligible",
        market_join_status="joined_price",
        behavior_disagreement_pp=31,
        market_disagreement_pp=4,
    )

    assert result["confidence_label"] == "Low"
    assert "large_behavior_disagreement" in result["confidence_flags"]


def test_venue_uncertainty_downgrades_confidence() -> None:
    result = score_forecast_confidence(
        calibration_sample_size=80,
        calibration_status="Calibrated, production-eligible",
        market_join_status="joined_price",
        venue_context="home_host",
    )

    assert result["confidence_score"] <= 60
    assert "venue_context_uncertain" in result["confidence_flags"]


def test_sufficient_calibrated_evidence_allows_high() -> None:
    result = score_forecast_confidence(
        calibration_sample_size=80,
        calibration_status="Calibrated, production-eligible",
        market_join_status="joined_price",
        behavior_disagreement_pp=2,
        market_disagreement_pp=3,
        venue_context="neutral",
        data_recency_hours=2,
        team_sample_size=30,
        draw_probability=0.20,
        favorite_probability=0.62,
        injuries_news_integrated=True,
    )

    assert result["confidence_label"] == "High"


def test_ui_status_strings_are_explicit() -> None:
    assert calibration_status_label(0) == "Raw only"
    assert calibration_status_label(3) == "Conservative shrinkage"
    assert calibration_status_label(40, review_only=True) == "Calibrated, review-only"
    assert calibration_status_label(40, production_eligible=True) == "Calibrated, production-eligible"
    assert model_market_benchmark_status(0) == "No market data"
    assert model_market_benchmark_status(2) == "Market data joined, sample too small"
    assert model_market_benchmark_status(10, raw_brier=0.30, market_brier=0.25) == "Model worse than market"
