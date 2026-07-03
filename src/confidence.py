from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils import coerce_float


HIGH_CONFIDENCE_MIN_SAMPLE = 30
PRODUCTION_ELIGIBLE_STATUS = "Calibrated, production-eligible"


def calibration_status_label(
    sample_size: int,
    production_eligible: bool = False,
    review_only: bool = False,
) -> str:
    if production_eligible:
        return PRODUCTION_ELIGIBLE_STATUS
    if sample_size <= 0:
        return "Raw only"
    if sample_size < HIGH_CONFIDENCE_MIN_SAMPLE:
        return "Conservative shrinkage"
    if review_only:
        return "Calibrated, review-only"
    return "Raw only"


def model_market_benchmark_status(
    market_sample_size: int,
    raw_brier: Any = pd.NA,
    market_brier: Any = pd.NA,
    min_sample_size: int = 5,
) -> str:
    if market_sample_size <= 0:
        return "No market data"
    if market_sample_size < min_sample_size:
        return "Market data joined, sample too small"
    raw = coerce_float(raw_brier, float("nan"))
    market = coerce_float(market_brier, float("nan"))
    if pd.isna(raw) or pd.isna(market):
        return "Market data joined, sample too small"
    diff = raw - market
    if diff > 0.015:
        return "Model worse than market"
    if diff < -0.015:
        return "Model improves over market"
    return "Model similar to market"


def score_forecast_confidence(
    *,
    calibration_sample_size: int,
    calibration_status: str,
    market_join_status: str,
    behavior_disagreement_pp: Any = pd.NA,
    market_disagreement_pp: Any = pd.NA,
    venue_context: str = "",
    venue_uncertainty: bool = False,
    data_recency_hours: Any = pd.NA,
    team_sample_size: Any = pd.NA,
    match_round: str = "",
    draw_probability: Any = pd.NA,
    favorite_probability: Any = pd.NA,
    injuries_news_integrated: bool = False,
) -> dict[str, Any]:
    """Evidence-based forecast confidence.

    This is a product-readiness confidence score, not a betting signal. It is
    deliberately conservative until calibration, market joins, and diagnostic
    disagreement have enough history.
    """

    score = 100.0
    caps = [100.0]
    flags: list[str] = []
    reasons: list[str] = []

    status = str(calibration_status or "")
    if status != PRODUCTION_ELIGIBLE_STATUS:
        score -= 25.0
        caps.append(60.0)
        flags.append("calibration_not_production_eligible")
        reasons.append("No High confidence without production-eligible calibrated probabilities.")

    if int(calibration_sample_size or 0) < HIGH_CONFIDENCE_MIN_SAMPLE:
        score -= 20.0
        caps.append(60.0)
        flags.append("calibration_sample_too_small")
        reasons.append("Calibration sample too small; using conservative shrinkage.")

    join_status = str(market_join_status or "")
    if join_status != "joined_price":
        score -= 15.0
        caps.append(60.0)
        flags.append("market_not_joined")
        reasons.append("No High confidence when market mapping or usable prices are missing.")

    behavior_gap = coerce_float(behavior_disagreement_pp, float("nan"))
    if pd.notna(behavior_gap):
        if behavior_gap >= 20.0:
            score -= 25.0
            caps.append(45.0)
            flags.append("large_behavior_disagreement")
            reasons.append("Large behavior disagreement requires a low-confidence downgrade.")
        elif behavior_gap >= 10.0:
            score -= 12.0
            caps.append(65.0)
            flags.append("behavior_disagreement")
            reasons.append("Behavior disagreement is large enough to cap confidence.")

    market_gap = coerce_float(market_disagreement_pp, float("nan"))
    if pd.notna(market_gap):
        if market_gap >= 20.0:
            score -= 18.0
            caps.append(60.0)
            flags.append("large_market_disagreement")
            reasons.append("Large model-market disagreement requires caution.")
        elif market_gap >= 10.0:
            score -= 8.0
            caps.append(70.0)
            flags.append("market_disagreement")
            reasons.append("Model-market disagreement is material.")
    elif join_status != "joined_price":
        flags.append("market_disagreement_unavailable")

    if pd.notna(behavior_gap) and pd.notna(market_gap) and behavior_gap >= 20.0 and market_gap >= 20.0:
        caps.append(40.0)
        flags.append("behavior_and_market_disagree")
        reasons.append("Both behavior and market diagnostics disagree strongly with the raw model.")

    venue_text = str(venue_context or "").strip().lower()
    if venue_uncertainty or venue_text in {"home_host", "away_host", "designated_home", "unknown"}:
        score -= 10.0
        caps.append(60.0)
        flags.append("venue_context_uncertain")
        reasons.append("Venue/home-advantage context is uncertain or materially non-neutral.")

    recency = coerce_float(data_recency_hours, float("nan"))
    if pd.isna(recency):
        score -= 5.0
        flags.append("data_recency_unknown")
    elif recency > 24.0:
        score -= 10.0
        flags.append("stale_data")
        reasons.append("Market or input data is stale.")

    sample = coerce_float(team_sample_size, float("nan"))
    if pd.notna(sample) and sample < 10.0:
        score -= 12.0
        caps.append(60.0)
        flags.append("team_sample_small")
        reasons.append("Team sample size is too small for High confidence.")

    round_text = str(match_round or "").lower()
    if any(token in round_text for token in ["round of", "knockout", "quarter", "semi", "final"]):
        score -= 5.0
        flags.append("knockout_match")
        reasons.append("Knockout matches add draw/advancement interpretation risk.")

    draw_prob = coerce_float(draw_probability, float("nan"))
    if pd.notna(draw_prob) and draw_prob >= 0.28:
        score -= 8.0
        flags.append("high_draw_probability")
        reasons.append("High draw probability increases 1X2 forecast uncertainty.")

    fav = coerce_float(favorite_probability, float("nan"))
    if pd.notna(fav) and fav >= 0.75 and status != PRODUCTION_ELIGIBLE_STATUS:
        score -= 8.0
        caps.append(60.0)
        flags.append("extreme_favorite_without_calibration")
        reasons.append("Extreme probabilities require strong calibration evidence.")

    if not injuries_news_integrated:
        score -= 3.0
        flags.append("injuries_news_not_integrated")

    capped_score = max(0.0, min(score, min(caps)))
    if capped_score >= 75.0:
        label = "High"
    elif capped_score >= 50.0:
        label = "Moderate"
    else:
        label = "Low"

    if not reasons:
        reasons.append("No major confidence caps were triggered.")

    return {
        "confidence_score": round(float(capped_score), 1),
        "confidence_label": label,
        "confidence_reason": " ".join(reasons),
        "confidence_flags": "; ".join(dict.fromkeys(flags)),
    }
