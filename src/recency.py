from __future__ import annotations

import math
from typing import Any

import pandas as pd

from src.behavior_calibration import DEFAULT_BEHAVIOR_CONFIG, BehaviorConfig, competition_weight_for_type


COMPETITION_WEIGHTS = {
    "world_cup": 1.50,
    "world cup": 1.50,
    "fifa world cup": 1.50,
    "world_cup_qualifier": 1.15,
    "world cup qualifier": 1.15,
    "qualifier": 1.15,
    "continental_tournament": 1.25,
    "continental tournament": 1.25,
    "euro": 1.25,
    "copa america": 1.25,
    "afcon": 1.25,
    "asian cup": 1.25,
    "nations_league": 0.90,
    "nations league": 0.90,
    "friendly": 0.45,
    "other": 0.80,
}


def calculate_recency_weight(
    match_date: Any,
    reference_date: Any,
    half_life_days: float = 365,
) -> float:
    """Return exponential recency weight for a match date.

    The project spec uses `exp(-days_since_match / half_life_days)`. Future
    dates are treated as same-day rather than overweighted.
    """
    try:
        match_ts = pd.Timestamp(match_date)
        reference_ts = pd.Timestamp(reference_date)
    except Exception:
        return 0.0
    if pd.isna(match_ts) or pd.isna(reference_ts):
        return 0.0
    if match_ts.tzinfo is not None:
        match_ts = match_ts.tz_convert("UTC").tz_localize(None)
    if reference_ts.tzinfo is not None:
        reference_ts = reference_ts.tz_convert("UTC").tz_localize(None)
    days_since = max((reference_ts.normalize() - match_ts.normalize()).days, 0)
    half_life = max(float(half_life_days), 1.0)
    return float(math.exp(-days_since / half_life))


def competition_weight(competition_type: str, config: BehaviorConfig | None = None) -> float:
    if config is not None:
        return competition_weight_for_type(competition_type, config)
    key = str(competition_type or "other").strip().lower().replace("-", "_")
    key = " ".join(key.replace("_", " ").split())
    return float(COMPETITION_WEIGHTS.get(key, COMPETITION_WEIGHTS["other"]))


def calculate_match_weight(
    row: pd.Series | dict[str, Any],
    reference_date: Any,
    config: BehaviorConfig | None = None,
) -> float:
    """Combine recency and competition weight for a match."""
    cfg = config or DEFAULT_BEHAVIOR_CONFIG
    match_date = row.get("date_utc") if hasattr(row, "get") else None
    comp_type = row.get("competition_type", "other") if hasattr(row, "get") else "other"
    return float(calculate_recency_weight(match_date, reference_date, cfg.half_life_days) * competition_weight(str(comp_type), cfg))
