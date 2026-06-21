from __future__ import annotations

from typing import Any

import pandas as pd

from src.model_policy import get_current_model_policy, model_policy_label
from src.utils import coerce_bool, coerce_float


ALPHA_COLUMNS = [
    "match_id",
    "market_id",
    "question",
    "market_type",
    "model_side",
    "polymarket_side",
    "model_probability",
    "primary_model_probability",
    "behavior_diagnostic_probability",
    "behavior_probability_delta",
    "model_policy",
    "edge_source",
    "fair_price_cents",
    "polymarket_price_cents",
    "alpha_gap_cents",
    "alpha_ev",
    "liquidity",
    "volume",
    "mapping_confidence",
    "signal_strength",
    "warning",
]


def calculate_polymarket_alpha(
    model_result: dict,
    mapped_markets: pd.DataFrame,
    min_liquidity: float = 100.0,
) -> pd.DataFrame:
    if mapped_markets is None or mapped_markets.empty:
        return pd.DataFrame(columns=ALPHA_COLUMNS)

    policy = get_current_model_policy()
    rows = []
    probs = model_result.get("probs", {})
    for _, mapping in mapped_markets.iterrows():
        model_side = str(mapping.get("model_side", ""))
        polymarket_side = str(mapping.get("polymarket_side", "YES")).upper()
        base_probability = model_probability_for_side(probs, model_side, mapping.get("market_type", ""))
        if pd.isna(base_probability):
            model_probability = pd.NA
        elif polymarket_side == "NO":
            model_probability = 1.0 - float(base_probability)
        else:
            model_probability = float(base_probability)

        yes_price = _price_probability(mapping.get("yes_price"))
        no_price = _price_probability(mapping.get("no_price"))
        market_probability = no_price if polymarket_side == "NO" else yes_price

        fair_price = (float(model_probability) * 100.0) if pd.notna(model_probability) else pd.NA
        market_price = (float(market_probability) * 100.0) if pd.notna(market_probability) else pd.NA
        alpha_gap = (float(fair_price) - float(market_price)) if pd.notna(fair_price) and pd.notna(market_price) else pd.NA
        alpha_ev = (
            float(model_probability) / float(market_probability) - 1.0
            if pd.notna(model_probability) and pd.notna(market_probability) and float(market_probability) > 0
            else pd.NA
        )

        liquidity = coerce_float(mapping.get("liquidity"), float("nan"))
        volume = coerce_float(mapping.get("volume"), float("nan"))
        warning = alpha_warnings(mapping, yes_price, no_price, liquidity, min_liquidity)
        signal = classify_signal(alpha_gap, mapping.get("mapping_confidence", ""), liquidity, warning, min_liquidity)

        rows.append(
            {
                "match_id": model_result.get("match_id", ""),
                "market_id": mapping.get("market_id", ""),
                "question": mapping.get("question", ""),
                "market_type": mapping.get("market_type", ""),
                "model_side": model_side,
                "polymarket_side": polymarket_side,
                "model_probability": model_probability,
                "primary_model_probability": model_probability,
                "behavior_diagnostic_probability": pd.NA,
                "behavior_probability_delta": pd.NA,
                "model_policy": model_policy_label(policy),
                "edge_source": "primary_model",
                "fair_price_cents": fair_price,
                "polymarket_price_cents": market_price,
                "alpha_gap_cents": alpha_gap,
                "alpha_ev": alpha_ev,
                "liquidity": liquidity,
                "volume": volume,
                "mapping_confidence": mapping.get("mapping_confidence", ""),
                "signal_strength": signal,
                "warning": warning,
            }
        )

    out = pd.DataFrame(rows, columns=ALPHA_COLUMNS)
    return out.sort_values(["alpha_gap_cents"], ascending=False, na_position="last").reset_index(drop=True)


def model_probability_for_side(probs: dict[str, Any], model_side: str, market_type: str = "") -> float | pd.NA:
    side = str(model_side or "").lower()
    market_type = str(market_type or "").lower()
    mapping = {
        "home_win": "home_win",
        "match_winner_home": "home_win",
        "away_win": "away_win",
        "match_winner_away": "away_win",
        "draw": "draw",
        "over_2_5": "over_2_5",
        "under_2_5": "under_2_5",
        "btts_yes": "btts_yes",
        "btts_no": "btts_no",
    }
    if side in {"home_not_win", "away_or_draw"}:
        return 1.0 - coerce_float(probs.get("home_win"), float("nan"))
    if side in {"away_not_win", "home_or_draw"}:
        return 1.0 - coerce_float(probs.get("away_win"), float("nan"))
    key = mapping.get(side) or mapping.get(market_type)
    if not key:
        return pd.NA
    value = probs.get(key, pd.NA)
    return coerce_float(value, float("nan")) if pd.notna(value) else pd.NA


def classify_signal(
    alpha_gap_cents: Any,
    mapping_confidence: Any,
    liquidity: float,
    warning: str,
    min_liquidity: float = 100.0,
) -> str:
    if pd.isna(alpha_gap_cents):
        return "No signal"
    gap = float(alpha_gap_cents)
    confidence = str(mapping_confidence).lower()
    high_confidence = confidence in {"high", "manual"}
    liquidity_ok = pd.notna(liquidity) and liquidity >= min_liquidity
    if gap >= 8.0 and high_confidence and liquidity_ok:
        return "Strong"
    if gap >= 5.0:
        return "Moderate"
    if gap >= 3.0:
        return "Weak"
    if -3.0 <= gap <= 3.0:
        return "No signal"
    return "Avoid"


def alpha_warnings(
    mapping: pd.Series,
    yes_price: Any,
    no_price: Any,
    liquidity: float,
    min_liquidity: float,
) -> str:
    warnings = []
    if pd.isna(yes_price) and pd.isna(no_price):
        warnings.append("price missing")
    if pd.isna(liquidity) or liquidity < min_liquidity:
        warnings.append("low liquidity")
    if pd.notna(yes_price) and pd.notna(no_price):
        price_sum = float(yes_price) + float(no_price)
        if price_sum < 0.92 or price_sum > 1.08:
            warnings.append("wide spread likely")
    if not coerce_bool(mapping.get("manual_confirmed", False)):
        warnings.append("manual confirmation required")
    if str(mapping.get("mapping_confidence", "")).lower() == "low":
        warnings.append("mapping uncertain")
    if coerce_bool(mapping.get("closed", False)):
        warnings.append("market closed")
    return "; ".join(dict.fromkeys(warnings))


def _price_probability(value: Any) -> float | pd.NA:
    price = coerce_float(value, float("nan"))
    if pd.isna(price):
        return pd.NA
    if price > 1.0:
        price = price / 100.0
    if price < 0.0 or price > 1.0:
        return pd.NA
    return float(price)
