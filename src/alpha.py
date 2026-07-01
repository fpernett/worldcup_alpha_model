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
    "market",
    "selection",
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
    "model_confidence",
    "signal_strength",
    "calibrated_probability_available",
    "historical_support",
    "signal_policy_reasons",
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
        confidence_label = ""
        if isinstance(model_result.get("confidence"), dict):
            confidence_label = str(model_result.get("confidence", {}).get("label", "") or "")
        calibrated_available = bool(model_result.get("calibrated_probability_available", False))
        historical_support = bool(model_result.get("historical_support", False))
        signal, signal_reasons = classify_signal(
            alpha_gap,
            mapping.get("mapping_confidence", ""),
            liquidity,
            warning,
            min_liquidity,
            model_confidence=confidence_label,
            behavior_probability_delta=mapping.get("behavior_probability_delta", pd.NA),
            calibrated_probability_available=calibrated_available,
            historical_support=historical_support,
        )
        market, selection = dashboard_market_selection(mapping)

        rows.append(
            {
                "match_id": model_result.get("match_id", ""),
                "market_id": mapping.get("market_id", ""),
                "question": mapping.get("question", ""),
                "market_type": mapping.get("market_type", ""),
                "market": market,
                "selection": selection,
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
                "model_confidence": confidence_label,
                "signal_strength": signal,
                "calibrated_probability_available": calibrated_available,
                "historical_support": historical_support,
                "signal_policy_reasons": signal_reasons,
                "warning": warning,
            }
        )

    out = pd.DataFrame(rows, columns=ALPHA_COLUMNS)
    return out.sort_values(["alpha_gap_cents"], ascending=False, na_position="last").reset_index(drop=True)


def dashboard_market_selection(mapping: pd.Series) -> tuple[str, str]:
    """Return the dashboard market/selection key for a mapped Polymarket row."""
    model_side = str(mapping.get("model_side", "") or "").lower()
    market_type = str(mapping.get("market_type", "") or "").lower()
    home = str(mapping.get("home", "") or "").strip()
    away = str(mapping.get("away", "") or "").strip()

    if model_side in {"home_win", "match_winner_home"} or market_type == "match_winner_home":
        return "1X2", home or "Home"
    if model_side in {"away_win", "match_winner_away"} or market_type == "match_winner_away":
        return "1X2", away or "Away"
    if model_side == "draw" or market_type == "draw":
        return "1X2", "Draw"
    if model_side == "over_2_5" or market_type == "over_2_5":
        return "Total", "Over 2.5"
    if model_side == "under_2_5" or market_type == "under_2_5":
        return "Total", "Under 2.5"
    if model_side == "btts_yes" or market_type == "btts_yes":
        return "BTTS", "Yes"
    if model_side == "btts_no" or market_type == "btts_no":
        return "BTTS", "No"
    return str(mapping.get("market_type", "") or ""), str(mapping.get("model_side", "") or "")


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
    model_confidence: Any = "",
    behavior_probability_delta: Any = pd.NA,
    calibrated_probability_available: bool = False,
    historical_support: bool = False,
) -> tuple[str, str]:
    reasons: list[str] = []
    if pd.isna(alpha_gap_cents):
        return "No trade", "missing alpha gap"
    gap = float(alpha_gap_cents)
    confidence = str(mapping_confidence).lower()
    high_confidence = confidence in {"high", "manual"}
    liquidity_ok = pd.notna(liquidity) and liquidity >= min_liquidity
    warning_text = str(warning or "").lower()
    if gap <= 0.0:
        return "No trade", "non-positive model gap"
    if gap < 3.0:
        return "No trade", "gap below watchlist threshold"
    if any(token in warning_text for token in ["mapping uncertain", "manual confirmation required", "market closed", "price missing"]):
        reasons.append("unresolved mapping or price status")
    if not liquidity_ok:
        reasons.append("insufficient liquidity")
    if not high_confidence:
        reasons.append("mapping confidence below high/manual")
    model_conf = str(model_confidence or "").strip().lower()
    if model_conf == "low":
        reasons.append("low model confidence")
    if not calibrated_probability_available:
        reasons.append("calibrated probability unavailable")
    if not historical_support:
        reasons.append("insufficient historical support")
    behavior_delta = coerce_float(behavior_probability_delta, float("nan"))
    if pd.notna(behavior_delta) and abs(behavior_delta) >= 0.10:
        reasons.append("large behavior disagreement")
    if gap >= 20.0 and not historical_support:
        reasons.append("extreme model-market gap without validation")

    if any(reason in reasons for reason in ["unresolved mapping or price status", "insufficient liquidity"]):
        return "No trade", "; ".join(dict.fromkeys(reasons))
    if reasons:
        return "Watchlist", "; ".join(dict.fromkeys(reasons))
    if gap >= 8.0:
        return "Strong edge", ""
    if gap >= 5.0:
        return "Small edge", ""
    return "Watchlist", ""


def apply_signal_policy(
    alpha_df: pd.DataFrame | None,
    model_confidence: Any = "",
    calibrated_probability_available: bool = False,
    historical_support: bool = False,
    min_liquidity: float = 100.0,
) -> pd.DataFrame:
    out = alpha_df.copy() if alpha_df is not None else pd.DataFrame(columns=ALPHA_COLUMNS)
    for col in ALPHA_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    if out.empty:
        return out
    signals = []
    reasons = []
    for _, row in out.iterrows():
        signal, reason = classify_signal(
            row.get("alpha_gap_cents", pd.NA),
            row.get("mapping_confidence", ""),
            coerce_float(row.get("liquidity"), float("nan")),
            str(row.get("warning", "") or ""),
            min_liquidity=min_liquidity,
            model_confidence=row.get("model_confidence", model_confidence),
            behavior_probability_delta=row.get("behavior_probability_delta", pd.NA),
            calibrated_probability_available=coerce_bool(row.get("calibrated_probability_available", calibrated_probability_available)),
            historical_support=coerce_bool(row.get("historical_support", historical_support)),
        )
        signals.append(signal)
        reasons.append(reason)
    out["signal_strength"] = signals
    out["signal_policy_reasons"] = reasons
    return out


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
