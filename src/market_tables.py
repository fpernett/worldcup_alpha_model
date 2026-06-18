from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils import coerce_float


MARKET_GROUPS = [
    "High Scoring",
    "Low Scoring",
    "Result Home",
    "Result Away",
    "Other Markets",
    "Polymarket Alpha",
]


def group_market_alpha(
    alpha_df: pd.DataFrame | None,
    polymarket_alpha_df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    groups = {name: _empty_group() for name in MARKET_GROUPS}

    alpha = alpha_df.copy() if alpha_df is not None else pd.DataFrame()
    for _, row in alpha.iterrows():
        group = _decimal_group(row)
        out_row = _decimal_market_row(row)
        groups[group] = pd.concat([groups[group], pd.DataFrame([out_row])], ignore_index=True)

    poly = polymarket_alpha_df.copy() if polymarket_alpha_df is not None else pd.DataFrame()
    for _, row in poly.iterrows():
        groups["Polymarket Alpha"] = pd.concat(
            [groups["Polymarket Alpha"], pd.DataFrame([_polymarket_row(row)])],
            ignore_index=True,
        )

    return {key: value.sort_values("Score", ascending=False).reset_index(drop=True) for key, value in groups.items()}


def _empty_group() -> pd.DataFrame:
    return pd.DataFrame(columns=["Market", "Odds / Price", "Fair Odds / Fair Price", "EV / Alpha Gap", "Score", "Signal"])


def _decimal_group(row: pd.Series) -> str:
    market = str(row.get("market", ""))
    selection = str(row.get("selection", ""))
    selection_l = selection.lower()

    if market == "Total" and "over" in selection_l:
        return "High Scoring"
    if market == "BTTS" and selection_l == "yes":
        return "High Scoring"
    if market == "Total" and "under" in selection_l:
        return "Low Scoring"
    if market == "BTTS" and selection_l == "no":
        return "Low Scoring"
    if market in {"1X2", "Double Chance", "Handicap"}:
        if "draw" in selection_l and "/" not in selection_l:
            return "Other Markets"
        if "away" in selection_l or "draw/" in selection_l:
            return "Result Away"
        if "home" in selection_l or "/draw" in selection_l:
            return "Result Home"
    return "Other Markets"


def _decimal_market_row(row: pd.Series) -> dict[str, Any]:
    ev = coerce_float(row.get("alpha_ev"), float("nan"))
    score = _ev_score(ev)
    signal = _signal_from_score(score, ev)
    market = str(row.get("market", ""))
    selection = str(row.get("selection", ""))
    return {
        "Market": f"{market}: {selection}".strip(": "),
        "Odds / Price": _fmt_decimal(row.get("market_odds")),
        "Fair Odds / Fair Price": _fmt_decimal(row.get("fair_odds")),
        "EV / Alpha Gap": "" if pd.isna(ev) else f"{100 * ev:.1f}%",
        "Score": score,
        "Signal": signal,
    }


def _polymarket_row(row: pd.Series) -> dict[str, Any]:
    gap = coerce_float(row.get("alpha_gap_cents"), float("nan"))
    confidence = _confidence_modifier(row.get("mapping_confidence"))
    liquidity = _liquidity_modifier(row.get("liquidity"))
    score = _gap_score(gap, confidence, liquidity)
    side = str(row.get("polymarket_side", "YES"))
    question = str(row.get("question", row.get("market_id", "")))
    return {
        "Market": f"{side}: {question}".strip(": "),
        "Odds / Price": _fmt_cents(row.get("polymarket_price_cents")),
        "Fair Odds / Fair Price": _fmt_cents(row.get("fair_price_cents")),
        "EV / Alpha Gap": "" if pd.isna(gap) else f"{gap:.1f} cents",
        "Score": score,
        "Signal": str(row.get("signal_strength") or _signal_from_gap(gap)),
    }


def _ev_score(ev: float) -> int:
    if pd.isna(ev) or ev <= 0:
        return 0
    return int(min(100, max(0, ev * 400)))


def _gap_score(gap: float, confidence_modifier: float, liquidity_modifier: float) -> int:
    if pd.isna(gap) or gap <= 0:
        return 0
    return int(min(100, max(0, gap * 10 * confidence_modifier * liquidity_modifier)))


def _confidence_modifier(value: Any) -> float:
    text = str(value or "").lower()
    if text in {"high", "manual"}:
        return 1.0
    if text == "low":
        return 0.65
    return 0.75


def _liquidity_modifier(value: Any) -> float:
    liquidity = coerce_float(value, float("nan"))
    if pd.isna(liquidity):
        return 0.65
    return float(min(1.0, max(0.35, liquidity / 500.0)))


def _signal_from_score(score: int, ev: float) -> str:
    if pd.isna(ev):
        return "No signal"
    if ev < 0:
        return "Avoid"
    if score >= 70:
        return "Strong"
    if score >= 40:
        return "Moderate"
    if score >= 15:
        return "Weak"
    return "No signal"


def _signal_from_gap(gap: float) -> str:
    if pd.isna(gap):
        return "No signal"
    if gap >= 8:
        return "Strong"
    if gap >= 5:
        return "Moderate"
    if gap >= 3:
        return "Weak"
    if gap < -3:
        return "Avoid"
    return "No signal"


def _fmt_decimal(value: Any) -> str:
    number = coerce_float(value, float("nan"))
    if pd.isna(number):
        return ""
    return f"{number:.2f}"


def _fmt_cents(value: Any) -> str:
    number = coerce_float(value, float("nan"))
    if pd.isna(number):
        return ""
    return f"{number:.1f}c"
