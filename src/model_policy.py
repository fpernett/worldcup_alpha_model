from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils import coerce_float


PRIMARY_MODEL_MODE = "baseline_manual"
SECONDARY_MODEL_MODE = "behavior_adjusted_asof"
BEHAVIOR_STATUS = "diagnostic_only"
BEHAVIOR_DEFAULT_BLEND = 0.00
STRICT_VALIDATION_REASON = (
    "Strict as-of-date blend sensitivity found no behavior blend that improved both Brier and log loss."
)
LAST_VALIDATED_REPORT = "reports/blend_sensitivity_report_2026-06-21.md"

POLICY_ALPHA_COLUMNS = [
    "primary_model_probability",
    "behavior_diagnostic_probability",
    "behavior_probability_delta",
    "model_policy",
    "edge_source",
]

POLICY_EXPORT_COLUMNS = [
    "model_policy",
    "primary_model_mode",
    "behavior_status",
    "behavior_blend_used",
    "strict_validation_summary",
]


def get_current_model_policy() -> dict[str, Any]:
    return {
        "primary_model_mode": PRIMARY_MODEL_MODE,
        "secondary_model_mode": SECONDARY_MODEL_MODE,
        "behavior_status": BEHAVIOR_STATUS,
        "behavior_default_blend": BEHAVIOR_DEFAULT_BLEND,
        "reason": STRICT_VALIDATION_REASON,
        "last_validated_report": LAST_VALIDATED_REPORT,
    }


def model_policy_label(policy: dict[str, Any] | None = None) -> str:
    active = policy or get_current_model_policy()
    return f"{active['primary_model_mode']} | behavior {active['behavior_status']}"


def policy_export_fields(
    policy: dict[str, Any] | None = None,
    behavior_blend_used: bool | float | int | str = False,
) -> dict[str, Any]:
    active = policy or get_current_model_policy()
    return {
        "model_policy": model_policy_label(active),
        "primary_model_mode": active["primary_model_mode"],
        "behavior_status": active["behavior_status"],
        "behavior_blend_used": bool(behavior_blend_used),
        "strict_validation_summary": active["reason"],
    }


def behavior_disagreement_warnings(
    primary_result: dict[str, Any] | None,
    behavior_result: dict[str, Any] | None,
    threshold: float = 0.05,
) -> list[str]:
    if not primary_result or not behavior_result:
        return []
    primary_probs = primary_result.get("probs", {}) or {}
    behavior_probs = behavior_result.get("probs", {}) or {}
    home = str(primary_result.get("home", "home team"))
    away = str(primary_result.get("away", "away team"))
    warnings: list[str] = []
    for key, team in [("home_win", home), ("away_win", away)]:
        primary_prob = coerce_float(primary_probs.get(key), float("nan"))
        behavior_prob = coerce_float(behavior_probs.get(key), float("nan"))
        if pd.isna(primary_prob) or pd.isna(behavior_prob):
            continue
        delta = behavior_prob - primary_prob
        if abs(delta) >= threshold:
            warnings.append(
                "Behavior disagreement: diagnostic only. "
                f"The behavior layer changes {team} win probability by {delta * 100:+.1f} percentage points, "
                "but strict backtesting has not validated behavior as a primary signal."
            )
    return warnings


def behavior_disagreement_warning(
    primary_result: dict[str, Any] | None,
    behavior_result: dict[str, Any] | None,
    threshold: float = 0.05,
) -> str:
    return " ".join(behavior_disagreement_warnings(primary_result, behavior_result, threshold=threshold))


def add_policy_columns_to_alpha(
    alpha_df: pd.DataFrame | None,
    probability_column: str,
    behavior_probability: pd.Series | None = None,
    show_behavior_diagnostic: bool = False,
    policy: dict[str, Any] | None = None,
) -> pd.DataFrame:
    out = alpha_df.copy() if alpha_df is not None else pd.DataFrame()
    if out.empty:
        for col in POLICY_ALPHA_COLUMNS:
            if col not in out.columns:
                out[col] = pd.NA
        return out

    if probability_column not in out.columns:
        out[probability_column] = pd.NA
    primary = pd.to_numeric(out[probability_column], errors="coerce")
    out["primary_model_probability"] = primary

    if behavior_probability is None:
        behavior = pd.Series([pd.NA] * len(out), index=out.index, dtype="object")
    else:
        behavior = behavior_probability.reindex(out.index)
    out["behavior_diagnostic_probability"] = behavior
    out["behavior_probability_delta"] = pd.to_numeric(behavior, errors="coerce") - primary
    out["model_policy"] = model_policy_label(policy)
    out["edge_source"] = "diagnostic_behavior_view" if show_behavior_diagnostic else "primary_model"
    return out


def annotate_decimal_alpha_with_policy(
    primary_alpha_df: pd.DataFrame | None,
    behavior_alpha_df: pd.DataFrame | None = None,
    show_behavior_diagnostic: bool = False,
    policy: dict[str, Any] | None = None,
) -> pd.DataFrame:
    out = primary_alpha_df.copy() if primary_alpha_df is not None else pd.DataFrame()
    behavior_probability = None
    if (
        behavior_alpha_df is not None
        and not behavior_alpha_df.empty
        and not out.empty
        and {"market", "selection", "model_prob"}.issubset(behavior_alpha_df.columns)
        and {"market", "selection"}.issubset(out.columns)
    ):
        behavior_lookup = behavior_alpha_df[["market", "selection", "model_prob"]].rename(
            columns={"model_prob": "_behavior_model_prob"}
        )
        merged = out[["market", "selection"]].merge(behavior_lookup, on=["market", "selection"], how="left")
        behavior_probability = merged["_behavior_model_prob"]
        behavior_probability.index = out.index
    return add_policy_columns_to_alpha(
        out,
        "model_prob",
        behavior_probability=behavior_probability,
        show_behavior_diagnostic=show_behavior_diagnostic,
        policy=policy,
    )


def annotate_polymarket_alpha_with_policy(
    primary_alpha_df: pd.DataFrame | None,
    behavior_result: dict[str, Any] | None = None,
    show_behavior_diagnostic: bool = False,
    policy: dict[str, Any] | None = None,
) -> pd.DataFrame:
    out = primary_alpha_df.copy() if primary_alpha_df is not None else pd.DataFrame()
    behavior_probability = None
    if behavior_result and not out.empty:
        from src.alpha import model_probability_for_side

        behavior_probs = behavior_result.get("probs", {}) or {}
        values = []
        for _, row in out.iterrows():
            model_side = str(row.get("model_side", ""))
            market_type = str(row.get("market_type", ""))
            prob = model_probability_for_side(behavior_probs, model_side, market_type)
            if pd.notna(prob) and str(row.get("polymarket_side", "YES")).upper() == "NO":
                prob = 1.0 - float(prob)
            values.append(prob)
        behavior_probability = pd.Series(values, index=out.index, dtype="object")
    return add_policy_columns_to_alpha(
        out,
        "model_probability",
        behavior_probability=behavior_probability,
        show_behavior_diagnostic=show_behavior_diagnostic,
        policy=policy,
    )
