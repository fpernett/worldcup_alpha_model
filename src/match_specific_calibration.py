from __future__ import annotations

from typing import Any

import pandas as pd

from src.historical_data import HISTORICAL_MATCH_COLUMNS
from src.model_training import _add_relevance_weights, generate_candidate_parameter_sets
from src.tournament_learning import (
    TOURNAMENT_LEARNING_LEDGER_COLUMNS,
    filter_tournament_learning_asof,
    load_tournament_learning_ledger,
)
from src.utils import coerce_float


MATCH_SPECIFIC_CALIBRATION_COLUMNS = [
    "match_id",
    "date_utc",
    "competition",
    "competition_type",
    "team",
    "opponent",
    "team_goals",
    "opponent_goals",
    "source",
    "relevance_weight",
    "eligible_after_utc",
    "lookahead_safe",
]


def build_match_specific_calibration_set(
    historical_matches_df: pd.DataFrame | None,
    learning_df: pd.DataFrame | None,
    home: str,
    away: str,
    target_kickoff_utc: Any,
    target_match_id: str | None = None,
    competition: str = "World Cup",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build pre-match calibration evidence without using future results."""
    historical = _historical_rows_asof(historical_matches_df, target_kickoff_utc, target_match_id)
    learning_rows, learning_diag = filter_tournament_learning_asof(learning_df, target_kickoff_utc, target_match_id=target_match_id)
    learning_long = _learning_to_long_rows(learning_rows)
    combined = pd.concat([historical, learning_long], ignore_index=True)
    if combined.empty:
        diagnostics = {
            **learning_diag,
            "calibration_rows": 0,
            "calibration_matches": 0,
            "home_rows": 0,
            "away_rows": 0,
            "lookahead_safe": bool(learning_diag.get("lookahead_safe", True)),
        }
        return pd.DataFrame(columns=MATCH_SPECIFIC_CALIBRATION_COLUMNS), diagnostics

    combined = _add_relevance_weights(combined, target_kickoff_utc, competition)
    combined["lookahead_safe"] = _row_lookahead_safe(combined, target_kickoff_utc)
    teams = {str(home).strip().lower(), str(away).strip().lower()}
    diagnostics = {
        **learning_diag,
        "calibration_rows": int(len(combined)),
        "calibration_matches": int(combined["match_id"].astype(str).nunique()),
        "home_rows": int((combined["team"].astype(str).str.lower() == str(home).strip().lower()).sum()),
        "away_rows": int((combined["team"].astype(str).str.lower() == str(away).strip().lower()).sum()),
        "teams_in_scope_rows": int(combined["team"].astype(str).str.lower().isin(teams).sum()),
        "lookahead_safe": bool(combined["lookahead_safe"].all() and learning_diag.get("lookahead_safe", True)),
    }
    return combined[MATCH_SPECIFIC_CALIBRATION_COLUMNS].reset_index(drop=True), diagnostics


def recommend_match_specific_parameter_set(
    calibration_set_df: pd.DataFrame | None,
    candidate_params_df: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Recommend a parameter set for diagnostics only.

    V1 keeps the recommendation deliberately conservative: if the calibration
    sample is thin, use `baseline_current`; otherwise prefer the first
    candidate table row with the strongest evidence tag already available.
    This does not change global model policy.
    """
    candidates = candidate_params_df.copy() if candidate_params_df is not None else generate_candidate_parameter_sets()
    if candidates.empty:
        return pd.Series(dtype="object"), candidates
    calibration = calibration_set_df.copy() if calibration_set_df is not None else pd.DataFrame()
    if calibration.empty or pd.to_numeric(calibration.get("relevance_weight", pd.Series(dtype=float)), errors="coerce").sum() < 10:
        selected = candidates.loc[candidates["parameter_set_id"].astype(str) == "baseline_current"]
        return (selected.iloc[0] if not selected.empty else candidates.iloc[0]), candidates
    selected = candidates.iloc[0]
    return selected, candidates


def load_current_match_specific_learning() -> pd.DataFrame:
    return load_tournament_learning_ledger()


def _historical_rows_asof(
    historical_matches_df: pd.DataFrame | None,
    target_kickoff_utc: Any,
    target_match_id: str | None,
) -> pd.DataFrame:
    if historical_matches_df is None or historical_matches_df.empty:
        return pd.DataFrame(columns=MATCH_SPECIFIC_CALIBRATION_COLUMNS)
    rows = historical_matches_df.copy()
    for col in HISTORICAL_MATCH_COLUMNS:
        if col not in rows.columns:
            rows[col] = pd.NA
    dates = pd.to_datetime(rows["date_utc"], errors="coerce", utc=True)
    target = pd.to_datetime(target_kickoff_utc, errors="coerce", utc=True)
    rows = rows.loc[dates.notna()].copy()
    if not pd.isna(target):
        rows = rows.loc[dates.loc[rows.index] < target].copy()
    if target_match_id:
        rows = rows.loc[rows["match_id"].astype(str) != str(target_match_id)].copy()
    rows["source"] = rows.get("source", "historical_matches").fillna("historical_matches")
    rows["eligible_after_utc"] = rows["date_utc"]
    rows["lookahead_safe"] = True
    out = pd.DataFrame(
        {
            "match_id": rows["match_id"],
            "date_utc": rows["date_utc"],
            "competition": rows["competition"],
            "competition_type": rows["competition_type"],
            "team": rows["team"],
            "opponent": rows["opponent"],
            "team_goals": rows["team_goals"],
            "opponent_goals": rows["opponent_goals"],
            "source": rows["source"],
            "eligible_after_utc": rows["eligible_after_utc"],
            "lookahead_safe": rows["lookahead_safe"],
        }
    )
    return out


def _learning_to_long_rows(learning_df: pd.DataFrame | None) -> pd.DataFrame:
    learning = learning_df.copy() if learning_df is not None else pd.DataFrame(columns=TOURNAMENT_LEARNING_LEDGER_COLUMNS)
    if learning.empty:
        return pd.DataFrame(columns=MATCH_SPECIFIC_CALIBRATION_COLUMNS)
    rows: list[dict[str, Any]] = []
    for _, row in learning.iterrows():
        home = row.get("home", "")
        away = row.get("away", "")
        home_goals = int(coerce_float(row.get("home_goals"), 0.0))
        away_goals = int(coerce_float(row.get("away_goals"), 0.0))
        base = {
            "match_id": row.get("match_id", ""),
            "date_utc": row.get("date_utc", ""),
            "competition": row.get("competition", ""),
            "competition_type": "world_cup",
            "source": "tournament_learning_ledger",
            "eligible_after_utc": row.get("eligible_after_utc", ""),
            "lookahead_safe": True,
        }
        rows.append({**base, "team": home, "opponent": away, "team_goals": home_goals, "opponent_goals": away_goals})
        rows.append({**base, "team": away, "opponent": home, "team_goals": away_goals, "opponent_goals": home_goals})
    return pd.DataFrame(rows, columns=[col for col in MATCH_SPECIFIC_CALIBRATION_COLUMNS if col != "relevance_weight"])


def _row_lookahead_safe(rows: pd.DataFrame, target_kickoff_utc: Any) -> pd.Series:
    target = pd.to_datetime(target_kickoff_utc, errors="coerce", utc=True)
    if pd.isna(target):
        return pd.Series([False] * len(rows), index=rows.index)
    eligible = pd.to_datetime(rows["eligible_after_utc"], errors="coerce", utc=True)
    dates = pd.to_datetime(rows["date_utc"], errors="coerce", utc=True)
    effective = eligible.fillna(dates)
    return effective < target
