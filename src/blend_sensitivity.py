from __future__ import annotations

from typing import Any

import pandas as pd

from src.asof_backtest import (
    _as_naive_timestamp,
    _canonical_teams,
    _date_label,
    _elo_asof_from_rolling_rows,
    _has_elo_inputs,
    _normalise_completed_matches,
    _rating_frame,
    _truthy,
    build_team_behavior_asof,
    get_team_ratings_asof,
)
from src.backtest import calculate_backtest_metrics, run_model_for_backtest_match
from src.config import DATA_DIR
from src.elo import add_elo_to_historical_matches, calculate_rolling_elo
from src.historical_data import HISTORICAL_MATCH_COLUMNS, load_historical_matches
from src.model_policy import get_current_model_policy, policy_export_fields
from src.ratings import TEAM_RATING_COLUMNS
from src.utils import coerce_float, read_csv_with_columns
from src.weather import load_venues


DEFAULT_BLEND_MULTIPLIERS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00]

BLEND_SENSITIVITY_PREDICTION_COLUMNS = [
    "match_id",
    "date_utc",
    "home",
    "away",
    "home_goals",
    "away_goals",
    "actual_result",
    "blend_multiplier",
    "home_win_prob",
    "draw_prob",
    "away_win_prob",
    "actual_result_prob",
    "home_xg",
    "away_xg",
    "over_2_5_prob",
    "under_2_5_prob",
    "btts_yes_prob",
    "btts_no_prob",
    "most_likely_result",
    "log_loss_1x2",
    "lookahead_safe",
    "warning",
    "model_policy",
    "primary_model_mode",
    "behavior_status",
    "behavior_blend_used",
    "strict_validation_summary",
]

BLEND_SENSITIVITY_METRIC_COLUMNS = [
    "blend_multiplier",
    "n_matches",
    "brier_score_1x2",
    "log_loss_1x2",
    "most_likely_result_accuracy",
    "mean_probability_assigned_to_actual_result",
    "over_2_5_brier",
    "btts_brier",
    "total_goals_mae",
    "delta_brier_vs_baseline",
    "delta_log_loss_vs_baseline",
    "delta_actual_prob_vs_baseline",
    "rank_by_brier",
    "rank_by_log_loss",
]

TEAM_SENSITIVITY_COLUMNS = [
    "team",
    "matches",
    "best_blend_by_actual_prob",
    "baseline_actual_prob",
    "best_actual_prob",
    "actual_prob_delta",
    "behavior_helped_count",
    "behavior_hurt_count",
    "warning",
]


def run_asof_blend_sensitivity(
    completed_matches_df: pd.DataFrame | None,
    team_ratings_df: pd.DataFrame | None,
    historical_matches_df: pd.DataFrame | None,
    blend_multipliers: list[float] | tuple[float, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    multipliers = normalise_blend_multipliers(blend_multipliers)
    matches = _normalise_completed_matches(completed_matches_df)
    if matches.empty:
        return (
            pd.DataFrame(columns=BLEND_SENSITIVITY_PREDICTION_COLUMNS),
            pd.DataFrame(columns=BLEND_SENSITIVITY_METRIC_COLUMNS),
        )

    ratings = _rating_frame(team_ratings_df)
    historical = historical_matches_df.copy() if historical_matches_df is not None else pd.DataFrame(columns=HISTORICAL_MATCH_COLUMNS)
    historical_with_elo = historical if _has_elo_inputs(historical) else add_elo_to_historical_matches(historical) if not historical.empty else historical
    rolling_elo = calculate_rolling_elo(historical) if not historical.empty else pd.DataFrame()
    venues = load_venues()

    behavior_cache: dict[tuple[str, tuple[str, ...]], pd.DataFrame] = {}
    elo_cache: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []

    for _, match in matches.iterrows():
        asof_label = _date_label(match.get("date_utc"))
        raw_teams = [str(match.get("home", "")), str(match.get("away", ""))]
        canonical_teams = tuple(sorted(_canonical_teams(raw_teams) or raw_teams))
        behavior_key = (asof_label, canonical_teams)
        if behavior_key not in behavior_cache:
            behavior_cache[behavior_key] = build_team_behavior_asof(historical_with_elo, ratings, asof_label, teams=list(canonical_teams))
        if asof_label not in elo_cache:
            elo_cache[asof_label] = _elo_asof_from_rolling_rows(rolling_elo, asof_label)

        for multiplier in multipliers:
            teams_asof = get_team_ratings_asof(
                ratings,
                historical_with_elo,
                asof_label,
                teams=list(raw_teams),
                mode="behavior_adjusted_asof",
                elo_asof_df=elo_cache[asof_label],
                behavior_asof_df=behavior_cache[behavior_key],
                behavior_blend_multiplier=multiplier,
            )
            mode_label = f"blend_{multiplier:.2f}"
            prediction = run_model_for_backtest_match(match, mode_label, teams_asof, venues, pd.DataFrame())
            rows.append(_sensitivity_prediction_row(prediction, match, teams_asof, multiplier))

    predictions = pd.DataFrame(rows, columns=BLEND_SENSITIVITY_PREDICTION_COLUMNS)
    metrics = calculate_blend_sensitivity_metrics(predictions)
    return predictions, metrics


def calculate_blend_sensitivity_metrics(predictions_df: pd.DataFrame | None) -> pd.DataFrame:
    predictions = predictions_df.copy() if predictions_df is not None else pd.DataFrame(columns=BLEND_SENSITIVITY_PREDICTION_COLUMNS)
    if predictions.empty:
        return pd.DataFrame(columns=BLEND_SENSITIVITY_METRIC_COLUMNS)

    scored = predictions.copy()
    scored["model_mode"] = scored["blend_multiplier"].map(lambda value: f"blend_{float(value):.2f}")
    base_metrics = calculate_backtest_metrics(scored)
    if base_metrics.empty:
        return pd.DataFrame(columns=BLEND_SENSITIVITY_METRIC_COLUMNS)
    base_metrics["blend_multiplier"] = base_metrics["model_mode"].str.replace("blend_", "", regex=False).astype(float)

    baseline_multiplier = 0.0 if (base_metrics["blend_multiplier"] == 0.0).any() else float(base_metrics["blend_multiplier"].min())
    baseline = base_metrics.loc[base_metrics["blend_multiplier"] == baseline_multiplier].iloc[0]
    rows = []
    for _, row in base_metrics.iterrows():
        rows.append(
            {
                "blend_multiplier": float(row["blend_multiplier"]),
                "n_matches": int(row["n_matches"]),
                "brier_score_1x2": float(row["brier_score_1x2"]),
                "log_loss_1x2": float(row["log_loss_1x2"]),
                "most_likely_result_accuracy": float(row["most_likely_result_accuracy"]),
                "mean_probability_assigned_to_actual_result": float(row["mean_probability_assigned_to_actual_result"]),
                "over_2_5_brier": float(row["over_2_5_brier"]),
                "btts_brier": float(row["btts_brier"]),
                "total_goals_mae": float(row["total_goals_mae"]),
                "delta_brier_vs_baseline": float(row["brier_score_1x2"] - baseline["brier_score_1x2"]),
                "delta_log_loss_vs_baseline": float(row["log_loss_1x2"] - baseline["log_loss_1x2"]),
                "delta_actual_prob_vs_baseline": float(
                    row["mean_probability_assigned_to_actual_result"] - baseline["mean_probability_assigned_to_actual_result"]
                ),
            }
        )
    metrics = pd.DataFrame(rows)
    metrics["rank_by_brier"] = metrics["brier_score_1x2"].rank(method="min", ascending=True).astype(int)
    metrics["rank_by_log_loss"] = metrics["log_loss_1x2"].rank(method="min", ascending=True).astype(int)
    return metrics.sort_values("blend_multiplier").reset_index(drop=True)[BLEND_SENSITIVITY_METRIC_COLUMNS]


def build_team_sensitivity_table(predictions_df: pd.DataFrame | None) -> pd.DataFrame:
    predictions = predictions_df.copy() if predictions_df is not None else pd.DataFrame(columns=BLEND_SENSITIVITY_PREDICTION_COLUMNS)
    if predictions.empty:
        return pd.DataFrame(columns=TEAM_SENSITIVITY_COLUMNS)

    baseline_multiplier = 0.0 if (predictions["blend_multiplier"].astype(float) == 0.0).any() else float(predictions["blend_multiplier"].min())
    default_multiplier = 1.0 if (predictions["blend_multiplier"].astype(float) == 1.0).any() else float(predictions["blend_multiplier"].max())
    teams = sorted(set(predictions["home"].dropna().astype(str)) | set(predictions["away"].dropna().astype(str)))
    rows = []
    for team in teams:
        team_rows = predictions.loc[(predictions["home"].astype(str) == team) | (predictions["away"].astype(str) == team)].copy()
        if team_rows.empty:
            continue
        by_blend = team_rows.groupby("blend_multiplier", dropna=False)["actual_result_prob"].mean().sort_index()
        baseline_actual = float(by_blend.get(baseline_multiplier, by_blend.iloc[0]))
        best_blend = float(by_blend.idxmax())
        best_actual = float(by_blend.max())
        baseline_rows = team_rows.loc[team_rows["blend_multiplier"].astype(float) == baseline_multiplier][["match_id", "actual_result_prob"]]
        default_rows = team_rows.loc[team_rows["blend_multiplier"].astype(float) == default_multiplier][["match_id", "actual_result_prob"]]
        merged = baseline_rows.merge(default_rows, on="match_id", suffixes=("_baseline", "_default"), how="inner")
        deltas = merged["actual_result_prob_default"] - merged["actual_result_prob_baseline"] if not merged.empty else pd.Series(dtype=float)
        helped = int((deltas > 0.0005).sum())
        hurt = int((deltas < -0.0005).sum())
        warning = _team_warning(best_blend, best_actual - baseline_actual, helped, hurt)
        rows.append(
            {
                "team": team,
                "matches": int(team_rows["match_id"].nunique()),
                "best_blend_by_actual_prob": best_blend,
                "baseline_actual_prob": baseline_actual,
                "best_actual_prob": best_actual,
                "actual_prob_delta": best_actual - baseline_actual,
                "behavior_helped_count": helped,
                "behavior_hurt_count": hurt,
                "warning": warning,
            }
        )
    return pd.DataFrame(rows, columns=TEAM_SENSITIVITY_COLUMNS).sort_values(["actual_prob_delta", "team"], ascending=[False, True]).reset_index(drop=True)


def blend_sensitivity_recommendation(metrics_df: pd.DataFrame | None) -> str:
    metrics = metrics_df.copy() if metrics_df is not None else pd.DataFrame(columns=BLEND_SENSITIVITY_METRIC_COLUMNS)
    if metrics.empty:
        return "No blend sensitivity results are available."
    improves_both = metrics.loc[(metrics["delta_brier_vs_baseline"] < 0) & (metrics["delta_log_loss_vs_baseline"] < 0)].copy()
    if improves_both.empty:
        return "Recommend behavior as diagnostic only; no tested blend improved both Brier and log loss versus baseline."
    weak = improves_both.loc[(improves_both["blend_multiplier"] >= 0.05) & (improves_both["blend_multiplier"] <= 0.15)]
    if not weak.empty:
        best = weak.sort_values(["brier_score_1x2", "log_loss_1x2"]).iloc[0]
        return f"Recommend reduced behavior blend around {best['blend_multiplier']:.2f}; weak behavior improved strict metrics."
    current = improves_both.loc[improves_both["blend_multiplier"] == 1.0]
    if not current.empty:
        return "Keep current default behavior blend; multiplier 1.00 improved both strict Brier and log loss."
    best = improves_both.sort_values(["brier_score_1x2", "log_loss_1x2"]).iloc[0]
    return f"Recommend further investigation; blend {best['blend_multiplier']:.2f} improved metrics but is outside the conservative weak-blend range."


def run_blend_sensitivity_result(
    start_date: str | None = None,
    end_date: str | None = None,
    teams: list[str] | tuple[str, ...] | None = None,
    competition: str | None = None,
    blend_multipliers: list[float] | tuple[float, ...] | None = None,
) -> dict[str, Any]:
    from src.backtest import load_completed_matches_for_backtest

    matches = load_completed_matches_for_backtest(start_date=start_date, end_date=end_date, teams=teams)
    if competition and not matches.empty and "competition" in matches.columns:
        needle = str(competition).strip().lower()
        matches = matches.loc[matches["competition"].astype(str).str.lower().str.contains(needle, na=False)].copy()
    ratings = read_csv_with_columns(DATA_DIR / "team_ratings.csv", TEAM_RATING_COLUMNS)
    historical = load_historical_matches(use_cache=False)
    predictions, metrics = run_asof_blend_sensitivity(matches, ratings, historical, blend_multipliers)
    team = build_team_sensitivity_table(predictions)
    return {
        "matches": matches.reset_index(drop=True),
        "predictions": predictions,
        "metrics": metrics,
        "team_sensitivity": team,
        "recommendation": blend_sensitivity_recommendation(metrics),
    }


def normalise_blend_multipliers(blend_multipliers: list[float] | tuple[float, ...] | None = None) -> list[float]:
    values = DEFAULT_BLEND_MULTIPLIERS if blend_multipliers is None else list(blend_multipliers)
    cleaned = sorted({round(max(0.0, coerce_float(value, 0.0)), 3) for value in values})
    if 0.0 not in cleaned:
        cleaned.insert(0, 0.0)
    return cleaned


def _sensitivity_prediction_row(
    prediction: dict[str, Any],
    match: pd.Series,
    teams_asof: pd.DataFrame,
    blend_multiplier: float,
) -> dict[str, Any]:
    home = str(match.get("home", ""))
    away = str(match.get("away", ""))
    home_row = _rating_row(teams_asof, home)
    away_row = _rating_row(teams_asof, away)
    lookahead_safe = bool(_truthy(home_row.get("lookahead_safe", True)) and _truthy(away_row.get("lookahead_safe", True)))
    warnings = []
    if not lookahead_safe:
        warnings.append("lookahead_violation")
    for row in [home_row, away_row]:
        warning = str(row.get("rating_warning", "") or "")
        if warning:
            warnings.append(warning)
    policy_fields = policy_export_fields(
        get_current_model_policy(),
        behavior_blend_used=float(blend_multiplier) > 0.0,
    )
    return {
        "match_id": prediction.get("match_id", ""),
        "date_utc": prediction.get("date_utc", match.get("date_utc", "")),
        "home": home,
        "away": away,
        "home_goals": prediction.get("home_goals", match.get("home_goals", pd.NA)),
        "away_goals": prediction.get("away_goals", match.get("away_goals", pd.NA)),
        "actual_result": prediction.get("actual_result", ""),
        "blend_multiplier": float(blend_multiplier),
        "home_win_prob": prediction.get("home_win_prob", pd.NA),
        "draw_prob": prediction.get("draw_prob", pd.NA),
        "away_win_prob": prediction.get("away_win_prob", pd.NA),
        "actual_result_prob": prediction.get("actual_result_prob", pd.NA),
        "home_xg": prediction.get("home_xg", pd.NA),
        "away_xg": prediction.get("away_xg", pd.NA),
        "over_2_5_prob": prediction.get("over_2_5_prob", pd.NA),
        "under_2_5_prob": prediction.get("under_2_5_prob", pd.NA),
        "btts_yes_prob": prediction.get("btts_yes_prob", pd.NA),
        "btts_no_prob": prediction.get("btts_no_prob", pd.NA),
        "most_likely_result": prediction.get("most_likely_result", ""),
        "log_loss_1x2": prediction.get("log_loss_1x2", pd.NA),
        "lookahead_safe": lookahead_safe,
        "warning": "; ".join(dict.fromkeys([warning for warning in warnings if warning])),
        **policy_fields,
    }


def _rating_row(ratings: pd.DataFrame, team: str) -> pd.Series:
    from src.ratings import rating_row_for_team

    row = rating_row_for_team(ratings, team)
    return row if not row.empty else pd.Series(dtype="object")


def _team_warning(best_blend: float, delta: float, helped: int, hurt: int) -> str:
    if best_blend == 0.0:
        return "baseline best for this team"
    if delta > 0.002 and helped > hurt:
        return "behavior helped this team in sample"
    if hurt > helped:
        return "default behavior hurt this team in sample"
    return ""
