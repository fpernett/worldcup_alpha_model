from __future__ import annotations

from typing import Any

import pandas as pd

from src.backtest import (
    BACKTEST_METRIC_COLUMNS,
    COMPLETED_MATCH_COLUMNS,
    calculate_backtest_metrics,
    classify_actual_result,
    load_completed_matches_for_backtest,
    run_model_for_backtest_match,
)
from src.config import DATA_DIR
from src.elo import ELO_ASOF_COLUMNS, add_elo_to_historical_matches, build_elo_asof, calculate_rolling_elo
from src.historical_data import HISTORICAL_MATCH_COLUMNS, load_historical_matches
from src.model import ModelConfig
from src.model_policy import get_current_model_policy, policy_export_fields
from src.ratings import TEAM_RATING_COLUMNS, _apply_behavior_blend, _normalise_ratings, rating_row_for_team
from src.team_behavior import TEAM_BEHAVIOR_COLUMNS, build_team_behavior_table
from src.team_names import normalize_team_name
from src.utils import coerce_float, read_csv_with_columns
from src.weather import load_venues


ASOF_BACKTEST_WARNING = (
    "Strict as-of-date backtest: each prediction only uses historical matches before that match date. "
    "This avoids current-behavior look-ahead bias, but teams with little prior history may receive no behavior blend."
)

ASOF_PREDICTION_COLUMNS = [
    "match_id",
    "date_utc",
    "home",
    "away",
    "home_goals",
    "away_goals",
    "actual_result",
    "mode",
    "model_mode",
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
    "home_latest_behavior_match",
    "away_latest_behavior_match",
    "home_matches_used_recent",
    "away_matches_used_recent",
    "lookahead_safe",
    "warning",
    "model_policy",
    "primary_model_mode",
    "behavior_status",
    "behavior_blend_used",
    "strict_validation_summary",
]

ASOF_METRIC_COLUMNS = [
    *BACKTEST_METRIC_COLUMNS,
    "lookahead_safe_rate",
    "mean_home_behavior_matches_used",
    "mean_away_behavior_matches_used",
    "matches_with_insufficient_asof_behavior",
]

ASOF_COMPARISON_COLUMNS = [
    "match_id",
    "date_utc",
    "home",
    "away",
    "score",
    "actual_result",
    "baseline_actual_prob",
    "behavior_asof_actual_prob",
    "actual_prob_delta",
    "baseline_log_loss",
    "behavior_asof_log_loss",
    "improved_by_behavior_asof",
    "notes",
]


def filter_historical_matches_asof(
    historical_matches_df: pd.DataFrame | None,
    as_of_date: Any,
    exclude_match_id: str | None = None,
) -> pd.DataFrame:
    """Keep only long-format historical rows strictly before as_of_date."""
    if historical_matches_df is None or historical_matches_df.empty:
        return pd.DataFrame(columns=HISTORICAL_MATCH_COLUMNS)
    out = historical_matches_df.copy()
    for col in HISTORICAL_MATCH_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out["_date"] = pd.to_datetime(out["date_utc"], errors="coerce")
    asof_ts = _as_naive_timestamp(as_of_date)
    out = out.loc[out["_date"].notna() & (out["_date"] < asof_ts)].copy()
    if exclude_match_id is not None and "match_id" in out.columns:
        out = out.loc[out["match_id"].astype(str) != str(exclude_match_id)].copy()
    out = out.drop(columns=["_date"])
    return out[HISTORICAL_MATCH_COLUMNS].reset_index(drop=True)


def assert_no_future_matches(asof_matches_df: pd.DataFrame, as_of_date: Any) -> None:
    if asof_matches_df is None or asof_matches_df.empty or "date_utc" not in asof_matches_df.columns:
        return
    asof_ts = _as_naive_timestamp(as_of_date)
    dates = pd.to_datetime(asof_matches_df["date_utc"], errors="coerce")
    violations = asof_matches_df.loc[dates >= asof_ts].copy()
    if not violations.empty:
        sample = violations[["match_id", "date_utc", "team", "opponent"]].head(5).to_dict("records")
        raise ValueError(f"As-of historical slice contains matches on/after {asof_ts.date().isoformat()}: {sample}")


def build_team_behavior_asof(
    historical_matches_df: pd.DataFrame | None,
    team_ratings_df: pd.DataFrame | None,
    as_of_date: Any,
    teams: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    selected_teams = _canonical_teams(teams)
    asof_matches = filter_historical_matches_asof(historical_matches_df, as_of_date)
    assert_no_future_matches(asof_matches, as_of_date)
    elo_enhanced = asof_matches if _has_elo_inputs(asof_matches) else add_elo_to_historical_matches(asof_matches)
    behavior = build_team_behavior_table(elo_enhanced, team_ratings_df, as_of_date, teams=selected_teams)
    if behavior.empty:
        behavior = pd.DataFrame(columns=TEAM_BEHAVIOR_COLUMNS)
    asof_label = _date_label(as_of_date)
    behavior["as_of_date"] = asof_label
    if "latest_match_used" not in behavior.columns:
        behavior["latest_match_used"] = ""
    behavior["lookahead_safe"] = behavior["latest_match_used"].map(lambda value: _latest_before_asof(value, asof_label))
    return behavior.reset_index(drop=True)


def get_team_ratings_asof(
    team_ratings_df: pd.DataFrame | None,
    historical_matches_df: pd.DataFrame | None,
    as_of_date: Any,
    teams: list[str] | tuple[str, ...] | None = None,
    mode: str = "behavior_adjusted_asof",
    elo_asof_df: pd.DataFrame | None = None,
    behavior_asof_df: pd.DataFrame | None = None,
    behavior_blend_multiplier: float = 1.0,
) -> pd.DataFrame:
    base = _normalise_ratings(_rating_frame(team_ratings_df), "manual_csv")
    base = _apply_elo_asof(base, historical_matches_df, as_of_date, elo_asof_df=elo_asof_df)
    asof_label = _date_label(as_of_date)

    if mode == "baseline_manual":
        out = base.copy()
        _ensure_asof_rating_columns(out)
        out["attack_source"] = "manual_or_external_benchmark_asof"
        out["defense_source"] = "manual_or_external_benchmark_asof"
        out["recent_form_source"] = "manual_or_external_benchmark_asof"
        out["behavior_blend_used"] = False
        out["latest_behavior_match_used"] = ""
        out["lookahead_safe"] = True
        out["rating_warning"] = ""
        out["as_of_date"] = asof_label
        return out.reset_index(drop=True)

    if mode != "behavior_adjusted_asof":
        raise ValueError(f"Unknown as-of rating mode: {mode}")

    behavior = behavior_asof_df.copy() if behavior_asof_df is not None else build_team_behavior_asof(historical_matches_df, base, as_of_date, teams=_canonical_teams(teams))
    out = _apply_behavior_blend(
        base.copy(),
        manual_base=base.copy(),
        behavior_df=behavior,
        behavior_blend_multiplier=behavior_blend_multiplier,
    )
    _ensure_asof_rating_columns(out)
    out["as_of_date"] = asof_label
    behavior_lookup = _behavior_lookup(behavior)
    for idx, row in out.iterrows():
        team = str(row.get("team", ""))
        behavior_row = behavior_lookup.get(team.lower(), pd.Series(dtype="object"))
        latest = str(behavior_row.get("latest_match_used", "") or "")
        matches_used = coerce_float(behavior_row.get("matches_used_recent"), 0.0)
        lookahead_safe = bool(behavior_row.get("lookahead_safe", True))
        out.at[idx, "latest_behavior_match_used"] = latest
        out.at[idx, "asof_behavior_matches_used_recent"] = int(matches_used)
        out.at[idx, "lookahead_safe"] = lookahead_safe
        warnings = []
        if matches_used < 8:
            warnings.append("insufficient_asof_behavior")
        if not lookahead_safe:
            warnings.append("lookahead_violation")
        existing_warning = str(row.get("rating_warning", "") or "")
        if existing_warning:
            warnings.append(existing_warning)
        out.at[idx, "rating_warning"] = "; ".join(dict.fromkeys(warnings))
    return out.reset_index(drop=True)


def run_asof_backtest(
    completed_matches_df: pd.DataFrame | None,
    team_ratings_df: pd.DataFrame | None,
    historical_matches_df: pd.DataFrame | None,
    mode: str = "both",
    behavior_blend_multiplier: float = 1.0,
    cfg: ModelConfig | None = None,
) -> pd.DataFrame:
    matches = _normalise_completed_matches(completed_matches_df)
    if matches.empty:
        return pd.DataFrame(columns=ASOF_PREDICTION_COLUMNS)

    historical = historical_matches_df.copy() if historical_matches_df is not None else pd.DataFrame(columns=HISTORICAL_MATCH_COLUMNS)
    historical_with_elo = historical if _has_elo_inputs(historical) else add_elo_to_historical_matches(historical) if not historical.empty else historical
    rolling_elo = calculate_rolling_elo(historical) if not historical.empty else pd.DataFrame()
    ratings = _rating_frame(team_ratings_df)
    venues = load_venues()
    selected_modes = _selected_modes(mode)
    rows: list[dict[str, Any]] = []
    ratings_cache: dict[tuple[str, str, tuple[str, ...]], pd.DataFrame] = {}

    for _, match in matches.iterrows():
        asof_label = _date_label(match.get("date_utc"))
        match_teams = tuple(sorted([str(match.get("home", "")), str(match.get("away", ""))]))
        for selected_mode in selected_modes:
            cache_key = (selected_mode, asof_label, match_teams, round(float(behavior_blend_multiplier), 6))
            if cache_key not in ratings_cache:
                elo_asof = _elo_asof_from_rolling_rows(rolling_elo, asof_label)
                ratings_cache[cache_key] = get_team_ratings_asof(
                    ratings,
                    historical_with_elo,
                    asof_label,
                    teams=list(match_teams),
                    mode=selected_mode,
                    elo_asof_df=elo_asof,
                    behavior_blend_multiplier=behavior_blend_multiplier,
                )
            teams_asof = ratings_cache[cache_key]
            prediction = run_model_for_backtest_match(match, selected_mode, teams_asof, venues, pd.DataFrame(), cfg=cfg)
            rows.append(_asof_prediction_row(prediction, match, teams_asof, selected_mode))

    return pd.DataFrame(rows, columns=ASOF_PREDICTION_COLUMNS)


def calculate_asof_backtest_metrics(predictions_df: pd.DataFrame | None) -> pd.DataFrame:
    if predictions_df is None or predictions_df.empty:
        return pd.DataFrame(columns=ASOF_METRIC_COLUMNS)
    scored = predictions_df.copy()
    if "model_mode" not in scored.columns and "mode" in scored.columns:
        scored["model_mode"] = scored["mode"]
    base_metrics = calculate_backtest_metrics(scored)
    rows = []
    for _, metric in base_metrics.iterrows():
        mode = str(metric.get("model_mode", ""))
        group = scored.loc[scored["model_mode"].astype(str) == mode].copy()
        lookahead_safe = group["lookahead_safe"].map(_truthy) if "lookahead_safe" in group.columns else pd.Series(dtype=bool)
        warnings = group["warning"].astype(str) if "warning" in group.columns else pd.Series("", index=group.index)
        row = metric.to_dict()
        row.update(
            {
                "lookahead_safe_rate": float(lookahead_safe.mean()) if len(lookahead_safe) else 0.0,
                "mean_home_behavior_matches_used": float(_numeric_group_column(group, "home_matches_used_recent").mean()) if not group.empty else 0.0,
                "mean_away_behavior_matches_used": float(_numeric_group_column(group, "away_matches_used_recent").mean()) if not group.empty else 0.0,
                "matches_with_insufficient_asof_behavior": int(warnings.str.contains("insufficient_asof_behavior", na=False).sum()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=ASOF_METRIC_COLUMNS)


def compare_asof_backtest_predictions(predictions_df: pd.DataFrame | None) -> pd.DataFrame:
    if predictions_df is None or predictions_df.empty:
        return pd.DataFrame(columns=ASOF_COMPARISON_COLUMNS)
    baseline = predictions_df.loc[predictions_df["mode"] == "baseline_manual"].copy()
    behavior = predictions_df.loc[predictions_df["mode"] == "behavior_adjusted_asof"].copy()
    if baseline.empty or behavior.empty:
        return pd.DataFrame(columns=ASOF_COMPARISON_COLUMNS)
    merged = baseline.merge(behavior, on="match_id", suffixes=("_baseline", "_behavior"), how="inner")
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        baseline_prob = coerce_float(row.get("actual_result_prob_baseline"), 0.0)
        behavior_prob = coerce_float(row.get("actual_result_prob_behavior"), 0.0)
        delta = behavior_prob - baseline_prob
        if delta > 0.0005:
            notes = "as-of behavior improved actual-result probability"
        elif delta < -0.0005:
            notes = "as-of behavior lowered actual-result probability"
        else:
            notes = "no material difference"
        rows.append(
            {
                "match_id": row.get("match_id", ""),
                "date_utc": row.get("date_utc_baseline", row.get("date_utc_behavior", "")),
                "home": row.get("home_baseline", row.get("home_behavior", "")),
                "away": row.get("away_baseline", row.get("away_behavior", "")),
                "score": f"{int(coerce_float(row.get('home_goals_baseline'), 0))}-{int(coerce_float(row.get('away_goals_baseline'), 0))}",
                "actual_result": row.get("actual_result_baseline", ""),
                "baseline_actual_prob": baseline_prob,
                "behavior_asof_actual_prob": behavior_prob,
                "actual_prob_delta": delta,
                "baseline_log_loss": row.get("log_loss_1x2_baseline", pd.NA),
                "behavior_asof_log_loss": row.get("log_loss_1x2_behavior", pd.NA),
                "improved_by_behavior_asof": bool(delta > 0),
                "notes": notes,
            }
        )
    return pd.DataFrame(rows, columns=ASOF_COMPARISON_COLUMNS).sort_values(["date_utc", "match_id"]).reset_index(drop=True)


def run_asof_backtest_result(
    start_date: str | None = None,
    end_date: str | None = None,
    teams: list[str] | tuple[str, ...] | None = None,
    competition: str | None = None,
) -> dict[str, Any]:
    matches = load_completed_matches_for_backtest(start_date=start_date, end_date=end_date, teams=teams)
    if competition and not matches.empty and "competition" in matches.columns:
        needle = str(competition).strip().lower()
        matches = matches.loc[matches["competition"].astype(str).str.lower().str.contains(needle, na=False)].copy()
    ratings = read_csv_with_columns(DATA_DIR / "team_ratings.csv", TEAM_RATING_COLUMNS)
    historical = load_historical_matches(use_cache=False)
    predictions = run_asof_backtest(matches, ratings, historical, mode="both")
    metrics = calculate_asof_backtest_metrics(predictions)
    comparison = compare_asof_backtest_predictions(predictions)
    return {
        "matches": matches.reset_index(drop=True),
        "predictions": predictions.reset_index(drop=True),
        "metrics": metrics,
        "comparison": comparison,
        "warning": ASOF_BACKTEST_WARNING,
    }


def _asof_prediction_row(
    prediction: dict[str, Any],
    match: pd.Series,
    teams_asof: pd.DataFrame,
    mode: str,
) -> dict[str, Any]:
    home = str(match.get("home", ""))
    away = str(match.get("away", ""))
    home_row = rating_row_for_team(teams_asof, home)
    away_row = rating_row_for_team(teams_asof, away)
    home_latest = str(home_row.get("latest_behavior_match_used", "") or "")
    away_latest = str(away_row.get("latest_behavior_match_used", "") or "")
    home_matches = int(coerce_float(home_row.get("asof_behavior_matches_used_recent"), 0.0))
    away_matches = int(coerce_float(away_row.get("asof_behavior_matches_used_recent"), 0.0))
    if mode == "baseline_manual":
        home_matches = 0
        away_matches = 0
        home_latest = ""
        away_latest = ""
    lookahead_safe = bool(_truthy(home_row.get("lookahead_safe", True)) and _truthy(away_row.get("lookahead_safe", True)))
    warnings = []
    if mode == "behavior_adjusted_asof" and (home_matches < 8 or away_matches < 8):
        warnings.append("insufficient_asof_behavior")
    if not lookahead_safe:
        warnings.append("lookahead_violation")
    for row in [home_row, away_row]:
        warning = str(row.get("rating_warning", "") or "")
        if warning:
            warnings.append(warning)

    policy_fields = policy_export_fields(
        get_current_model_policy(),
        behavior_blend_used=mode == "behavior_adjusted_asof",
    )
    return {
        "match_id": prediction.get("match_id", ""),
        "date_utc": prediction.get("date_utc", ""),
        "home": prediction.get("home", home),
        "away": prediction.get("away", away),
        "home_goals": prediction.get("home_goals", pd.NA),
        "away_goals": prediction.get("away_goals", pd.NA),
        "actual_result": prediction.get("actual_result", classify_actual_result(match.get("home_goals"), match.get("away_goals"))),
        "mode": mode,
        "model_mode": mode,
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
        "home_latest_behavior_match": home_latest,
        "away_latest_behavior_match": away_latest,
        "home_matches_used_recent": home_matches,
        "away_matches_used_recent": away_matches,
        "lookahead_safe": lookahead_safe,
        "warning": "; ".join(dict.fromkeys([warning for warning in warnings if warning])),
        **policy_fields,
    }


def _apply_elo_asof(
    ratings: pd.DataFrame,
    historical_matches_df: pd.DataFrame | None,
    as_of_date: Any,
    elo_asof_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = ratings.copy()
    if "elo" in out.columns:
        out["elo"] = pd.to_numeric(out["elo"], errors="coerce").astype(float)
    elo = elo_asof_df.copy() if elo_asof_df is not None else build_elo_asof(historical_matches_df, as_of_date)
    if elo.empty or "team" not in elo.columns:
        return out
    lookup = elo.set_index(elo["team"].astype(str).str.lower())
    for idx, row in out.iterrows():
        key = str(row.get("team", "")).lower()
        if key not in lookup.index:
            continue
        elo_row = lookup.loc[key]
        if isinstance(elo_row, pd.DataFrame):
            elo_row = elo_row.iloc[-1]
        out.at[idx, "elo"] = coerce_float(elo_row.get("elo_asof"), out.at[idx, "elo"])
    return out


def _elo_asof_from_rolling_rows(rolling_elo_df: pd.DataFrame | None, as_of_date: Any) -> pd.DataFrame:
    if rolling_elo_df is None or rolling_elo_df.empty:
        return pd.DataFrame(columns=ELO_ASOF_COLUMNS)
    out = rolling_elo_df.copy()
    if "date_utc" not in out.columns:
        return pd.DataFrame(columns=ELO_ASOF_COLUMNS)
    asof_ts = _as_naive_timestamp(as_of_date)
    out["date_utc"] = pd.to_datetime(out["date_utc"], errors="coerce")
    out = out.loc[out["date_utc"].notna() & (out["date_utc"] < asof_ts)].copy()
    if out.empty:
        return pd.DataFrame(columns=ELO_ASOF_COLUMNS)
    out = out.sort_values(["team", "date_utc", "match_id"])
    rows = []
    for team, group in out.groupby("team", dropna=False):
        latest = group.iloc[-1]
        rows.append(
            {
                "team": str(team),
                "elo_asof": coerce_float(latest.get("team_elo_post"), 1500.0),
                "matches_used_for_elo": int(len(group)),
                "latest_match_used_for_elo": latest["date_utc"].date().isoformat(),
                "as_of_date": asof_ts.date().isoformat(),
            }
        )
    return pd.DataFrame(rows, columns=ELO_ASOF_COLUMNS)


def _normalise_completed_matches(completed_matches_df: pd.DataFrame | None) -> pd.DataFrame:
    if completed_matches_df is None or completed_matches_df.empty:
        return pd.DataFrame(columns=COMPLETED_MATCH_COLUMNS)
    out = completed_matches_df.copy()
    for col in COMPLETED_MATCH_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out["home_goals"] = pd.to_numeric(out["home_goals"], errors="coerce")
    out["away_goals"] = pd.to_numeric(out["away_goals"], errors="coerce")
    out = out.dropna(subset=["date_utc", "home", "away", "home_goals", "away_goals"]).copy()
    return out[COMPLETED_MATCH_COLUMNS].reset_index(drop=True)


def _has_elo_inputs(matches: pd.DataFrame) -> bool:
    if matches is None or matches.empty:
        return True
    if "team_elo_pre" not in matches.columns or "opponent_elo" not in matches.columns:
        return False
    coverage = pd.to_numeric(matches["team_elo_pre"], errors="coerce").notna() & pd.to_numeric(matches["opponent_elo"], errors="coerce").notna()
    return bool(coverage.mean() >= 0.95)


def _selected_modes(mode: str) -> list[str]:
    if mode in {"both", "all", ""}:
        return ["baseline_manual", "behavior_adjusted_asof"]
    if mode in {"baseline_manual", "behavior_adjusted_asof"}:
        return [mode]
    raise ValueError(f"Unknown as-of backtest mode: {mode}")


def _canonical_teams(teams: list[str] | tuple[str, ...] | None) -> list[str] | None:
    if not teams:
        return None
    return [normalize_team_name(team) for team in teams if str(team).strip()]


def _ensure_asof_rating_columns(df: pd.DataFrame) -> None:
    defaults = {
        "attack_source": "",
        "defense_source": "",
        "recent_form_source": "",
        "behavior_blend_used": False,
        "as_of_date": "",
        "latest_behavior_match_used": "",
        "asof_behavior_matches_used_recent": 0,
        "lookahead_safe": True,
        "rating_warning": "",
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default


def _behavior_lookup(behavior: pd.DataFrame) -> dict[str, pd.Series]:
    if behavior is None or behavior.empty or "team" not in behavior.columns:
        return {}
    return {str(row.get("team", "")).lower(): row for _, row in behavior.iterrows()}


def _rating_frame(team_ratings_df: pd.DataFrame | None) -> pd.DataFrame:
    ratings = team_ratings_df.copy() if team_ratings_df is not None else pd.DataFrame(columns=TEAM_RATING_COLUMNS)
    for col in TEAM_RATING_COLUMNS:
        if col not in ratings.columns:
            ratings[col] = pd.NA
    return ratings[TEAM_RATING_COLUMNS].copy()


def _latest_before_asof(latest_match_used: Any, as_of_date: Any) -> bool:
    latest = str(latest_match_used or "").strip()
    if not latest:
        return True
    latest_ts = pd.to_datetime(latest, errors="coerce")
    if pd.isna(latest_ts):
        return False
    return latest_ts < _as_naive_timestamp(as_of_date)


def _as_naive_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts.normalize()


def _date_label(value: Any) -> str:
    return _as_naive_timestamp(value).date().isoformat()


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _numeric_group_column(group: pd.DataFrame, column: str) -> pd.Series:
    if column not in group.columns:
        return pd.Series([0.0] * len(group), index=group.index)
    return pd.to_numeric(group[column], errors="coerce").fillna(0.0)
