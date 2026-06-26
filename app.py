from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.alpha import calculate_polymarket_alpha
from src.asof_backtest import get_team_ratings_asof, run_asof_backtest_result
from src.backtest import run_backtest
from src.backtesting import evaluate_predictions, load_prediction_log, load_results_log, save_prediction_snapshot
from src.blend_sensitivity import run_blend_sensitivity_result
from src.behavior_driver_report import (
    audit_final_model_inputs,
    calculate_competition_breakdown,
    calculate_opponent_tier_breakdown,
    get_behavior_driver_matches,
)
from src.climate import get_team_training_climate, get_venue_environment, load_venues
from src.config import api_summary
from src.data_sources import filter_future_fixtures, get_upcoming_fixtures, update_all_sources
from src.environment_response import calculate_environment_response
from src.external_benchmark_calibration import (
    audit_external_benchmark_coverage,
    build_external_calibration_proposals,
    external_benchmark_coverage_summary,
    external_benchmark_calibration_summary,
    load_external_calibration_proposals,
)
from src.external_priors import (
    compare_ratings_to_external_priors,
    external_prior_review_summary,
    load_external_priors,
    load_manual_rating_review_sheet,
    load_review_proposals,
)
from src.feature_engineering import load_recent_matches
from src.fixture_diagnostics import audit_fixture_availability
from src.historical_binding import audit_historical_binding_for_match, audit_historical_file_status
from src.fifa_ranking_import import fifa_ranking_import_dashboard_status
from src.fifa_snapshot_validation import fifa_snapshot_validation_dashboard_status
from src.historical_data import load_historical_matches
from src.market_mapping import explain_unmapped_polymarket_markets, map_match_to_polymarket_markets, mapping_status
from src.model import ModelConfig, fair_odds, run_match_model
from src.model_policy import (
    annotate_decimal_alpha_with_policy,
    annotate_polymarket_alpha_with_policy,
    behavior_disagreement_warning,
    get_current_model_policy,
)
from src.odds import load_market_odds
from src.polymarket import get_match_polymarket_markets, get_polymarket_markets, update_polymarket_markets
from src.polymarket_match_search import build_polymarket_match_queries, score_polymarket_markets_for_match
from src.polymarket_sports_discovery import polymarket_discovery_cache_status
from src.polymarket_url_resolver import (
    parse_polymarket_url_or_slug,
    resolve_polymarket_event_markets_from_url_or_slug,
    resolved_url_markets_to_polymarket_rows,
    score_polymarket_event_slug_for_fixture,
)
from src.polymarket_slug_join import (
    build_polymarket_sports_slug_candidates,
    joined_market_groups,
    join_polymarket_prices_to_model_markets,
    load_polymarket_event_markets_by_slug,
    polymarket_alpha_rows,
    resolve_polymarket_slug_for_fixture,
)
from src.postmortem import build_postmortem_report, calculate_prediction_errors, join_predictions_to_results
from src.prediction_ledger import load_prediction_ledger, load_results_ledger
from src.rating_coverage import (
    audit_rating_coverage,
    collect_required_teams,
    load_team_ratings_csv,
    propose_team_aliases,
    rating_coverage_summary,
)
from src.rating_review import audit_generated_ratings, rating_review_summary
from src.recency import calculate_match_weight
from src.ratings import get_team_ratings
from src.report_charts import (
    create_compact_score_matrix,
    create_expected_goals_chart,
    create_goal_distribution_chart,
    create_match_outcome_donut,
    create_score_timeline_chart,
    create_team_ratings_chart,
)
from src.report_metrics import (
    build_climate_factor_table,
    build_data_support,
    build_expected_goals_df,
    calculate_goal_distribution,
    calculate_league_context,
    calculate_team_rating_percentiles,
)
from src.sensitivity import assess_alpha_robustness, run_sensitivity_analysis
from src.team_behavior import load_team_behavior
from src.team_names import load_team_name_aliases_df
from src.timeline import calculate_score_timeline
from src.tournament_learning import filter_tournament_learning_asof, load_tournament_learning_ledger


st.set_page_config(page_title="World Cup Alpha Model", layout="wide")


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def odds_fmt(value: float) -> str:
    if pd.isna(value):
        return ""
    if value == float("inf"):
        return "inf"
    return f"{value:.2f}"


def _first_value(df: pd.DataFrame, column: str, default: str = ""):
    if df is None or df.empty or column not in df.columns:
        return default
    value = df[column].iloc[0]
    if pd.isna(value):
        return default
    return value


def alpha_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["model_prob", "primary_model_probability", "behavior_diagnostic_probability"]:
        if col in out.columns:
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else pct(float(x)))
    if "behavior_probability_delta" in out.columns:
        out["behavior_probability_delta"] = out["behavior_probability_delta"].map(
            lambda x: "" if pd.isna(x) else f"{100*float(x):+.1f} pp"
        )
    out["fair_odds"] = out["fair_odds"].map(odds_fmt)
    out["market_odds"] = out["market_odds"].map(odds_fmt)
    out["alpha_ev"] = out["alpha_ev"].map(lambda x: "" if pd.isna(x) else f"{100*x:.1f}%")
    return out


def polymarket_alpha_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["model_probability", "primary_model_probability", "behavior_diagnostic_probability"]:
        if col in out.columns:
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{100*x:.1f}%")
    if "behavior_probability_delta" in out.columns:
        out["behavior_probability_delta"] = out["behavior_probability_delta"].map(
            lambda x: "" if pd.isna(x) else f"{100*float(x):+.1f} pp"
        )
    for col in ["fair_price_cents", "polymarket_price_cents", "alpha_gap_cents"]:
        if col in out.columns:
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.1f}")
    if "alpha_ev" in out.columns:
        out["alpha_ev"] = out["alpha_ev"].map(lambda x: "" if pd.isna(x) else f"{100*x:.1f}%")
    return out


def league_context_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def fmt(row, col):
        value = row[col]
        if pd.isna(value):
            return ""
        if row.get("unit") == "probability":
            return f"{100 * float(value):.1f}%"
        return f"{float(value):.2f}"

    def fmt_delta(row):
        value = row["Delta"]
        if pd.isna(value):
            return ""
        if row.get("unit") == "probability":
            return f"{100 * float(value):+.1f} pp"
        return f"{float(value):+.2f}"

    for col in ["League", "This match"]:
        out[col] = out.apply(lambda row: fmt(row, col), axis=1)
    out["Delta"] = out.apply(fmt_delta, axis=1)
    return out[["Metric", "League", "This match", "Delta"]]


def climate_factor_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out
    out["value"] = out.apply(
        lambda row: "" if pd.isna(row["value"]) else f"{float(row['value']):.1f} {row['unit']}",
        axis=1,
    )
    out["multiplier"] = out["multiplier"].map(lambda x: "" if pd.isna(x) else f"x{float(x):.3f}")
    return out[["factor", "value", "category", "favours", "multiplier"]]


def behavior_metric_display(value: float, as_pct: bool = False) -> str:
    if pd.isna(value):
        return ""
    return f"{100 * float(value):.1f}%" if as_pct else f"{float(value):.3f}"


def behavior_delta_display(value: float) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):+.3f}"


def build_source_status(fixtures, teams, venues, odds, historical_long_matches=None, team_behavior=None) -> pd.DataFrame:
    apis = api_summary()

    def source_label(df, fallback):
        try:
            return df.attrs.get("source_label", fallback)
        except Exception:
            return fallback

    def last_updated(df):
        try:
            return df.attrs.get("last_updated", "")
        except Exception:
            return ""

    def warning(df):
        try:
            return df.attrs.get("warning", "")
        except Exception:
            return ""

    def safe_len(df):
        try:
            return len(df) if df is not None else 0
        except Exception:
            return 0

    def missing_columns(df, required_cols):
        if df is None:
            return ", ".join(required_cols)
        try:
            return ", ".join([c for c in required_cols if c not in df.columns])
        except Exception:
            return ", ".join(required_cols)

    diagnostics = [
        {
            "input": "fixtures",
            "source": source_label(fixtures, "local CSV"),
            "rows": safe_len(fixtures),
            "api_configured": apis.get("fixtures", False),
            "last_updated": last_updated(fixtures),
            "warning": warning(fixtures),
            "missing_columns": missing_columns(
                fixtures,
                [
                    "match_id",
                    "date_utc",
                    "time_utc",
                    "competition",
                    "group",
                    "home",
                    "away",
                    "venue",
                    "city",
                    "country",
                ],
            ),
        },
        {
            "input": "team_ratings",
            "source": source_label(teams, "local CSV"),
            "rows": safe_len(teams),
            "api_configured": apis.get("ratings", False),
            "last_updated": last_updated(teams),
            "warning": warning(teams),
            "missing_columns": missing_columns(
                teams,
                [
                    "team",
                    "elo",
                    "attack",
                    "defense",
                    "recent_form",
                    "fifa_rank_proxy",
                    "training_temp_c",
                    "training_humidity_pct",
                    "data_quality",
                    "last_updated",
                    "notes",
                ],
            ),
        },
        {
            "input": "venues",
            "source": source_label(venues, "local CSV"),
            "rows": safe_len(venues),
            "api_configured": apis.get("weather", False),
            "last_updated": last_updated(venues),
            "warning": warning(venues),
            "missing_columns": missing_columns(
                venues,
                [
                    "venue",
                    "city",
                    "country",
                    "latitude",
                    "longitude",
                    "altitude_m",
                    "temp_c",
                    "humidity_pct",
                    "wind_kmh",
                    "precipitation_mm",
                    "roof_expected_closed",
                    "neutral_site",
                ],
            ),
        },
        {
            "input": "market_odds",
            "source": source_label(odds, "local CSV"),
            "rows": safe_len(odds),
            "api_configured": apis.get("odds", False),
            "last_updated": last_updated(odds),
            "warning": warning(odds),
            "missing_columns": missing_columns(
                odds,
                [
                    "match_id",
                    "market",
                    "selection",
                    "odds",
                    "source",
                    "last_updated",
                ],
            ),
        },
        {
            "input": "historical_matches",
            "source": source_label(historical_long_matches, "local CSV"),
            "rows": safe_len(historical_long_matches),
            "api_configured": apis.get("fixtures", False),
            "last_updated": last_updated(historical_long_matches),
            "warning": warning(historical_long_matches),
            "missing_columns": missing_columns(
                historical_long_matches,
                [
                    "match_id",
                    "date_utc",
                    "competition",
                    "competition_type",
                    "team",
                    "opponent",
                    "team_goals",
                    "opponent_goals",
                    "source",
                    "last_updated",
                ],
            ),
        },
        {
            "input": "team_behavior",
            "source": source_label(team_behavior, "local CSV"),
            "rows": safe_len(team_behavior),
            "api_configured": False,
            "last_updated": last_updated(team_behavior),
            "warning": warning(team_behavior),
            "missing_columns": missing_columns(
                team_behavior,
                [
                    "team",
                    "reference_date",
                    "n_matches",
                    "attack_index",
                    "defense_index",
                    "recent_form_index",
                    "overall_data_quality",
                    "last_updated",
                ],
            ),
        },
    ]

    for row in diagnostics:
        if row["missing_columns"]:
            row["warning"] = "Missing required columns"

    return pd.DataFrame(diagnostics)


def local_input_version() -> tuple[float, ...]:
    data_dir = Path("data")
    filenames = [
        "fixtures.csv",
        "team_ratings.csv",
        "venues.csv",
        "market_odds.csv",
        "historical_matches.csv",
        "team_behavior.csv",
    ]
    return tuple((data_dir / filename).stat().st_mtime if (data_dir / filename).exists() else 0.0 for filename in filenames)


def selected_behavior_rows(team_behavior: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    behavior = team_behavior.copy() if team_behavior is not None else pd.DataFrame()
    if behavior.empty or "team" not in behavior.columns:
        return pd.DataFrame()
    selected = behavior.loc[behavior["team"].astype(str).str.lower().isin([home.lower(), away.lower()])].copy()
    if selected.empty:
        return selected
    if "matches_used_recent" not in selected.columns and "n_matches" in selected.columns:
        selected["matches_used_recent"] = selected["n_matches"]
    columns = [
        "team",
        "matches_used_recent",
        "weighted_goals_for",
        "weighted_goals_against",
        "attack_index",
        "defense_index",
        "recent_form_index",
        "weighted_btts_rate",
        "weighted_over_2_5_rate",
        "clean_sheet_rate",
        "failed_to_score_rate",
        "overall_data_quality",
    ]
    for col in columns:
        if col not in selected.columns:
            selected[col] = pd.NA
    display = selected[columns].copy()
    for col in ["weighted_goals_for", "weighted_goals_against", "attack_index", "defense_index", "recent_form_index"]:
        display[col] = display[col].map(lambda x: behavior_metric_display(x))
    for col in ["weighted_btts_rate", "weighted_over_2_5_rate", "clean_sheet_rate", "failed_to_score_rate"]:
        display[col] = display[col].map(lambda x: behavior_metric_display(x, as_pct=True))
    return display


def behavior_window_display(team_behavior: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    behavior = team_behavior.copy() if team_behavior is not None else pd.DataFrame()
    if behavior.empty or "team" not in behavior.columns:
        return pd.DataFrame()
    selected = behavior.loc[behavior["team"].astype(str).str.lower().isin([team.lower() for team in teams])].copy()
    if selected.empty:
        return selected
    if "matches_used_recent" not in selected.columns and "n_matches" in selected.columns:
        selected["matches_used_recent"] = selected["n_matches"]
    columns = [
        "team",
        "matches_available_all_time",
        "matches_used_recent",
        "behavior_window_start",
        "oldest_match_used",
        "latest_match_used",
        "overall_data_quality",
        "behavior_config_name",
        "behavior_warning",
        "sample_size_warning",
        "staleness_warning",
        "opponent_quality_warning",
    ]
    for col in columns:
        if col not in selected.columns:
            selected[col] = pd.NA
    display = selected[columns].copy()
    display["warnings"] = display.apply(
        lambda row: str(row.get("behavior_warning", "") or "").strip()
        or "; ".join(
            [
                str(row[col])
                for col in ["sample_size_warning", "staleness_warning", "opponent_quality_warning"]
                if str(row.get(col, "") or "").strip()
            ]
        ),
        axis=1,
    )
    return display[
        [
            "team",
            "matches_available_all_time",
            "matches_used_recent",
            "behavior_window_start",
            "oldest_match_used",
            "latest_match_used",
            "overall_data_quality",
            "behavior_config_name",
            "warnings",
        ]
    ]


def recent_vs_all_time_display(team_behavior: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    behavior = team_behavior.copy() if team_behavior is not None else pd.DataFrame()
    if behavior.empty or "team" not in behavior.columns:
        return pd.DataFrame()
    selected = behavior.loc[behavior["team"].astype(str).str.lower().isin([team.lower() for team in teams])].copy()
    if selected.empty:
        return selected
    columns = [
        "team",
        "all_time_goals_for",
        "weighted_goals_for",
        "all_time_goals_against",
        "weighted_goals_against",
    ]
    for col in columns:
        if col not in selected.columns:
            selected[col] = pd.NA
    display = selected[columns].copy()
    for col in columns:
        if col != "team":
            display[col] = display[col].map(lambda x: behavior_metric_display(x))
    return display


def schedule_strength_display(team_behavior: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    behavior = team_behavior.copy() if team_behavior is not None else pd.DataFrame()
    if behavior.empty or "team" not in behavior.columns:
        return pd.DataFrame()
    selected = behavior.loc[behavior["team"].astype(str).str.lower().isin([team.lower() for team in teams])].copy()
    if selected.empty:
        return selected
    if "attack_index_adjusted_old" not in selected.columns and "attack_index_adjusted" in selected.columns:
        selected["attack_index_adjusted_old"] = selected["attack_index_adjusted"]
    if "defense_index_adjusted_old" not in selected.columns and "defense_index_adjusted" in selected.columns:
        selected["defense_index_adjusted_old"] = selected["defense_index_adjusted"]
    columns = [
        "team",
        "mean_opponent_elo_recent",
        "strong_opponent_match_count",
        "weak_opponent_match_count",
        "schedule_strength_label",
        "attack_index_raw",
        "attack_index_adjusted_old",
        "defense_index_raw",
        "defense_index_adjusted_old",
        "opponent_adjustment_warning",
        "schedule_strength_warning",
    ]
    for col in columns:
        if col not in selected.columns:
            selected[col] = pd.NA
    display = selected[columns].copy()
    for col in [
        "mean_opponent_elo_recent",
        "attack_index_raw",
        "attack_index_adjusted_old",
        "defense_index_raw",
        "defense_index_adjusted_old",
    ]:
        display[col] = display[col].map(lambda x: behavior_metric_display(x))
    display["warnings"] = display.apply(
        lambda row: "; ".join(
            [
                str(row.get(col, "") or "")
                for col in ["schedule_strength_warning", "opponent_adjustment_warning"]
                if str(row.get(col, "") or "").strip()
            ]
        ),
        axis=1,
    )
    return display[
        [
            "team",
            "mean_opponent_elo_recent",
            "strong_opponent_match_count",
            "weak_opponent_match_count",
            "schedule_strength_label",
            "attack_index_raw",
            "attack_index_adjusted_old",
            "defense_index_raw",
            "defense_index_adjusted_old",
            "warnings",
        ]
    ].rename(
        columns={
            "attack_index_adjusted_old": "attack_index_adjusted_old_opponent",
            "defense_index_adjusted_old": "defense_index_adjusted_old_opponent",
        }
    )


def expected_performance_display(team_behavior: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    behavior = team_behavior.copy() if team_behavior is not None else pd.DataFrame()
    if behavior.empty or "team" not in behavior.columns:
        return pd.DataFrame()
    selected = behavior.loc[behavior["team"].astype(str).str.lower().isin([team.lower() for team in teams])].copy()
    if selected.empty:
        return selected
    if "attack_index_adjusted_old" not in selected.columns and "attack_index_adjusted" in selected.columns:
        selected["attack_index_adjusted_old"] = selected["attack_index_adjusted"]
    if "defense_index_adjusted_old" not in selected.columns and "defense_index_adjusted" in selected.columns:
        selected["defense_index_adjusted_old"] = selected["defense_index_adjusted"]
    if "attack_index_residual_raw" not in selected.columns and "attack_index_residual" in selected.columns:
        selected["attack_index_residual_raw"] = selected["attack_index_residual"]
    if "attack_index_residual_robust" not in selected.columns and "attack_index_residual" in selected.columns:
        selected["attack_index_residual_robust"] = selected["attack_index_residual"]
    if "defense_index_residual_raw" not in selected.columns and "defense_index_residual" in selected.columns:
        selected["defense_index_residual_raw"] = selected["defense_index_residual"]
    if "defense_index_residual_robust" not in selected.columns and "defense_index_residual" in selected.columns:
        selected["defense_index_residual_robust"] = selected["defense_index_residual"]
    if "weighted_goal_for_residual_raw" not in selected.columns and "weighted_goal_for_residual" in selected.columns:
        selected["weighted_goal_for_residual_raw"] = selected["weighted_goal_for_residual"]
    if "weighted_goal_for_residual_robust" not in selected.columns and "weighted_goal_for_residual" in selected.columns:
        selected["weighted_goal_for_residual_robust"] = selected["weighted_goal_for_residual"]
    if "weighted_goal_against_residual_raw" not in selected.columns and "weighted_goal_against_residual" in selected.columns:
        selected["weighted_goal_against_residual_raw"] = selected["weighted_goal_against_residual"]
    if "weighted_goal_against_residual_robust" not in selected.columns and "weighted_goal_against_residual" in selected.columns:
        selected["weighted_goal_against_residual_robust"] = selected["weighted_goal_against_residual"]
    columns = [
        "team",
        "mean_opponent_elo_recent",
        "attack_index_raw",
        "attack_index_adjusted_old",
        "attack_index_residual_raw",
        "attack_index_residual_robust",
        "attack_index_final",
        "weighted_goal_for_residual_raw",
        "weighted_goal_for_residual_robust",
        "top_3_attack_residual_share",
        "defense_index_raw",
        "defense_index_adjusted_old",
        "defense_index_residual_raw",
        "defense_index_residual_robust",
        "defense_index_final",
        "weighted_goal_against_residual_raw",
        "weighted_goal_against_residual_robust",
        "top_3_defense_residual_share",
        "weighted_result_residual",
        "residual_coverage_recent",
        "residual_concentration_warning",
        "residual_warning",
        "opponent_adjustment_warning",
    ]
    for col in columns:
        if col not in selected.columns:
            selected[col] = pd.NA
    display = selected[columns].copy()
    for col in [
        "mean_opponent_elo_recent",
        "attack_index_raw",
        "attack_index_adjusted_old",
        "attack_index_residual_raw",
        "attack_index_residual_robust",
        "attack_index_final",
        "defense_index_raw",
        "defense_index_adjusted_old",
        "defense_index_residual_raw",
        "defense_index_residual_robust",
        "defense_index_final",
    ]:
        display[col] = display[col].map(lambda x: behavior_metric_display(x))
    for col in [
        "weighted_goal_for_residual_raw",
        "weighted_goal_for_residual_robust",
        "weighted_goal_against_residual_raw",
        "weighted_goal_against_residual_robust",
        "weighted_result_residual",
    ]:
        display[col] = display[col].map(lambda x: behavior_delta_display(x))
    for col in ["residual_coverage_recent", "top_3_attack_residual_share", "top_3_defense_residual_share"]:
        display[col] = display[col].map(lambda x: behavior_metric_display(x, as_pct=True))
    display["warnings"] = display.apply(
        lambda row: "; ".join(
            [
                str(row.get(col, "") or "")
                for col in ["residual_concentration_warning", "residual_warning", "opponent_adjustment_warning"]
                if str(row.get(col, "") or "").strip()
            ]
        ),
        axis=1,
    )
    return display[
        [
            "team",
            "mean_opponent_elo_recent",
            "attack_index_raw",
            "attack_index_adjusted_old",
            "attack_index_residual_raw",
            "attack_index_residual_robust",
            "attack_index_final",
            "weighted_goal_for_residual_raw",
            "weighted_goal_for_residual_robust",
            "top_3_attack_residual_share",
            "defense_index_raw",
            "defense_index_adjusted_old",
            "defense_index_residual_raw",
            "defense_index_residual_robust",
            "defense_index_final",
            "weighted_goal_against_residual_raw",
            "weighted_goal_against_residual_robust",
            "top_3_defense_residual_share",
            "weighted_result_residual",
            "residual_coverage_recent",
            "warnings",
        ]
    ].rename(
        columns={
            "attack_index_adjusted_old": "attack_index_adjusted_old_opponent",
            "defense_index_adjusted_old": "defense_index_adjusted_old_opponent",
        }
    )


def behavior_driver_matches_display(matches_df: pd.DataFrame, team: str, reference_date, sort_by: str, limit: int = 10) -> pd.DataFrame:
    drivers = get_behavior_driver_matches(matches_df, team, reference_date, n=limit, sort_by=sort_by)
    if drivers.empty:
        return drivers
    display = drivers.copy()
    numeric_cols = [
        "team_goals",
        "opponent_goals",
        "team_elo_pre",
        "opponent_elo",
        "match_weight",
        "blowout_weight",
        "expected_goals_for",
        "expected_goals_against",
        "goals_for_residual_raw",
        "goals_for_residual_robust",
        "goals_against_residual_raw",
        "goals_against_residual_robust",
        "attack_contribution",
        "defense_contribution",
    ]
    for col in numeric_cols:
        if col in display.columns:
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
    return display


def behavior_breakdown_display(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    display = df.copy()
    for col in [
        "weighted_goals_for",
        "weighted_goals_against",
        "weighted_goal_for_residual_robust",
        "weighted_goal_against_residual_robust",
        "attack_contribution",
        "defense_contribution",
    ]:
        if col in display.columns:
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
    return display


def final_model_input_audit_display(team_ratings: pd.DataFrame, team_behavior: pd.DataFrame, selected_teams: list[str]) -> pd.DataFrame:
    audit = audit_final_model_inputs(team_ratings, team_ratings, team_behavior)
    if audit.empty or "team" not in audit.columns:
        return audit
    wanted = {team.lower() for team in selected_teams}
    audit = audit.loc[audit["team"].astype(str).str.lower().isin(wanted)].copy()
    for col in [
        "manual_attack",
        "behavior_attack_final",
        "adjusted_attack_used_by_model",
        "attack_delta",
        "manual_defense",
        "behavior_defense_final",
        "adjusted_defense_used_by_model",
        "defense_delta",
        "manual_recent_form",
        "behavior_recent_form",
        "adjusted_recent_form_used_by_model",
        "recent_form_delta",
    ]:
        if col in audit.columns:
            audit[col] = audit[col].map(lambda x: behavior_delta_display(x) if col.endswith("_delta") else behavior_metric_display(x))
    return audit


def historical_match_history_display(matches_df: pd.DataFrame, team: str, reference_date, limit: int = 20) -> pd.DataFrame:
    matches = matches_df.copy() if matches_df is not None else pd.DataFrame()
    if matches.empty or "team" not in matches.columns:
        return pd.DataFrame()
    view = matches.loc[matches["team"].astype(str).str.lower() == str(team).lower()].copy()
    if view.empty:
        return view
    view["date_utc"] = pd.to_datetime(view["date_utc"], errors="coerce")
    view = view.dropna(subset=["date_utc"]).sort_values("date_utc", ascending=False).head(limit)
    view["weight"] = view.apply(lambda row: calculate_match_weight(row, reference_date), axis=1)
    view["score"] = (
        pd.to_numeric(view.get("team_goals"), errors="coerce").map(lambda x: "" if pd.isna(x) else str(int(x)))
        + "-"
        + pd.to_numeric(view.get("opponent_goals"), errors="coerce").map(lambda x: "" if pd.isna(x) else str(int(x)))
    )
    view["environment_flags"] = view.apply(_environment_flags, axis=1)
    for col in ["team_goals", "opponent_goals"]:
        view[col] = pd.to_numeric(view[col], errors="coerce")
    display = view[
        [
            "date_utc",
            "opponent",
            "competition",
            "score",
            "weight",
            "team_goals",
            "opponent_goals",
            "environment_flags",
        ]
    ].copy()
    display["date_utc"] = display["date_utc"].dt.date.astype(str)
    display["weight"] = display["weight"].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
    return display.rename(columns={"team_goals": "goals_for", "opponent_goals": "goals_against"})


def environment_response_display(matches_df: pd.DataFrame, teams: list[str], reference_date) -> pd.DataFrame:
    rows = [calculate_environment_response(matches_df, team, reference_date) for team in teams]
    display = pd.DataFrame(rows)
    if display.empty:
        return display
    columns = [
        "team",
        "environment_sample_size",
        "hot_match_count",
        "humid_match_count",
        "altitude_match_count",
        "hot_attack_delta",
        "hot_defense_delta",
        "humid_attack_delta",
        "humid_defense_delta",
        "altitude_attack_delta",
        "altitude_defense_delta",
        "environment_response_index",
        "environment_data_quality",
        "environment_warning",
    ]
    for col in columns:
        if col not in display.columns:
            display[col] = pd.NA
    for col in [c for c in columns if c.endswith("_delta") or c == "environment_response_index"]:
        display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
    return display[columns]


def model_impact_display(result: dict) -> pd.DataFrame:
    rows = []
    for side in ["home_inputs", "away_inputs"]:
        inputs = result.get(side, pd.Series(dtype="object"))
        team = str(inputs.get("team", ""))
        for label, manual_col, final_col, behavior_col, source_col, delta_col, cap_col in [
            ("Attack", "manual_attack", "attack", "behavior_attack_index", "attack_source", "behavior_attack_delta", "behavior_attack_cap_hit"),
            ("Defense", "manual_defense", "defense", "behavior_defense_index", "defense_source", "behavior_defense_delta", "behavior_defense_cap_hit"),
            (
                "Recent form",
                "manual_recent_form",
                "recent_form",
                "behavior_recent_form_index",
                "recent_form_source",
                "behavior_recent_form_delta",
                "behavior_recent_form_cap_hit",
            ),
        ]:
            delta = inputs.get(delta_col)
            if pd.isna(delta):
                manual = inputs.get(manual_col)
                final = inputs.get(final_col)
                delta = pd.NA if pd.isna(manual) or pd.isna(final) else float(final) - float(manual)
            rows.append(
                {
                    "team": team,
                    "input": label,
                    "manual_value": behavior_metric_display(inputs.get(manual_col)),
                    "behavior_index": behavior_metric_display(inputs.get(behavior_col)),
                    "model_input_after_blend": behavior_metric_display(inputs.get(final_col)),
                    "delta": behavior_delta_display(delta),
                    "cap_hit": _truthy(inputs.get(cap_col, False)),
                    "source": inputs.get(source_col, ""),
                    "blend_used": _truthy(inputs.get("behavior_blend_used", False)),
                    "behavior_quality": inputs.get("behavior_overall_data_quality", ""),
                }
            )
    return pd.DataFrame(rows)


def historical_behavior_overview(matches_df: pd.DataFrame, behavior_df: pd.DataFrame, teams: list[str]) -> pd.DataFrame:
    matches = matches_df.copy() if matches_df is not None else pd.DataFrame()
    behavior = behavior_df.copy() if behavior_df is not None else pd.DataFrame()
    rows = []
    for team in teams:
        team_rows = pd.DataFrame()
        if not matches.empty and "team" in matches.columns:
            team_rows = matches.loc[matches["team"].astype(str).str.lower() == team.lower()].copy()
        latest = ""
        if not team_rows.empty and "date_utc" in team_rows.columns:
            dates = pd.to_datetime(team_rows["date_utc"], errors="coerce").dropna()
            latest = dates.max().date().isoformat() if not dates.empty else ""
        quality = ""
        if not behavior.empty and "team" in behavior.columns:
            row = behavior.loc[behavior["team"].astype(str).str.lower() == team.lower()]
            if not row.empty:
                quality = str(row.iloc[0].get("overall_data_quality", "") or "")
        rows.append(
            {
                "team": team,
                "historical_rows": len(team_rows),
                "latest_match_date": latest,
                "behavior_quality": quality or "none",
            }
        )
    return pd.DataFrame(rows)


def _environment_flags(row: pd.Series) -> str:
    flags = []
    temperature = _row_float(row, "temperature_c")
    humidity = _row_float(row, "humidity_pct")
    altitude = _row_float(row, "altitude_m")
    wind = _row_float(row, "wind_kmh")
    precipitation = _row_float(row, "precipitation_mm")
    roof_closed = _row_float(row, "roof_closed")
    if temperature is not None and temperature >= 28:
        flags.append("hot")
    if humidity is not None and humidity >= 70:
        flags.append("humid")
    if altitude is not None and altitude >= 1000:
        flags.append("altitude")
    if wind is not None and wind >= 20:
        flags.append("wind")
    if precipitation is not None and precipitation > 0:
        flags.append("rain")
    if roof_closed is not None and roof_closed >= 1:
        flags.append("roof closed")
    return ", ".join(flags)


def _row_float(row: pd.Series, column: str) -> float | None:
    value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
    if pd.isna(value):
        return None
    return float(value)


def _truthy(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


@st.cache_data(ttl=600)
def load_inputs(
    start_date: date,
    end_date: date,
    refresh_counter: int,
    input_version: tuple[float, ...],
    now_utc_iso: str,
    horizon_hours: float | None,
    include_past: bool,
):
    raw_fixtures = get_upcoming_fixtures(start_date, end_date)
    fixture_audit, fixture_audit_summary = audit_fixture_availability(
        raw_fixtures,
        start_date=start_date,
        end_date=end_date,
        hide_past=not include_past,
        now_utc=now_utc_iso,
    )
    fixtures = filter_future_fixtures(
        raw_fixtures,
        now_utc=now_utc_iso,
        horizon_hours=horizon_hours,
        include_past=include_past,
    )
    teams = get_team_ratings()
    venues = load_venues()
    odds = load_market_odds()
    recent_matches = load_recent_matches()
    historical_long_matches = load_historical_matches()
    team_behavior = load_team_behavior()
    status = build_source_status(fixtures, teams, venues, odds, historical_long_matches, team_behavior)
    return (
        fixtures,
        teams,
        venues,
        odds,
        recent_matches,
        historical_long_matches,
        team_behavior,
        status,
        fixture_audit,
        fixture_audit_summary,
    )


@st.cache_data(ttl=600)
def load_polymarket_inputs(query: str, use_cache: bool, refresh_counter: int):
    return get_polymarket_markets(query=query or None, use_cache=use_cache)


@st.cache_data(ttl=600)
def load_match_polymarket_inputs(home: str, away: str, date_utc: str, refresh_counter: int):
    return get_match_polymarket_markets(home, away, date_utc)


@st.cache_data(ttl=600)
def load_polymarket_url_or_slug_inputs(value: str, refresh_counter: int):
    return resolve_polymarket_event_markets_from_url_or_slug(value)


@st.cache_data(ttl=600)
def load_polymarket_event_markets_for_slug(slug: str, refresh_counter: int):
    return load_polymarket_event_markets_by_slug(slug)


@st.cache_data(ttl=600)
def load_behavior_backtest_results(
    start_date_iso: str,
    end_date_iso: str,
    competition: str,
    refresh_counter: int,
):
    return run_backtest(
        start_date=start_date_iso,
        end_date=end_date_iso,
        competition=competition or None,
        strict_as_of_date=False,
    )


@st.cache_data(ttl=600)
def load_asof_backtest_results(
    start_date_iso: str,
    end_date_iso: str,
    competition: str,
    refresh_counter: int,
):
    return run_asof_backtest_result(
        start_date=start_date_iso,
        end_date=end_date_iso,
        competition=competition or None,
    )


@st.cache_data(ttl=600)
def load_blend_sensitivity_results(
    start_date_iso: str,
    end_date_iso: str,
    competition: str,
    refresh_counter: int,
):
    return run_blend_sensitivity_result(
        start_date=start_date_iso,
        end_date=end_date_iso,
        competition=competition or None,
    )


def build_behavior_diagnostic_result(
    match: pd.Series,
    historical_matches: pd.DataFrame,
    venues: pd.DataFrame,
    market_odds: pd.DataFrame,
    cfg: ModelConfig,
) -> tuple[dict | None, str]:
    try:
        policy = get_current_model_policy()
        base_ratings = load_team_ratings_csv()
        teams_asof = get_team_ratings_asof(
            base_ratings,
            historical_matches,
            match.get("date_utc"),
            teams=[str(match.get("home", "")), str(match.get("away", ""))],
            mode=str(policy["secondary_model_mode"]),
            behavior_blend_multiplier=1.0,
        )
        diagnostic = run_match_model(
            match,
            teams_asof,
            venues,
            market_odds,
            cfg,
            model_mode=str(policy["secondary_model_mode"]),
        )
        return diagnostic, ""
    except Exception as exc:
        return None, f"Behavior diagnostic unavailable: {exc}"


def merge_polymarket_inputs(primary: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
    frames = [df for df in [primary, secondary] if df is not None and not df.empty]
    if not frames:
        out = primary.copy() if primary is not None else pd.DataFrame()
    else:
        out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["market_id"]).reset_index(drop=True)
    source_labels = list(
        dict.fromkeys(
            str(df.attrs.get("source_label", "")).strip()
            for df in [primary, secondary]
            if df is not None and str(df.attrs.get("source_label", "")).strip()
        )
    )
    warnings = list(
        dict.fromkeys(
            str(df.attrs.get("warning", "")).strip()
            for df in [primary, secondary]
            if df is not None and str(df.attrs.get("warning", "")).strip()
        )
    )
    out.attrs["source_label"] = " + ".join(source_labels) if source_labels else "unknown source"
    if out.empty and warnings:
        out.attrs["warning"] = "; ".join(warnings)
    return out


st.title("FIFA World Cup Match Alpha Model")
current_model_policy = get_current_model_policy()
st.caption(
    "Transparent local model for statistical alpha estimates. It estimates probabilities, fair odds, scorelines, "
    "environmental effects, and alpha EV versus optional market odds. It does not provide staking or investment advice."
)

if "refresh_counter" not in st.session_state:
    st.session_state["refresh_counter"] = 0
if "polymarket_refresh_counter" not in st.session_state:
    st.session_state["polymarket_refresh_counter"] = 0

with st.sidebar:
    st.header("Fixture Window")
    utc_today = pd.Timestamp.now(tz="UTC").date()
    window = st.radio(
        "Quick range",
        ["Upcoming 48 hours", "Today (UTC)", "Tomorrow (UTC)", "Next 7 days", "Custom UTC date range"],
        index=0,
    )
    horizon_hours = None
    if window == "Upcoming 48 hours":
        start_date = utc_today
        end_date = utc_today + timedelta(days=2)
        horizon_hours = 48.0
    elif window == "Today (UTC)":
        anchor_date = st.date_input("Anchor date (UTC)", value=utc_today)
        start_date = anchor_date
        end_date = anchor_date
    elif window == "Tomorrow (UTC)":
        anchor_date = st.date_input("Anchor date (UTC)", value=utc_today)
        start_date = anchor_date + timedelta(days=1)
        end_date = start_date
    elif window == "Next 7 days":
        anchor_date = st.date_input("Anchor date (UTC)", value=utc_today)
        start_date = anchor_date
        end_date = anchor_date + timedelta(days=7)
    else:
        custom_start = st.date_input("Range start (UTC)", value=utc_today)
        custom_end = st.date_input("Range end (UTC)", value=utc_today + timedelta(days=7))
        if custom_end < custom_start:
            st.warning("Range end is before range start, so the app will swap them.")
        start_date, end_date = sorted((custom_start, custom_end))
    hide_past_kickoffs = st.checkbox("Hide past kickoffs", value=True)
    st.caption(f"UTC fixture window: {start_date.isoformat()} to {end_date.isoformat()}.")

    st.header("Data")
    if st.button("Update API data", width="stretch"):
        with st.spinner("Refreshing configured APIs and local caches..."):
            summary = update_all_sources(start_date, end_date)
        st.session_state["refresh_counter"] += 1
        st.cache_data.clear()
        st.session_state["last_update_summary"] = summary

    if "last_update_summary" in st.session_state:
        st.caption(f"Last update: {st.session_state['last_update_summary'].get('updated_at', '')}")
        for step in st.session_state["last_update_summary"].get("steps", []):
            status = step.get("status", "")
            name = step.get("name", "source")
            message = step.get("message") or f"{step.get('rows', 0)} row(s); source: {step.get('source', '')}"
            if status == "success":
                st.success(f"{name}: {message}")
            elif status in {"partial", "skipped"}:
                st.warning(f"{name}: {message}")
            elif status == "failed":
                st.error(f"{name}: {message}")
            if step.get("warning"):
                st.warning(f"{name}: {step['warning']}")
            for error in step.get("errors", [])[:3]:
                st.warning(f"{name}: {error}")

    st.header("Polymarket")
    polymarket_query = st.text_input("Market search query", value="World Cup")
    use_cached_polymarket = st.checkbox("Use cached markets", value=True)
    min_liquidity = st.number_input("Minimum liquidity", min_value=0.0, value=0.0, step=25.0)
    min_alpha_gap = st.slider("Minimum alpha gap (cents)", 0.0, 20.0, 0.0, 0.5)
    show_only_mapped = st.toggle("Show only mapped markets", value=True)
    search_selected_match = st.toggle("Search selected match markets", value=True)
    polymarket_event_url_or_slug = st.text_input(
        "Optional Polymarket event URL or slug",
        value="",
        help="Paste a Polymarket sports event URL when automatic search cannot find the selected match.",
    )
    show_behavior_diagnostic_probabilities = st.toggle("Show diagnostic behavior probabilities", value=False)

    if st.button("Update Polymarket markets", width="stretch"):
        with st.spinner("Refreshing Polymarket markets if an API is configured..."):
            pm_summary = update_polymarket_markets(polymarket_query or None)
        st.session_state["polymarket_refresh_counter"] += 1
        st.cache_data.clear()
        st.session_state["last_polymarket_update"] = pm_summary

    if "last_polymarket_update" in st.session_state:
        pm_summary = st.session_state["last_polymarket_update"]
        message = pm_summary.get("message", "")
        if pm_summary.get("status") == "success":
            st.success(message)
        else:
            st.warning(message)
        if pm_summary.get("warning"):
            st.warning(pm_summary["warning"])

    st.header("Model Parameters")
    base_total = st.slider("Base total goals", 1.8, 3.2, 2.50, 0.05)
    elo_weight = st.slider(
        "Elo xG weight",
        0.0010,
        0.0060,
        0.0032,
        0.0001,
        format="%.4f",
    )
    attack_weight = st.slider("Attack weight", 0.20, 0.90, 0.55, 0.05)
    defense_weight = st.slider("Defense weight", 0.20, 0.90, 0.45, 0.05)
    form_weight = st.slider("Recent form weight", 0.00, 0.40, 0.20, 0.02)
    env_weight = st.slider("Environment weight", 0.000, 0.030, 0.010, 0.001)

(
    fixtures,
    teams,
    venues,
    market_odds,
    recent_matches,
    historical_long_matches,
    team_behavior,
    source_status,
    fixture_audit,
    fixture_audit_summary,
) = load_inputs(
    start_date,
    end_date,
    st.session_state["refresh_counter"],
    local_input_version(),
    pd.Timestamp.now(tz="UTC").floor("min").isoformat(),
    horizon_hours,
    include_past=not hide_past_kickoffs,
)
polymarket_markets = load_polymarket_inputs(
    polymarket_query,
    use_cached_polymarket,
    st.session_state["polymarket_refresh_counter"],
)

with st.sidebar:
    st.header("Source Diagnostics")
    st.dataframe(
        source_status[["input", "source", "rows", "api_configured", "last_updated", "warning"]],
        hide_index=True,
        width="stretch",
    )
    st.caption("Source labels are API, cache, or local CSV. API keys are optional; failed requests fall back safely.")
    for _, source_row in source_status.iterrows():
        if source_row.get("missing_columns"):
            st.warning(f"{source_row['input']} missing columns: {source_row['missing_columns']}")
    st.caption(
        "Polymarket markets: "
        f"{len(polymarket_markets)} rows from {polymarket_markets.attrs.get('source_label', 'unknown source')}."
    )
    if polymarket_markets.attrs.get("warning"):
        st.warning(polymarket_markets.attrs["warning"])
    if polymarket_markets.empty:
        st.warning(
            "No Polymarket market rows are loaded. The default Gamma API returned no usable rows, "
            "the selected query found no matches, or the request failed. You can also paste fallback "
            "rows into `data/polymarket_markets.csv`."
        )

    st.header("Data Health")
    st.write("Fixture availability")
    fa = fixture_audit_summary
    st.caption(
        f"Loaded: {fa.get('total_fixtures_loaded', 0)} | "
        f"Visible: {fa.get('fixtures_visible', 0)} | "
        f"Hidden as past: {fa.get('fixtures_hidden_as_past', 0)}"
    )
    st.caption(
        f"Current UTC: {fa.get('current_utc', '')} | "
        f"Selected UTC: {fa.get('selected_start_utc', '')} to {fa.get('selected_end_utc', '')}"
    )
    st.caption(
        f"Earliest fixture: {fa.get('earliest_kickoff_utc', '') or 'n/a'} | "
        f"Latest fixture: {fa.get('latest_kickoff_utc', '') or 'n/a'}"
    )
    if fa.get("warning"):
        st.warning(fa["warning"])

    historical_status = audit_historical_file_status(historical_long_matches)
    st.write("Historical data file")
    st.caption(
        f"Rows: {historical_status.get('rows_loaded', 0):,} | "
        f"Teams: {historical_status.get('unique_teams', 0)} | "
        f"Range: {historical_status.get('earliest_date', '') or 'n/a'} to "
        f"{historical_status.get('latest_date', '') or 'n/a'}"
    )
    if historical_status.get("warning"):
        st.warning(historical_status["warning"])

    st.header("Model Policy")
    st.write("Primary model: Baseline external-calibrated")
    st.write("Behavior layer: Diagnostic only")
    st.caption(current_model_policy["reason"])

st.subheader("Available Matches")
if fixtures.empty:
    st.warning("No fixtures found for the selected UTC window. Add rows to `data/fixtures.csv`, widen the date range, or configure a fixture API.")
    st.stop()

fixture_view = fixtures.copy()
fixture_view["match_label"] = (
    fixture_view["date_utc"].astype(str)
    + " "
    + fixture_view["time_utc"].astype(str)
    + " UTC - "
    + fixture_view["home"].astype(str)
    + " vs "
    + fixture_view["away"].astype(str)
)

st.dataframe(
    fixture_view[["match_id", "kickoff_utc", "competition", "group", "home", "away", "venue", "city"]],
    width="stretch",
    hide_index=True,
)
if hide_past_kickoffs:
    st.caption("Only fixtures with future UTC kickoff times are shown. Clear the sidebar checkbox to inspect past kickoffs in the selected UTC range.")
else:
    st.caption("All fixtures in the selected UTC date range are shown, including past kickoffs.")

selected_labels = st.multiselect(
    "Select one or more games to model",
    options=fixture_view["match_label"].tolist(),
    default=[],
)

if not selected_labels:
    st.info("Select a match above to run the model.")
    st.stop()

cfg = ModelConfig(
    base_total_goals=base_total,
    environment_weight=env_weight,
    elo_xg_weight=elo_weight,
    attack_weight=attack_weight,
    defense_weight=defense_weight,
    form_weight=form_weight,
)

for label in selected_labels:
    match = fixture_view.loc[fixture_view["match_label"] == label].iloc[0]
    result = run_match_model(
        match,
        teams,
        venues,
        market_odds,
        cfg,
        model_mode=str(current_model_policy["primary_model_mode"]),
    )
    behavior_diagnostic_result, behavior_diagnostic_error = build_behavior_diagnostic_result(
        match,
        historical_long_matches,
        venues,
        market_odds,
        cfg,
    )
    behavior_warning = behavior_disagreement_warning(result, behavior_diagnostic_result)

    probs = result["probs"]
    behavior_alpha = behavior_diagnostic_result.get("alpha", pd.DataFrame()) if behavior_diagnostic_result else pd.DataFrame()
    alpha = annotate_decimal_alpha_with_policy(
        result["alpha"].copy(),
        behavior_alpha,
        show_behavior_diagnostic=show_behavior_diagnostic_probabilities,
        policy=current_model_policy,
    )

    if "odds_source" not in alpha.columns:
        alpha["odds_source"] = ""

    if "odds_last_updated" not in alpha.columns:
        alpha["odds_last_updated"] = ""

    confidence = result["confidence"]
    slug_candidates = build_polymarket_sports_slug_candidates(
        str(match.get("home", "")),
        str(match.get("away", "")),
        str(match.get("date_utc", "")),
        str(match.get("competition", "World Cup") or "World Cup"),
    )
    slug_resolution = resolve_polymarket_slug_for_fixture(
        str(match.get("home", "")),
        str(match.get("away", "")),
        str(match.get("date_utc", "")),
        str(match.get("competition", "World Cup") or "World Cup"),
        user_supplied_slug_or_url=polymarket_event_url_or_slug.strip() or None,
    )
    resolved_event_markets = pd.DataFrame()
    if slug_resolution.get("resolved_slug"):
        resolved_event_markets = load_polymarket_event_markets_for_slug(
            str(slug_resolution.get("resolved_slug", "")),
            st.session_state["polymarket_refresh_counter"],
        )
    joined_polymarket_markets = join_polymarket_prices_to_model_markets(
        alpha,
        resolved_event_markets,
        str(match.get("home", "")),
        str(match.get("away", "")),
    )
    joined_polymarket_alpha_rows = polymarket_alpha_rows(joined_polymarket_markets)
    mapped_markets = map_match_to_polymarket_markets(match, polymarket_markets)
    match_polymarket_markets = pd.DataFrame()
    url_resolved_markets = pd.DataFrame()
    url_resolved_market_rows = pd.DataFrame()
    url_resolver_diagnostics: dict = {}
    url_slug_score: dict = {}
    mapping_market_pool = polymarket_markets
    if mapped_markets.empty and search_selected_match:
        match_polymarket_markets = load_match_polymarket_inputs(
            str(match.get("home", "")),
            str(match.get("away", "")),
            str(match.get("date_utc", "")),
            st.session_state["polymarket_refresh_counter"],
        )
        mapping_market_pool = merge_polymarket_inputs(polymarket_markets, match_polymarket_markets)
        mapped_markets = map_match_to_polymarket_markets(match, mapping_market_pool)
    if polymarket_event_url_or_slug.strip():
        url_resolved_markets, url_resolver_diagnostics = load_polymarket_url_or_slug_inputs(
            polymarket_event_url_or_slug.strip(),
            st.session_state["polymarket_refresh_counter"],
        )
        parsed_url = parse_polymarket_url_or_slug(polymarket_event_url_or_slug.strip())
        url_slug_score = score_polymarket_event_slug_for_fixture(
            parsed_url.get("slug", ""),
            str(match.get("home", "")),
            str(match.get("away", "")),
            fixture_date=str(match.get("date_utc", "") or ""),
        )
        url_resolved_market_rows = resolved_url_markets_to_polymarket_rows(url_resolved_markets)
        if not url_resolved_market_rows.empty:
            mapping_market_pool = merge_polymarket_inputs(mapping_market_pool, url_resolved_market_rows)
            mapped_markets = map_match_to_polymarket_markets(match, mapping_market_pool)
    unmapped_diagnostics = (
        explain_unmapped_polymarket_markets(match, mapping_market_pool)
        if mapped_markets.empty and not mapping_market_pool.empty
        else pd.DataFrame()
    )
    polymarket_search_queries = build_polymarket_match_queries(
        str(match.get("home", "")),
        str(match.get("away", "")),
        str(match.get("competition", "World Cup") or "World Cup"),
    )
    polymarket_search_diagnostics = score_polymarket_markets_for_match(
        mapping_market_pool,
        str(match.get("home", "")),
        str(match.get("away", "")),
        fixture_date=str(match.get("date_utc", "") or ""),
        competition=str(match.get("competition", "World Cup") or "World Cup"),
        include_rejected=True,
    )
    polymarket_alpha = calculate_polymarket_alpha(result, mapped_markets, min_liquidity=min_liquidity)
    polymarket_alpha = annotate_polymarket_alpha_with_policy(
        polymarket_alpha,
        behavior_diagnostic_result,
        show_behavior_diagnostic=show_behavior_diagnostic_probabilities,
        policy=current_model_policy,
    )
    sensitivity_df = run_sensitivity_analysis(match, teams, venues, market_odds)
    robustness_df = assess_alpha_robustness(polymarket_alpha, sensitivity_df)
    if not polymarket_alpha.empty and not robustness_df.empty:
        polymarket_alpha = polymarket_alpha.merge(robustness_df, on="market_id", how="left")

    filtered_polymarket_alpha = polymarket_alpha.copy()
    if min_liquidity > 0 and not filtered_polymarket_alpha.empty:
        filtered_polymarket_alpha = filtered_polymarket_alpha.loc[
            pd.to_numeric(filtered_polymarket_alpha["liquidity"], errors="coerce").fillna(-1) >= min_liquidity
        ]
    if min_alpha_gap > 0 and not filtered_polymarket_alpha.empty:
        filtered_polymarket_alpha = filtered_polymarket_alpha.loc[
            pd.to_numeric(filtered_polymarket_alpha["alpha_gap_cents"], errors="coerce").fillna(-999) >= min_alpha_gap
        ]

    venue_env = result.get("environment", {})
    data_support = build_data_support(match, historical_long_matches)
    goal_distribution = calculate_goal_distribution(result.get("score_matrix"), max_goals=14)
    league_context = calculate_league_context(recent_matches, result)
    score_timeline = calculate_score_timeline(result["hxg"], result["axg"])
    expected_goals_df = build_expected_goals_df(result)
    rating_percentiles = calculate_team_rating_percentiles(teams, result["home"], result["away"])
    climate_factors = build_climate_factor_table(result, venue_env)
    market_groups = joined_market_groups(joined_polymarket_markets)

    st.divider()
    st.header(f"{result['home']} vs {result['away']}")
    st.caption(
        f"{match['competition']} | {match['group']} | "
        f"{match['date_utc']} {match['time_utc']} UTC | {match['venue']}"
    )
    policy_cols = st.columns(2)
    policy_cols[0].metric("Primary model", "Baseline external-calibrated")
    policy_cols[1].metric("Behavior layer", "Diagnostic only")
    st.info(
        "The model tested recent historical behavior fairly using only pre-match data. "
        "In that strict test, recent behavior did not improve both Brier score and log loss. "
        "The main probabilities and alpha tables therefore use external-calibrated team ratings; "
        "behavior is shown only as a warning or context signal."
    )
    if behavior_diagnostic_error:
        st.caption(behavior_diagnostic_error)
    if behavior_warning:
        st.warning(behavior_warning)

    aliases_df = load_team_name_aliases_df()
    selected_binding = audit_historical_binding_for_match(
        result["home"],
        result["away"],
        historical_long_matches,
        team_behavior_df=team_behavior,
        team_ratings_df=teams,
        aliases_df=aliases_df,
        as_of_date=str(match.get("date_utc", "") or ""),
    )
    with st.expander("Data Health", expanded=False):
        st.caption(
            "This checks whether fixture names bind correctly to historical matches, ratings, "
            "and external benchmarks before the report interprets missing data."
        )
        dh_cols = st.columns(4)
        dh_cols[0].metric("Canonical home", selected_binding["home_canonical"])
        dh_cols[1].metric("Canonical away", selected_binding["away_canonical"])
        dh_cols[2].metric("Home historical rows", selected_binding["home_historical_rows_before_asof"])
        dh_cols[3].metric("Away historical rows", selected_binding["away_historical_rows_before_asof"])
        dh_cols2 = st.columns(4)
        dh_cols2[0].metric("H2H rows", selected_binding["h2h_rows_before_asof"])
        dh_cols2[1].metric("Latest home match", selected_binding["home_latest_match"] or "n/a")
        dh_cols2[2].metric("Latest away match", selected_binding["away_latest_match"] or "n/a")
        dh_cols2[3].metric("Binding status", selected_binding["data_binding_status"])

        rating_rows = teams.copy()
        if "team" in rating_rows.columns:
            rating_rows["_team_key"] = rating_rows["team"].astype(str).str.lower()
        home_rating = rating_rows.loc[rating_rows.get("_team_key", pd.Series(dtype=str)) == selected_binding["home_canonical"].lower()]
        away_rating = rating_rows.loc[rating_rows.get("_team_key", pd.Series(dtype=str)) == selected_binding["away_canonical"].lower()]
        external_priors = load_external_priors()
        if "team" in external_priors.columns:
            external_priors["_team_key"] = external_priors["team"].astype(str).str.lower()
        home_external = external_priors.loc[
            external_priors.get("_team_key", pd.Series(dtype=str)) == selected_binding["home_canonical"].lower()
        ]
        away_external = external_priors.loc[
            external_priors.get("_team_key", pd.Series(dtype=str)) == selected_binding["away_canonical"].lower()
        ]
        st.write("Selected match source binding")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "team": selected_binding["home_canonical"],
                        "rating_found": selected_binding["home_rating_found"],
                        "behavior_found": selected_binding["home_behavior_found"],
                        "rating_source": _first_value(home_rating, "data_quality"),
                        "external_benchmark": _first_value(home_external, "reference_overall_strength"),
                    },
                    {
                        "team": selected_binding["away_canonical"],
                        "rating_found": selected_binding["away_rating_found"],
                        "behavior_found": selected_binding["away_behavior_found"],
                        "rating_source": _first_value(away_rating, "data_quality"),
                        "external_benchmark": _first_value(away_external, "reference_overall_strength"),
                    },
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        if selected_binding.get("warning"):
            st.warning(selected_binding["warning"])

    tabs = st.tabs(
        [
            "Summary",
            "Full report",
            "Score matrix",
            "Markets",
            "Polymarket alpha",
            "Sensitivity",
            "Backtesting",
            "Team inputs",
            "Venue/environment",
            "Historical behavior",
            "Model notes",
            "Post-mortem Training",
        ]
    )

    with tabs[0]:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"{result['home']} win", pct(probs["home_win"]), f"fair {fair_odds(probs['home_win']):.2f}")
        c2.metric("Draw", pct(probs["draw"]), f"fair {fair_odds(probs['draw']):.2f}")
        c3.metric(f"{result['away']} win", pct(probs["away_win"]), f"fair {fair_odds(probs['away_win']):.2f}")
        c4.metric("Expected goals", f"{result['hxg']:.2f} - {result['axg']:.2f}")

        if behavior_diagnostic_result:
            behavior_probs = behavior_diagnostic_result.get("probs", {})
            diagnostic_summary = pd.DataFrame(
                [
                    {
                        "outcome": f"{result['home']} win",
                        "primary_probability": probs["home_win"],
                        "behavior_diagnostic_probability": behavior_probs.get("home_win", pd.NA),
                    },
                    {
                        "outcome": "Draw",
                        "primary_probability": probs["draw"],
                        "behavior_diagnostic_probability": behavior_probs.get("draw", pd.NA),
                    },
                    {
                        "outcome": f"{result['away']} win",
                        "primary_probability": probs["away_win"],
                        "behavior_diagnostic_probability": behavior_probs.get("away_win", pd.NA),
                    },
                ]
            )
            diagnostic_summary["delta"] = (
                pd.to_numeric(diagnostic_summary["behavior_diagnostic_probability"], errors="coerce")
                - pd.to_numeric(diagnostic_summary["primary_probability"], errors="coerce")
            )
            display_diag = diagnostic_summary.copy()
            for col in ["primary_probability", "behavior_diagnostic_probability"]:
                display_diag[col] = display_diag[col].map(lambda x: "" if pd.isna(x) else pct(float(x)))
            display_diag["delta"] = display_diag["delta"].map(lambda x: "" if pd.isna(x) else f"{100*float(x):+.1f} pp")
            st.dataframe(display_diag, hide_index=True, width="stretch")
            st.caption("Behavior probabilities are diagnostic-only and do not drive the primary alpha tables.")

        st.subheader("Model Confidence")
        st.write(f"**{confidence['label']} confidence**: {', '.join(confidence['reasons'])}.")

        st.subheader("Top Alpha Signals")
        positive_alpha = alpha.loc[alpha["alpha_ev"].notna() & (alpha["alpha_ev"] > 0)].head(5)
        if positive_alpha.empty:
            st.caption("No positive alpha EV rows are available from the current market odds file.")
        else:
            st.dataframe(
                alpha_display(positive_alpha)[["market", "selection", "model_prob", "fair_odds", "market_odds", "alpha_ev"]],
                hide_index=True,
                width="stretch",
            )
        st.caption("Alpha EV = model_probability x market_decimal_odds - 1. This is a statistical estimate, not a staking instruction.")

    with tabs[1]:
        st.subheader("Alpha Read")
        top_alpha_rows = alpha.loc[alpha["alpha_ev"].notna()].sort_values("alpha_ev", ascending=False)
        top_pm_rows = polymarket_alpha.loc[polymarket_alpha["alpha_gap_cents"].notna()].sort_values(
            "alpha_gap_cents", ascending=False
        ) if not polymarket_alpha.empty else pd.DataFrame()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model confidence", confidence["label"])
        c2.metric("Expected goals", f"{result['hxg']:.2f} - {result['axg']:.2f}")
        c3.metric(
            "Best local EV",
            "n/a" if top_alpha_rows.empty or pd.isna(top_alpha_rows.iloc[0]["alpha_ev"]) else f"{100 * top_alpha_rows.iloc[0]['alpha_ev']:.1f}%",
        )
        c4.metric(
            "Best Polymarket gap",
            "n/a" if top_pm_rows.empty else f"{float(top_pm_rows.iloc[0]['alpha_gap_cents']):.1f}c",
        )
        st.caption(
            "Alpha Read is a statistical screen from the model and loaded market data. "
            "It is not staking advice, trade execution, or an investment recommendation."
        )

        st.subheader("Data Support")
        if data_support.get("warnings"):
            for warning in data_support["warnings"]:
                st.warning(warning)
        d1, d2, d3, d4 = st.columns(4)
        d1.metric(f"{result['home'].lower()} matches", f"{data_support['home_team_match_count']:,}")
        d2.metric(f"{result['away'].lower()} matches", f"{data_support['away_team_match_count']:,}")
        d3.metric("H2H matches", f"{data_support['h2h_match_count']:,}")
        d4.metric("Training matches", f"{data_support['training_match_count']:,}")
        st.caption(
            f"{data_support['h2h_note']}. Training data range: "
            f"{data_support['training_data_start'] or 'n/a'} to {data_support['training_data_end'] or 'n/a'}."
        )

        st.subheader("Goal Distribution And Outcome")
        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(
                create_goal_distribution_chart(goal_distribution, result["home"], result["away"]),
                width="stretch",
            )
        with g2:
            st.plotly_chart(create_match_outcome_donut(probs, result["home"], result["away"]), width="stretch")

        st.subheader("League Context")
        st.dataframe(league_context_display(league_context), hide_index=True, width="stretch")
        context_source = league_context["source"].iloc[0] if "source" in league_context.columns and not league_context.empty else ""
        st.caption(f"League context source: {context_source}.")

        st.subheader("Score Timeline")
        st.plotly_chart(create_score_timeline_chart(score_timeline), width="stretch")

        st.subheader("Scoreline Matrix")
        st.plotly_chart(create_compact_score_matrix(result["score_matrix"], result["home"], result["away"], max_goal=4), width="stretch")
        with st.expander("Full scoreline matrix"):
            st.plotly_chart(
                create_compact_score_matrix(result["score_matrix"], result["home"], result["away"], max_goal=cfg.max_goals),
                width="stretch",
            )

        st.subheader("Expected Goals And Team Ratings")
        x1, x2 = st.columns(2)
        with x1:
            st.plotly_chart(create_expected_goals_chart(expected_goals_df), width="stretch")
            st.caption(expected_goals_df.attrs.get("note", ""))
        with x2:
            st.plotly_chart(create_team_ratings_chart(rating_percentiles), width="stretch")
            st.dataframe(
                rating_percentiles[
                    ["team", "attack", "defense", "attack_label", "defense_label"]
                ],
                hide_index=True,
                width="stretch",
            )

        st.subheader("Climate Factors And Model Information")
        m1, m2 = st.columns(2)
        with m1:
            st.dataframe(climate_factor_display(climate_factors), hide_index=True, width="stretch")
            if climate_factors.attrs.get("team_factor_note"):
                st.caption(climate_factors.attrs["team_factor_note"])
            if climate_factors.attrs.get("roof_note"):
                st.caption(climate_factors.attrs["roof_note"])
        with m2:
            source_summary = ", ".join(
                f"{row['input']}: {row['source']}" for _, row in source_status.iterrows() if row.get("input")
            )
            model_info = pd.DataFrame(
                [
                    {"Item": "Model version", "Value": "Match report v1 / transparent Poisson xG model"},
                    {"Item": "Primary model mode", "Value": current_model_policy["primary_model_mode"]},
                    {"Item": "Behavior status", "Value": current_model_policy["behavior_status"]},
                    {"Item": "Behavior default blend", "Value": f"{float(current_model_policy['behavior_default_blend']):.2f}"},
                    {"Item": "Strict validation summary", "Value": current_model_policy["reason"]},
                    {"Item": "Last validation report", "Value": current_model_policy["last_validated_report"]},
                    {"Item": "Training data match count", "Value": f"{data_support['training_match_count']:,}"},
                    {
                        "Item": "Training data date range",
                        "Value": f"{data_support['training_data_start'] or 'n/a'} to {data_support['training_data_end'] or 'n/a'}",
                    },
                    {"Item": "RPS", "Value": "RPS placeholder / not yet backtested"},
                    {"Item": "Data source status", "Value": source_summary},
                ]
            )
            st.dataframe(model_info, hide_index=True, width="stretch")

        st.subheader("Market Value Tables")
        markets_loaded_count = len(resolved_event_markets)
        markets_joined_count = int(joined_polymarket_markets["market_price_cents"].notna().sum()) if not joined_polymarket_markets.empty else 0
        if slug_resolution.get("resolution_status") != "resolved":
            st.warning("No Polymarket event resolved for this fixture.")
        elif markets_loaded_count == 0:
            st.warning("Polymarket event resolved, but no nested market prices were loaded.")
        elif markets_joined_count == 0:
            st.warning("Polymarket event resolved, but no matching market prices were found for the local market groups.")
        for group_name, group_df in market_groups.items():
            st.markdown(f"**{group_name}**")
            if group_df.empty:
                if slug_resolution.get("resolution_status") != "resolved":
                    st.caption("No Polymarket event resolved for this fixture.")
                else:
                    st.caption("Polymarket event resolved, but no matching market prices were found for this market group.")
            else:
                st.dataframe(group_df, hide_index=True, width="stretch")
                if "Odds / Price" in group_df.columns and not group_df["Odds / Price"].astype(str).str.strip().any():
                    st.caption("Polymarket event resolved, but no matching market prices were found for this market group.")

    with tabs[2]:
        mat = result["score_matrix"].copy()
        pivot = mat.pivot(index="home_goals", columns="away_goals", values="prob") * 100
        fig = px.imshow(
            pivot,
            text_auto=".1f",
            labels={"x": f"{result['away']} goals", "y": f"{result['home']} goals", "color": "Probability %"},
            aspect="auto",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig, width="stretch")

        scores = result["top_scores"].copy()
        scores["probability"] = scores["prob"].map(pct)
        st.dataframe(scores[["label", "probability"]], hide_index=True, width="stretch")

    with tabs[3]:
        st.subheader("1X2, Totals, BTTS, Handicap, and Market Alpha")
        market_cols = [
            "market",
            "selection",
            "model_prob",
            "primary_model_probability",
            "behavior_diagnostic_probability",
            "behavior_probability_delta",
            "model_policy",
            "edge_source",
            "fair_odds",
            "market_odds",
            "alpha_ev",
            "odds_source",
            "odds_last_updated",
        ]
        st.dataframe(
            alpha_display(alpha)[[col for col in market_cols if col in alpha.columns]],
            hide_index=True,
            width="stretch",
        )
        st.caption("Market alpha uses the primary model probability. Behavior-only differences are diagnostic context, not primary alpha.")

    with tabs[4]:
        st.subheader("Candidate Matched Markets")
        pm_source = mapping_market_pool.attrs.get("source_label", "unknown source")
        pm_warning = mapping_market_pool.attrs.get("warning", "")
        match_search_note = ""
        if search_selected_match:
            match_search_note = f" | Selected-match search rows: {len(match_polymarket_markets)}"
        st.caption(
            f"Mapping status: {mapping_status(mapped_markets)} | "
            f"Candidate Polymarket markets: {len(mapping_market_pool)} from {pm_source} | "
            f"Sidebar rows: {len(polymarket_markets)}{match_search_note}"
        )
        slug_summary = {
            "slug_resolution_status": slug_resolution.get("resolution_status", ""),
            "resolved_slug": slug_resolution.get("resolved_slug", ""),
            "confidence": slug_resolution.get("confidence", ""),
            "matched_home": slug_resolution.get("matched_home", False),
            "matched_away": slug_resolution.get("matched_away", False),
            "matched_date": slug_resolution.get("matched_date", False),
            "team_order": slug_resolution.get("team_order", ""),
            "markets_loaded_count": len(resolved_event_markets),
            "markets_joined_count": int(joined_polymarket_markets["market_price_cents"].notna().sum()) if not joined_polymarket_markets.empty else 0,
            "source": slug_resolution.get("source", ""),
            "warning": slug_resolution.get("warning", ""),
        }
        st.write("Polymarket sports slug resolution")
        st.dataframe(pd.DataFrame([slug_summary]), hide_index=True, width="stretch")
        if slug_candidates:
            with st.expander("Slug candidates tried", expanded=False):
                st.dataframe(pd.DataFrame({"slug_candidate": slug_candidates}), hide_index=True, width="stretch")
        if pm_warning:
            st.warning(pm_warning)
        st.write("Polymarket market search diagnostics")
        st.caption(
            "The app searches by the full matchup, not by country alone. This prevents unrelated political markets "
            "from being treated as football match markets."
        )
        discovery_cache_status = polymarket_discovery_cache_status()
        discovery_summary = {
            "retrieval_layers_tried": ", ".join(match_polymarket_markets.attrs.get("retrieval_layers_tried", []))
            if not match_polymarket_markets.empty
            else "",
            "events_fetched": match_polymarket_markets.attrs.get("events_fetched", 0),
            "nested_markets_scanned": match_polymarket_markets.attrs.get("markets_flattened", 0),
            "cache_markets_rows": discovery_cache_status.get("markets_cache_rows", 0),
            "cache_last_updated": discovery_cache_status.get("markets_cache_last_updated", ""),
            "no_match_explanation": ""
            if not mapped_markets.empty
            else "No high-confidence match market was mapped. The market may not exist yet, may be closed, may use different team names, or the API/cache may be stale.",
        }
        st.dataframe(pd.DataFrame([discovery_summary]), hide_index=True, width="stretch")
        if polymarket_event_url_or_slug.strip():
            st.write("Polymarket URL / slug resolver")
            st.caption(
                "If automatic search cannot find a match, paste the Polymarket sports event URL here. "
                "The app parses the event slug, checks it against the selected fixture, and extracts nested markets. "
                "Resolved mappings are still unconfirmed by default."
            )
            parsed_url = url_resolver_diagnostics.get("parsed", parse_polymarket_url_or_slug(polymarket_event_url_or_slug.strip()))
            url_summary = {
                "input_slug": parsed_url.get("slug", ""),
                "parsed_team_codes": ", ".join(
                    part for part in [parsed_url.get("team_code_1", ""), parsed_url.get("team_code_2", "")] if part
                ),
                "canonical_slug_teams": ", ".join(
                    part for part in [url_slug_score.get("slug_team_1", ""), url_slug_score.get("slug_team_2", "")] if part
                ),
                "matched_home": url_slug_score.get("matched_home", False),
                "matched_away": url_slug_score.get("matched_away", False),
                "matched_date": url_slug_score.get("matched_date", False),
                "mapping_confidence": url_slug_score.get("confidence", ""),
                "event_title": url_resolver_diagnostics.get("event_title", ""),
                "markets_extracted": len(url_resolved_markets),
                "market_types_found": ", ".join(url_resolver_diagnostics.get("market_types_found", [])),
                "warning": "; ".join(
                    part
                    for part in [
                        parsed_url.get("warning", ""),
                        url_resolver_diagnostics.get("warning", ""),
                        url_slug_score.get("warning", ""),
                    ]
                    if part
                ),
            }
            st.dataframe(pd.DataFrame([url_summary]), hide_index=True, width="stretch")
            if not url_resolved_markets.empty:
                url_cols = [
                    "market_id",
                    "question",
                    "market_type",
                    "yes_price",
                    "no_price",
                    "liquidity",
                    "volume",
                    "extraction_method",
                ]
                st.dataframe(
                    url_resolved_markets[[col for col in url_cols if col in url_resolved_markets.columns]],
                    hide_index=True,
                    width="stretch",
                )
                st.caption("To manually confirm any row, add its market_id and side mapping to data/market_mappings.csv.")
        if polymarket_search_queries:
            st.dataframe(pd.DataFrame({"search_queries_used": polymarket_search_queries}), hide_index=True, width="stretch")
        if not polymarket_search_diagnostics.empty:
            best_candidate = polymarket_search_diagnostics.loc[
                ~polymarket_search_diagnostics["rejected"].astype(bool)
            ].head(1)
            if not best_candidate.empty:
                st.write("Best match-search candidate")
                st.dataframe(
                    best_candidate[
                        [
                            "market_id",
                            "question",
                            "score",
                            "confidence",
                            "matched_home",
                            "matched_away",
                            "matched_sports_terms",
                            "score_breakdown",
                        ]
                    ],
                    hide_index=True,
                    width="stretch",
                )
            accepted_search = polymarket_search_diagnostics.loc[
                ~polymarket_search_diagnostics["rejected"].astype(bool)
            ].head(10)
            rejected_search = polymarket_search_diagnostics.loc[
                polymarket_search_diagnostics["rejected"].astype(bool)
            ].head(10)
            diag_cols = [
                "market_id",
                "question",
                "event_title",
                "category",
                "score",
                "confidence",
                "reject_reason",
                "score_breakdown",
            ]
            if not accepted_search.empty:
                st.write("Accepted search candidates")
                st.dataframe(accepted_search[[col for col in diag_cols if col in accepted_search.columns]], hide_index=True, width="stretch")
            if not rejected_search.empty:
                st.write("Rejected search candidates")
                st.dataframe(rejected_search[[col for col in diag_cols if col in rejected_search.columns]], hide_index=True, width="stretch")
        if mapping_market_pool.empty:
            st.warning(
                "No Polymarket market data is loaded, so there is nothing to map for this match. "
                "Click **Update Polymarket markets** to refresh the default Gamma API, broaden the "
                "sidebar market search query, or add fallback rows to `data/polymarket_markets.csv`."
            )
            st.caption(
                "Required CSV columns: market_id, question, slug, event_title, category, start_date, "
                "end_date, active, closed, outcomes, yes_price, no_price, liquidity, volume, source, last_updated."
            )
        elif mapped_markets.empty:
            mostly_outrights = (
                not unmapped_diagnostics.empty
                and unmapped_diagnostics["rejection_reason"]
                .astype(str)
                .str.contains("Tournament outright", case=False, na=False)
                .all()
            )
            search_detail = (
                " The app also searched Gamma by the selected home and away team names."
                if search_selected_match
                else " Turn on **Search selected match markets** to query Gamma by the selected teams."
            )
            reason_detail = (
                " Gamma only returned tournament-winner markets for these teams."
                if mostly_outrights
                else ""
            )
            st.warning(
                f"Checked {len(mapping_market_pool)} Polymarket market row(s), but none matched "
                f"{result['home']} vs {result['away']}.{search_detail}{reason_detail} Add a confirmed row to "
                "`data/market_mappings.csv` or adjust the Polymarket market data/query. Turn off "
                "**Show only mapped markets** to inspect the loaded Gamma rows."
            )
            if not unmapped_diagnostics.empty:
                diagnostic_display = unmapped_diagnostics.head(25).copy()
                for col in ["yes_price", "no_price"]:
                    if col in diagnostic_display.columns:
                        diagnostic_display[col] = diagnostic_display[col].map(
                            lambda x: "" if pd.isna(x) else f"{100*float(x):.1f}"
                        )
                st.dataframe(
                    diagnostic_display[
                        [
                            "market_id",
                            "question",
                            "event_title",
                            "has_home",
                            "has_away",
                            "market_type_guess",
                            "rejection_reason",
                            "yes_price",
                            "no_price",
                            "liquidity",
                            "volume",
                        ]
                    ],
                    hide_index=True,
                    width="stretch",
                )
        else:
            candidate_display = mapped_markets.copy()
            for col in ["yes_price", "no_price"]:
                if col in candidate_display.columns:
                    candidate_display[col] = candidate_display[col].map(lambda x: "" if pd.isna(x) else f"{100*float(x):.1f}")
            st.dataframe(
                candidate_display[
                    [
                        "market_id",
                        "question",
                        "market_type",
                        "model_side",
                        "polymarket_side",
                        "mapping_confidence",
                        "mapping_score",
                        "mapping_reason",
                        "reject_reason",
                        "manual_confirmed",
                        "yes_price",
                        "no_price",
                        "liquidity",
                        "volume",
                    ]
                ],
                hide_index=True,
                width="stretch",
            )

        st.subheader("Polymarket Alpha")
        if joined_polymarket_alpha_rows.empty:
            if slug_resolution.get("resolution_status") != "resolved":
                reason_no_rows = "No Polymarket event resolved for this fixture."
            elif resolved_event_markets.empty:
                reason_no_rows = "Polymarket event resolved, but no nested market prices were loaded."
            else:
                reason_no_rows = "Polymarket event resolved, but no matching market prices were found for local model markets."
            st.info(reason_no_rows)
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "slug_resolution_status": slug_resolution.get("resolution_status", ""),
                            "resolved_slug": slug_resolution.get("resolved_slug", ""),
                            "candidates_tried": ", ".join(slug_resolution.get("candidates_tried", [])),
                            "markets_loaded_count": len(resolved_event_markets),
                            "markets_joined_count": int(joined_polymarket_markets["market_price_cents"].notna().sum()) if not joined_polymarket_markets.empty else 0,
                            "reason_no_rows": reason_no_rows,
                        }
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        else:
            st.dataframe(
                joined_polymarket_alpha_rows[
                    [
                        "market",
                        "model_probability",
                        "fair_price_cents",
                        "market_price_cents",
                        "alpha_gap_cents",
                        "score",
                        "signal",
                        "polymarket_question",
                        "mapping_confidence",
                        "event_slug",
                    ]
                ],
                hide_index=True,
                width="stretch",
            )
            st.caption("Model fair price and alpha gap are shown in cents. This is a statistical screen, not betting advice.")

        if not show_only_mapped:
            st.subheader("Loaded Polymarket Markets")
            st.dataframe(
                mapping_market_pool.head(50)[
                    [
                        "market_id",
                        "question",
                        "slug",
                        "event_title",
                        "yes_price",
                        "no_price",
                        "liquidity",
                        "volume",
                        "source",
                        "last_updated",
                    ]
                ],
                hide_index=True,
                width="stretch",
            )

        if st.button("Save prediction snapshot", key=f"save_snapshot_{match['match_id']}"):
            save_prediction_snapshot(match, result, polymarket_alpha)
            st.success(f"Saved {len(polymarket_alpha)} prediction row(s) to data/prediction_log.csv.")

    with tabs[5]:
        st.subheader("Sensitivity Analysis")
        sens_display = sensitivity_df.copy()
        for col in ["home_win", "draw", "away_win", "over_2_5", "under_2_5", "btts_yes", "btts_no"]:
            sens_display[col] = sens_display[col].map(lambda x: "" if pd.isna(x) else f"{100*x:.1f}%")
        for col in ["home_xg", "away_xg"]:
            sens_display[col] = sens_display[col].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
        st.dataframe(sens_display, hide_index=True, width="stretch")

        st.subheader("Alpha Robustness")
        if polymarket_alpha.empty:
            st.info("No Polymarket alpha rows to test for robustness.")
        else:
            robustness_display = polymarket_alpha[
                [
                    "market_id",
                    "question",
                    "market_type",
                    "polymarket_side",
                    "alpha_gap_cents",
                    "signal_strength",
                    "robustness",
                    "supporting_scenarios",
                ]
            ].copy()
            robustness_display["alpha_gap_cents"] = robustness_display["alpha_gap_cents"].map(
                lambda x: "" if pd.isna(x) else f"{x:.1f}"
            )
            st.dataframe(robustness_display, hide_index=True, width="stretch")
            st.caption("A signal is robust when the same Polymarket side remains positive alpha in at least 2 of 3 scenarios.")

    with tabs[6]:
        st.subheader("Backtesting")
        st.caption(
            "Backtesting and QA can be slow, so these diagnostics run only when requested. "
            "This keeps match selection responsive and avoids parsing local ledgers unless needed."
        )
        bt_key = str(match.get("match_id", label)).replace(" ", "_")

        if st.checkbox("Load saved prediction-log evaluation", value=False, key=f"load_prediction_log_{bt_key}"):
            prediction_log = load_prediction_log()
            results_log = load_results_log()
            if prediction_log.attrs.get("warning"):
                st.warning(prediction_log.attrs["warning"])
                st.caption("Run `.venv/bin/python scripts/repair_prediction_log.py` to quarantine and rebuild the malformed log.")
            evaluation = evaluate_predictions(prediction_log, results_log)
            c1, c2, c3 = st.columns(3)
            c1.metric("Saved predictions", f"{len(prediction_log):,}")
            if not evaluation.empty:
                eval_row = evaluation.iloc[0]
                c2.metric("Brier score", "" if pd.isna(eval_row["brier_score"]) else f"{eval_row['brier_score']:.4f}")
                c3.metric("Log loss", "" if pd.isna(eval_row["log_loss"]) else f"{eval_row['log_loss']:.4f}")
                st.dataframe(evaluation, hide_index=True, width="stretch")
            if results_log.empty:
                st.info("Add completed match rows to `data/results_log.csv` to evaluate saved predictions.")
        else:
            st.info("Saved prediction-log evaluation is idle. Enable it here if you want to inspect legacy prediction snapshots.")

        st.divider()
        st.subheader("Behavior-Adjusted Backtest")
        st.caption(
            "Lower Brier score and log loss are better. Higher probability assigned to the actual result is better. "
            "Run this only when you want to evaluate completed-match predictions."
        )
        bt_default_end = utc_today
        bt_default_start = bt_default_end - timedelta(days=10)
        c_start, c_end, c_comp = st.columns([1, 1, 2])
        with c_start:
            bt_start = st.date_input("Backtest start", value=bt_default_start, key=f"bt_start_{bt_key}")
        with c_end:
            bt_end = st.date_input("Backtest end", value=bt_default_end, key=f"bt_end_{bt_key}")
        with c_comp:
            bt_competition = st.text_input("Competition filter", value="World Cup", key=f"bt_comp_{bt_key}")
        bt_start, bt_end = sorted((bt_start, bt_end))

        run_behavior_backtest = st.checkbox("Run behavior-adjusted backtest", value=False, key=f"run_behavior_backtest_{bt_key}")
        bt_matches = pd.DataFrame()
        if run_behavior_backtest:
            backtest_result = load_behavior_backtest_results(
                bt_start.isoformat(),
                bt_end.isoformat(),
                bt_competition,
                st.session_state["refresh_counter"],
            )
            bt_matches = backtest_result.get("matches", pd.DataFrame())
            bt_metrics = backtest_result.get("metrics", pd.DataFrame())
            bt_comparison = backtest_result.get("comparison", pd.DataFrame())
            bt_calibration = backtest_result.get("calibration", pd.DataFrame())
            bt_qa_summary = backtest_result.get("qa_summary", {})
            bt_qa_matches = backtest_result.get("qa_match_inputs", pd.DataFrame())
            bt_warning = str(backtest_result.get("warning", ""))

            if bt_warning:
                st.warning(bt_warning)
            if isinstance(bt_qa_summary, dict) and bt_qa_summary.get("qa_warning"):
                st.warning(bt_qa_summary["qa_warning"])
            st.metric("Completed matches in backtest", f"{len(bt_matches):,}")
            if bt_metrics.empty:
                st.info("No completed matches found for this backtest filter. Widen the date range or clear the competition filter.")
            else:
                st.write("Summary metrics")
                st.dataframe(bt_metrics, hide_index=True, width="stretch")

                st.write("Per-match comparison")
                comparison_cols = [
                    "date_utc",
                    "home",
                    "away",
                    "score",
                    "actual_result",
                    "baseline_actual_prob",
                    "behavior_actual_prob",
                    "actual_prob_delta",
                    "improved_by_behavior",
                    "notes",
                ]
                st.dataframe(
                    bt_comparison[[col for col in comparison_cols if col in bt_comparison.columns]],
                    hide_index=True,
                    width="stretch",
                )

                st.write("Calibration buckets")
                st.dataframe(bt_calibration, hide_index=True, width="stretch")

                if isinstance(bt_qa_summary, dict) and bt_qa_summary:
                    st.write("Backtest QA summary")
                    st.dataframe(pd.DataFrame([bt_qa_summary]), hide_index=True, width="stretch")
                if isinstance(bt_qa_matches, pd.DataFrame) and not bt_qa_matches.empty:
                    qa_cols = [
                        "date_utc",
                        "home",
                        "away",
                        "score",
                        "baseline_home_prob",
                        "baseline_draw_prob",
                        "baseline_away_prob",
                        "behavior_home_prob",
                        "behavior_draw_prob",
                        "behavior_away_prob",
                        "baseline_actual_prob",
                        "behavior_actual_prob",
                        "rating_coverage_warning",
                        "probability_warning",
                    ]
                    st.dataframe(bt_qa_matches[[col for col in qa_cols if col in bt_qa_matches.columns]], hide_index=True, width="stretch")
        else:
            st.info("Behavior-adjusted backtest is idle. Enable it to run the completed-match evaluation.")

        if st.checkbox("Run rating/external benchmark diagnostics", value=False, key=f"run_rating_diagnostics_{bt_key}"):
            st.write("Rating Coverage")
            rating_required = collect_required_teams(fixtures, historical_long_matches, team_behavior, bt_matches)
            manual_rating_rows = load_team_ratings_csv()
            rating_aliases = load_team_name_aliases_df(include_defaults=False)
            rating_audit = audit_rating_coverage(rating_required, manual_rating_rows, rating_aliases)
            rating_alias_proposals = propose_team_aliases(rating_required, manual_rating_rows, rating_aliases)
            rating_summary = rating_coverage_summary(rating_audit, rating_alias_proposals)
            st.dataframe(pd.DataFrame([rating_summary]), hide_index=True, width="stretch")
            if not rating_audit.empty:
                st.dataframe(rating_audit.loc[rating_audit["warning"].astype(str) != ""], hide_index=True, width="stretch")

            st.write("Rating Review")
            review_audit = audit_generated_ratings(manual_rating_rows, team_behavior, historical_long_matches, bt_matches)
            st.dataframe(pd.DataFrame([rating_review_summary(review_audit)]), hide_index=True, width="stretch")

            st.write("External Prior / Benchmark Review")
            external_priors = load_external_priors()
            review_proposals = load_review_proposals()
            manual_review_sheet = load_manual_rating_review_sheet()
            prior_comparison = compare_ratings_to_external_priors(manual_rating_rows, review_proposals, external_priors)
            st.dataframe(pd.DataFrame([external_prior_review_summary(prior_comparison, manual_review_sheet)]), hide_index=True, width="stretch")
            benchmark_coverage = audit_external_benchmark_coverage(rating_required, external_priors, manual_rating_rows)
            st.dataframe(pd.DataFrame([external_benchmark_coverage_summary(benchmark_coverage)]), hide_index=True, width="stretch")
            calibration_proposals = load_external_calibration_proposals()
            if calibration_proposals.empty:
                calibration_proposals = build_external_calibration_proposals(manual_rating_rows, team_behavior, external_priors)
            st.dataframe(pd.DataFrame([external_benchmark_calibration_summary(calibration_proposals, manual_rating_rows)]), hide_index=True, width="stretch")

            st.write("FIFA Snapshot / Ranking Import Status")
            st.dataframe(pd.DataFrame([fifa_snapshot_validation_dashboard_status(), fifa_ranking_import_dashboard_status()]), hide_index=True, width="stretch")

        if st.checkbox("Run strict as-of-date backtest", value=False, key=f"run_asof_backtest_{bt_key}"):
            st.write("Strict As-Of-Date Backtest")
            asof_result = load_asof_backtest_results(
                bt_start.isoformat(),
                bt_end.isoformat(),
                bt_competition,
                st.session_state["refresh_counter"],
            )
            asof_metrics = asof_result.get("metrics", pd.DataFrame())
            asof_predictions = asof_result.get("predictions", pd.DataFrame())
            asof_comparison = asof_result.get("comparison", pd.DataFrame())
            asof_warning = str(asof_result.get("warning", ""))
            if asof_warning:
                st.warning(asof_warning)
            if isinstance(asof_metrics, pd.DataFrame) and not asof_metrics.empty:
                st.dataframe(asof_metrics, hide_index=True, width="stretch")
            if isinstance(asof_comparison, pd.DataFrame) and not asof_comparison.empty:
                st.dataframe(asof_comparison, hide_index=True, width="stretch")
            if isinstance(asof_predictions, pd.DataFrame) and not asof_predictions.empty:
                diag_cols = [
                    "date_utc",
                    "home",
                    "away",
                    "mode",
                    "home_latest_behavior_match",
                    "away_latest_behavior_match",
                    "home_matches_used_recent",
                    "away_matches_used_recent",
                    "lookahead_safe",
                    "warning",
                ]
                st.dataframe(asof_predictions[[col for col in diag_cols if col in asof_predictions.columns]], hide_index=True, width="stretch")

        if st.checkbox("Run strict behavior blend sensitivity", value=False, key=f"blend_sensitivity_{bt_key}"):
            blend_result = load_blend_sensitivity_results(
                bt_start.isoformat(),
                bt_end.isoformat(),
                bt_competition,
                st.session_state["refresh_counter"],
            )
            blend_metrics = blend_result.get("metrics", pd.DataFrame())
            blend_team = blend_result.get("team_sensitivity", pd.DataFrame())
            blend_recommendation = str(blend_result.get("recommendation", ""))
            if blend_recommendation:
                st.info(blend_recommendation)
            if isinstance(blend_metrics, pd.DataFrame) and not blend_metrics.empty:
                st.write("Metrics by blend multiplier")
                st.dataframe(blend_metrics, hide_index=True, width="stretch")
            if isinstance(blend_team, pd.DataFrame) and not blend_team.empty:
                st.write("Team-level helped/hurt")
                st.dataframe(blend_team, hide_index=True, width="stretch")

    with tabs[7]:
        st.subheader("Team Inputs")
        team_inputs = pd.DataFrame([result["home_inputs"], result["away_inputs"]])
        for col in [
            "manual_attack",
            "manual_defense",
            "manual_recent_form",
            "attack_source",
            "defense_source",
            "recent_form_source",
            "behavior_blend_used",
        ]:
            if col not in team_inputs.columns:
                team_inputs[col] = pd.NA
        st.dataframe(
            team_inputs[
                [
                    "team",
                    "elo",
                    "attack",
                    "defense",
                    "recent_form",
                    "fifa_rank_proxy",
                    "training_temp_c",
                    "training_humidity_pct",
                    "manual_attack",
                    "manual_defense",
                    "manual_recent_form",
                    "attack_source",
                    "defense_source",
                    "recent_form_source",
                    "behavior_blend_used",
                    "data_quality",
                    "last_updated",
                    "notes",
                ]
            ],
            hide_index=True,
            width="stretch",
        )

        st.subheader("Training Climate")
        climates = pd.DataFrame(
            [
                {"team": result["home"], **get_team_training_climate(result["home"])},
                {"team": result["away"], **get_team_training_climate(result["away"])},
            ]
        )
        st.dataframe(climates, hide_index=True, width="stretch")

    with tabs[8]:
        st.subheader("Venue And Environment")
        env = venue_env
        env_df = pd.DataFrame([env])
        st.dataframe(
            env_df[
                [
                    "venue",
                    "city",
                    "country",
                    "altitude_m",
                    "temp_c",
                    "humidity_pct",
                    "wind_kmh",
                    "precipitation_mm",
                    "roof_expected_closed",
                    "effective_temp_c",
                    "effective_humidity_pct",
                    "effective_wind_kmh",
                    "effective_precipitation_mm",
                    "source_label",
                    "environment_source",
                    "last_updated",
                ]
            ],
            hide_index=True,
            width="stretch",
        )

        comp = result["components"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Home env log adj", f"{comp['home_environment_log_adj']:.3f}")
        c2.metric("Away env log adj", f"{comp['away_environment_log_adj']:.3f}")
        c3.metric("Total-goals env adj", f"{comp['total_environment_log_adj']:.3f}")
        st.write(comp["environment_note"])
        st.caption(
            "Roof-closed venues dampen outdoor weather by 85%. Altitude mainly matters above about 1,200 m. "
            "Wind and precipitation mostly reduce total-goals quality, while heat/humidity mismatch affects each team slightly."
        )

    with tabs[9]:
        if historical_long_matches.empty:
            st.info(
                "No historical matches loaded yet. Run:\n\n"
                "`.venv/bin/python scripts/import_historical_csv.py --input path/to/results.csv "
                "--source public_csv --rebuild-behavior`"
            )
        else:
            h1, h2, h3 = st.columns(3)
            h1.metric("Historical rows", f"{len(historical_long_matches):,}")
            selected_overview = historical_behavior_overview(
                historical_long_matches,
                team_behavior,
                [result["home"], result["away"]],
            )
            h2.metric(f"{result['home']} rows", f"{int(selected_overview.iloc[0]['historical_rows']):,}")
            h3.metric(f"{result['away']} rows", f"{int(selected_overview.iloc[1]['historical_rows']):,}")
            st.dataframe(selected_overview, hide_index=True, width="stretch")

        st.subheader("Recent Window Summary")
        window_summary = behavior_window_display(team_behavior, [result["home"], result["away"]])
        if window_summary.empty:
            st.caption("No calibrated behavior window rows are available.")
        else:
            st.dataframe(window_summary, hide_index=True, width="stretch")
            for warning_text in window_summary.get("warnings", pd.Series(dtype="object")).dropna().astype(str):
                if warning_text:
                    st.warning(warning_text)

        st.subheader("Recent vs All-Time")
        all_time_compare = recent_vs_all_time_display(team_behavior, [result["home"], result["away"]])
        if all_time_compare.empty:
            st.caption("No all-time diagnostic behavior rows are available.")
        else:
            st.dataframe(all_time_compare, hide_index=True, width="stretch")

        st.subheader("Schedule Strength")
        st.caption("This section checks whether recent good results came against strong or weak opponents.")
        schedule_display = schedule_strength_display(team_behavior, [result["home"], result["away"]])
        if schedule_display.empty:
            st.caption("No opponent-Elo schedule diagnostics are available.")
        else:
            st.dataframe(schedule_display, hide_index=True, width="stretch")
            for warning_text in schedule_display.get("warnings", pd.Series(dtype="object")).dropna().astype(str):
                if warning_text:
                    st.warning(warning_text)

        st.subheader("Expected Performance vs Actual Performance")
        st.caption("This checks whether a team performed better or worse than expected given the strength of its opponents.")
        residual_display = expected_performance_display(team_behavior, [result["home"], result["away"]])
        if residual_display.empty:
            st.caption("No residual-based performance diagnostics are available.")
        else:
            st.dataframe(residual_display, hide_index=True, width="stretch")
            for warning_text in residual_display.get("warnings", pd.Series(dtype="object")).dropna().astype(str):
                if warning_text:
                    st.warning(warning_text)

        st.subheader("Behavior Drivers")
        st.caption(
            "This section explains whether a team's high score comes from consistent performance, weak opponents, "
            "friendlies, or a few large wins."
        )
        run_behavior_drivers = st.checkbox(
            "Run behavior driver diagnostics",
            value=False,
            key=f"run_behavior_drivers_{str(match.get('match_id', label)).replace(' ', '_')}",
        )
        if run_behavior_drivers:
            for selected_team in [result["home"], result["away"]]:
                with st.expander(f"{selected_team} behavior drivers", expanded=False):
                    d1, d2 = st.columns(2)
                    with d1:
                        st.markdown("**Top attack driver matches**")
                        attack_drivers = behavior_driver_matches_display(
                            historical_long_matches,
                            selected_team,
                            match.get("date_utc"),
                            sort_by="attack",
                            limit=10,
                        )
                        if attack_drivers.empty:
                            st.caption("No attack driver rows are available.")
                        else:
                            st.dataframe(attack_drivers, hide_index=True, width="stretch")
                    with d2:
                        st.markdown("**Top defense driver matches**")
                        defense_drivers = behavior_driver_matches_display(
                            historical_long_matches,
                            selected_team,
                            match.get("date_utc"),
                            sort_by="defense",
                            limit=10,
                        )
                        if defense_drivers.empty:
                            st.caption("No defense driver rows are available.")
                        else:
                            st.dataframe(defense_drivers, hide_index=True, width="stretch")

                    b1, b2 = st.columns(2)
                    with b1:
                        st.markdown("**Opponent tier breakdown**")
                        opponent_breakdown = behavior_breakdown_display(
                            calculate_opponent_tier_breakdown(historical_long_matches, selected_team, match.get("date_utc"))
                        )
                        if opponent_breakdown.empty:
                            st.caption("No opponent-tier breakdown is available.")
                        else:
                            st.dataframe(opponent_breakdown, hide_index=True, width="stretch")
                    with b2:
                        st.markdown("**Competition breakdown**")
                        competition_breakdown = behavior_breakdown_display(
                            calculate_competition_breakdown(historical_long_matches, selected_team, match.get("date_utc"))
                        )
                        if competition_breakdown.empty:
                            st.caption("No competition breakdown is available.")
                        else:
                            st.dataframe(competition_breakdown, hide_index=True, width="stretch")
        else:
            st.info("Behavior driver diagnostics are idle. Enable them to scan historical match drivers and breakdowns.")

        st.subheader("Final Model Input Impact")
        input_audit = final_model_input_audit_display(teams, team_behavior, [result["home"], result["away"]])
        if input_audit.empty:
            st.caption("No final model input audit rows are available.")
        else:
            st.dataframe(input_audit, hide_index=True, width="stretch")
            for warning_text in input_audit.get("warning", pd.Series(dtype="object")).dropna().astype(str):
                if warning_text:
                    st.warning(warning_text)

        st.subheader("Team Behavior Summary")
        st.caption(
            "Historical behavior is a recency-weighted descriptive layer, not causal proof. "
            "Friendlies, opponent quality, and missing event data can materially affect these metrics."
        )
        behavior_summary = selected_behavior_rows(team_behavior, result["home"], result["away"])
        if behavior_summary.empty:
            st.info(
                "No team behavior rows are available for this selected match. Add rows to "
                "`data/historical_matches.csv` or run the historical ingestion command above, then rebuild "
                "`data/team_behavior.csv`."
            )
        else:
            st.dataframe(behavior_summary, hide_index=True, width="stretch")

        st.subheader("Recency-Weighted Match History")
        h1, h2 = st.columns(2)
        with h1:
            st.markdown(f"**{result['home']}**")
            home_history = historical_match_history_display(historical_long_matches, result["home"], match.get("date_utc"), limit=20)
            if home_history.empty:
                st.caption("No long-format historical rows available.")
            else:
                st.dataframe(home_history, hide_index=True, width="stretch")
        with h2:
            st.markdown(f"**{result['away']}**")
            away_history = historical_match_history_display(historical_long_matches, result["away"], match.get("date_utc"), limit=20)
            if away_history.empty:
                st.caption("No long-format historical rows available.")
            else:
                st.dataframe(away_history, hide_index=True, width="stretch")

        st.subheader("Environment Response")
        env_response = environment_response_display(historical_long_matches, [result["home"], result["away"]], match.get("date_utc"))
        st.dataframe(env_response, hide_index=True, width="stretch")
        st.caption(
            "Environmental response uses hot >= 28 C, humid >= 70%, altitude >= 1000 m, "
            "wind >= 20 km/h, and rain > 0 mm. Effects are capped and conservative."
        )
        for warning_text in env_response.get("environment_warning", pd.Series(dtype="object")).dropna().astype(str):
            if warning_text:
                st.warning(warning_text)

        st.subheader("Model Impact")
        impact_display = model_impact_display(result)
        st.dataframe(impact_display, hide_index=True, width="stretch")
        if not impact_display.empty and impact_display["cap_hit"].any():
            st.warning("One or more behavior adjustments reached the configured movement cap; manual ratings remain the base input.")
        st.caption(
            "Current default policy: primary probabilities use baseline external-calibrated ratings with behavior blend 0.00. "
            "The behavior-adjusted as-of layer remains available here as a diagnostic, with the historical blend rules and caps shown for transparency. "
            "The model remains a statistical alpha screen, not betting advice."
        )

    with tabs[10]:
        st.markdown(
            """
            **Assumptions**

            - Expected goals combine Elo, attack, opponent defense, recent form, and conservative environmental adjustments.
            - The primary displayed model uses baseline external-calibrated ratings.
            - Historical behavior is diagnostic-only by default because strict as-of-date blend sensitivity did not improve both Brier score and log loss.
            - Scorelines come from an independent Poisson goal model and are normalized over the displayed score grid.
            - Fair odds are calculated as `1 / probability`.
            - Alpha EV is calculated as `model_probability x market_decimal_odds - 1`.
            - Manual CSV inputs remain valid overrides. Optional APIs only refresh cache files when configured.

            **Limitations**

            - This is not a black-box machine learning model and does not account for every lineup, tactical, injury, or motivation factor.
            - Missing market odds leave alpha EV blank.
            - Weather and training climates are approximate when API or recent match data are unavailable.
            - Historical behavior is descriptive, not causal proof; friendlies may not reflect full competitive strength and behavior-only edges are not primary alpha.
            - Opponent quality matters, and environmental response requires enough previous matches to be meaningful.
            - Environmental effects are capped and conservative.
            - Outputs are statistical estimates only and are not staking, bet sizing, or investment recommendations.
            """
        )

    with tabs[11]:
        st.subheader("Post-mortem Training")
        st.caption(
            "This section evaluates saved pre-match predictions after results are known. "
            "It supports evidence-based model improvement without tuning to one match or copying another model."
        )
        pm_key = str(match.get("match_id", label)).replace(" ", "_")
        if not st.checkbox("Load post-mortem diagnostics", value=False, key=f"load_postmortem_{pm_key}"):
            st.info("Post-mortem diagnostics are idle. Enable them to load ledgers, candidate reports, and rolling tournament learning state.")
            st.caption(
                "Training uses walk-forward validation. Historical senior national-team games before each prediction date are included with relevance weights. "
                "This is evaluation-only and does not provide staking, trade execution, Kelly sizing, or betting advice."
            )
        else:
            ledger = load_prediction_ledger()
            results_ledger = load_results_ledger()
            joined_pm = join_predictions_to_results(ledger, results_ledger)
            errors_pm = calculate_prediction_errors(joined_pm)
            report_pm = build_postmortem_report(errors_pm, joined_pm)

            pm1, pm2, pm3 = st.columns(3)
            pm1.metric("Prediction ledger rows", f"{len(ledger):,}")
            pm2.metric("Completed result rows", f"{len(results_ledger):,}")
            pm3.metric("Scored pre-kickoff predictions", f"{len(errors_pm):,}")

            st.divider()
            st.write("Rolling tournament learning")
            learning_ledger = load_tournament_learning_ledger()
            target_kickoff_for_learning = pd.to_datetime(match.get("date_utc"), errors="coerce", utc=True)
            if pd.isna(target_kickoff_for_learning):
                target_kickoff_for_learning = pd.Timestamp.utcnow()
            learning_used, learning_diag = filter_tournament_learning_asof(
                learning_ledger,
                target_kickoff_for_learning,
                target_match_id=str(match.get("match_id", "")),
            )
            l1, l2, l3, l4 = st.columns(4)
            l1.metric("Completed WC matches available", f"{learning_diag.get('completed_world_cup_matches_available', 0):,}")
            l2.metric("Eligible before selected kickoff", f"{learning_diag.get('completed_world_cup_matches_used', 0):,}")
            l3.metric("Latest match used", str(learning_diag.get("latest_completed_match_used", "") or "n/a"))
            l4.metric("Lookahead safe", "yes" if learning_diag.get("lookahead_safe", True) else "no")
            st.caption(
                "After each match finishes, the model can learn from that result for future games. "
                "It does not use a game's result to predict itself."
            )
            if not learning_used.empty:
                display_cols = ["date_utc", "home", "away", "actual_result", "eligible_after_utc", "source"]
                st.dataframe(
                    learning_used[[col for col in display_cols if col in learning_used.columns]].tail(20),
                    hide_index=True,
                    width="stretch",
                )

            if errors_pm.empty:
                st.info(
                    "No scored pre-kickoff predictions are available yet. Run "
                    "`scripts/snapshot_upcoming_predictions.py` before matches and "
                    "`scripts/import_completed_results.py` after results are known."
                )
            else:
                summary_pm = report_pm.get("summary_metrics", pd.DataFrame())
                st.write("Post-mortem metrics")
                st.dataframe(summary_pm, hide_index=True, width="stretch")
                c_best, c_worst = st.columns(2)
                with c_worst:
                    st.write("Worst misses")
                    st.dataframe(report_pm.get("worst_misses", pd.DataFrame()).head(10), hide_index=True, width="stretch")
                with c_best:
                    st.write("Best calls")
                    st.dataframe(report_pm.get("best_calls", pd.DataFrame()).head(10), hide_index=True, width="stretch")

            st.divider()
            st.write("Candidate model leaderboard")
            training_reports = sorted((Path(__file__).resolve().parent / "reports").glob("model_training_candidates_*.csv"))
            if training_reports:
                latest_training = training_reports[-1]
                try:
                    leaderboard = pd.read_csv(latest_training)
                except Exception:
                    leaderboard = pd.DataFrame()
                st.caption(f"Latest saved candidate report: `{latest_training.name}`")
                if leaderboard.empty:
                    st.info("Latest candidate report could not be read.")
                else:
                    st.dataframe(leaderboard.head(20), hide_index=True, width="stretch")
                    promoted = leaderboard.loc[leaderboard["promotion_status"].astype(str) == "promotion_candidate"] if "promotion_status" in leaderboard.columns else pd.DataFrame()
                    if promoted.empty:
                        st.info("No candidate is currently promoted; baseline external-calibrated remains the primary model.")
                    else:
                        st.success("A candidate met the promotion rule in the latest saved report. Review it before changing model policy.")
            else:
                st.info("No candidate training report exists yet. Run `scripts/train_model_candidates.py --save-report`.")

            st.caption(
                "Training uses walk-forward validation. Historical senior national-team games before each prediction date are included with relevance weights: "
                "recent qualifiers and tournament matches carry more weight; old friendlies carry less. "
                "This is evaluation-only and does not provide staking, trade execution, Kelly sizing, or betting advice."
            )
