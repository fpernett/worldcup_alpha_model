from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, is_dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtest import classify_actual_result, load_completed_matches_for_backtest
from src.completed_results import (
    COMPLETED_RESULTS_PATH,
    COMPLETED_RESULT_COLUMNS,
    completed_results_to_backtest_matches,
    load_completed_results,
    refresh_completed_results_for_fixture,
)
from src.config import DATA_DIR
from src.match_identity import validate_results_ledger_semantics
from src.model import ModelConfig, run_match_model, score_matrix
from src.model_policy import get_current_model_policy
from src.ratings import TEAM_RATING_COLUMNS, get_team_ratings
from src.team_names import normalize_team_name
from src.utils import coerce_bool, coerce_float, read_csv_with_columns, utc_now_iso
from src.venue_features import classify_venue_context
from src.weather import load_venues


PREDICTION_LEDGER_PATH = DATA_DIR / "prediction_ledger.csv"
RESULTS_LEDGER_PATH = DATA_DIR / "results_ledger.csv"
MANUAL_MISSING_RESULTS_PATH = DATA_DIR / "manual_missing_results.csv"

PREDICTION_LEDGER_COLUMNS = [
    "prediction_id",
    "snapshot_utc",
    "match_id",
    "kickoff_utc",
    "competition",
    "group",
    "home",
    "away",
    "venue",
    "primary_model_mode",
    "model_version",
    "parameter_set_id",
    "home_win_prob",
    "draw_prob",
    "away_win_prob",
    "home_xg",
    "away_xg",
    "over_0_5_prob",
    "over_1_5_prob",
    "over_2_5_prob",
    "over_3_5_prob",
    "btts_yes_prob",
    "model_confidence",
    "behavior_home_prob",
    "behavior_draw_prob",
    "behavior_away_prob",
    "market_home_prob",
    "market_draw_prob",
    "market_away_prob",
    "venue_country",
    "neutral_site",
    "venue_context",
    "altitude",
    "temperature",
    "humidity",
    "wind",
    "source_status",
    "market_snapshot_available",
    "prediction_before_kickoff",
    "notes",
]

RESULTS_LEDGER_COLUMNS = [
    "match_id",
    "date_utc",
    "competition",
    "home",
    "away",
    "home_goals",
    "away_goals",
    "actual_result",
    "total_goals",
    "btts_actual",
    "over_0_5_actual",
    "over_1_5_actual",
    "over_2_5_actual",
    "over_3_5_actual",
    "actual_advancing_team",
    "result_semantics",
    "evaluation_eligible_1x2",
    "had_extra_time",
    "had_penalties",
    "penalties_home",
    "penalties_away",
    "result_source",
    "last_updated",
]

DEFAULT_AUTO_SNAPSHOT_MIN_INTERVAL_MINUTES = 60
DEFAULT_RESULT_READY_DELAY_HOURS = 15.0 / 60.0
RESULT_IMPORT_DATE_TOLERANCE_DAYS = 1
SNAPSHOT_SELECTION_METADATA_COLUMNS = [
    "n_snapshots_for_match",
    "snapshot_rank_for_match",
    "is_latest_valid_snapshot",
]
UNRESOLVED_PREDICTION_TEAM_RE = re.compile(
    "|".join(
        [
            r"\bwinner\s",
            r"\bloser\s",
            r"\bwinner\s+group\b",
            r"\b3rd\s+group\b",
            r"\bthird\s+group\b",
            r"\btbd\b",
            r"\bto\s+be\s+determined\b",
        ]
    ),
    re.IGNORECASE,
)

HOME_REGULAR_TIME_GOAL_COLUMNS = [
    "home_goals_90",
    "home_score_90",
    "regular_time_home_goals",
    "regular_time_home_score",
    "home_goals_regular",
    "home_goals",
]
AWAY_REGULAR_TIME_GOAL_COLUMNS = [
    "away_goals_90",
    "away_score_90",
    "regular_time_away_goals",
    "regular_time_away_score",
    "away_goals_regular",
    "away_goals",
]


def load_prediction_ledger(path: str | Path = PREDICTION_LEDGER_PATH) -> pd.DataFrame:
    return read_csv_with_columns(Path(path), PREDICTION_LEDGER_COLUMNS)[PREDICTION_LEDGER_COLUMNS].copy()


def select_latest_valid_snapshots(
    df: pd.DataFrame,
    *,
    require_pre_kickoff: bool = True,
    exclude_unresolved_teams: bool = True,
    add_metadata: bool = True,
) -> pd.DataFrame:
    """Return one latest valid prediction snapshot per match without mutating the ledger.

    The append-only ledger keeps historical snapshots. Operational reporting and
    calibration should use this selector when they need one concrete prediction
    per match.
    """
    out = df.copy() if df is not None else pd.DataFrame()
    if out.empty:
        return _with_empty_snapshot_metadata(out, add_metadata)
    if "match_id" not in out.columns:
        return _with_empty_snapshot_metadata(out.iloc[0:0].copy(), add_metadata)

    original_columns = list(out.columns)
    out["_original_order"] = range(len(out))
    out["_snapshot_ts"] = pd.to_datetime(out.get("snapshot_utc", pd.Series(pd.NA, index=out.index)), errors="coerce", utc=True)

    if require_pre_kickoff:
        if "prediction_before_kickoff" in out.columns:
            kickoff_ts = pd.to_datetime(out.get("kickoff_utc", pd.Series(pd.NA, index=out.index)), errors="coerce", utc=True)
            explicit_flag = out["prediction_before_kickoff"].map(_prediction_ledger_has_value)
            flag_true = out["prediction_before_kickoff"].map(_prediction_ledger_truthy)
            inferred_before = out["_snapshot_ts"].notna() & kickoff_ts.notna() & (out["_snapshot_ts"] < kickoff_ts)
            out = out.loc[flag_true | (~explicit_flag & inferred_before)].copy()
        else:
            kickoff_ts = pd.to_datetime(out.get("kickoff_utc", pd.Series(pd.NA, index=out.index)), errors="coerce", utc=True)
            out = out.loc[out["_snapshot_ts"].notna() & kickoff_ts.notna() & (out["_snapshot_ts"] < kickoff_ts)].copy()

    if exclude_unresolved_teams and not out.empty:
        home_unresolved = out["home"].map(is_unresolved_prediction_team) if "home" in out.columns else pd.Series(False, index=out.index)
        away_unresolved = out["away"].map(is_unresolved_prediction_team) if "away" in out.columns else pd.Series(False, index=out.index)
        out = out.loc[~(home_unresolved | away_unresolved)].copy()

    if out.empty:
        return _with_empty_snapshot_metadata(out.loc[:, original_columns].copy(), add_metadata)

    ordered = out.sort_values(["match_id", "_snapshot_ts", "_original_order"], na_position="last").copy()
    ordered["n_snapshots_for_match"] = ordered.groupby("match_id")["match_id"].transform("size").astype(int)
    ordered["snapshot_rank_for_match"] = ordered.groupby("match_id").cumcount().add(1).astype(int)
    ordered["is_latest_valid_snapshot"] = ordered["snapshot_rank_for_match"].eq(ordered["n_snapshots_for_match"])
    latest = ordered.loc[ordered["is_latest_valid_snapshot"]].copy()
    keep_columns = original_columns + (SNAPSHOT_SELECTION_METADATA_COLUMNS if add_metadata else [])
    return latest.loc[:, keep_columns].reset_index(drop=True)


def is_unresolved_prediction_team(value: Any) -> bool:
    if pd.isna(value):
        return False
    text = re.sub(r"\s+", " ", str(value).strip().lower())
    return bool(UNRESOLVED_PREDICTION_TEAM_RE.search(f"{text} "))


def _prediction_ledger_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _prediction_ledger_has_value(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() not in {"", "nan", "none", "null", "<na>"}


def _with_empty_snapshot_metadata(df: pd.DataFrame, add_metadata: bool) -> pd.DataFrame:
    out = df.copy()
    if add_metadata:
        for col in SNAPSHOT_SELECTION_METADATA_COLUMNS:
            if col not in out.columns:
                out[col] = pd.Series(dtype="object")
    return out


def load_results_ledger(path: str | Path = RESULTS_LEDGER_PATH) -> pd.DataFrame:
    raw = read_csv_with_columns(Path(path), RESULTS_LEDGER_COLUMNS)
    return validate_results_ledger_semantics(raw)[RESULTS_LEDGER_COLUMNS].copy()


def load_manual_missing_results(
    start_date: str | None = None,
    end_date: str | None = None,
    teams: list[str] | None = None,
    path: str | Path = MANUAL_MISSING_RESULTS_PATH,
) -> pd.DataFrame:
    """Load manually verified missing 90-minute results as match candidates."""
    path = Path(path)
    candidate_columns = [
        "match_id",
        "provider_match_id",
        "provider",
        "date_utc",
        "competition",
        "group",
        "home",
        "away",
        "home_goals",
        "away_goals",
        "home_goals_90",
        "away_goals_90",
        "score_source",
        "score_semantics",
        "provider_kickoff_utc",
        "result_source",
        "last_updated",
    ]
    if not path.exists():
        return pd.DataFrame(columns=candidate_columns)
    try:
        raw = pd.read_csv(path)
    except Exception:
        return pd.DataFrame(columns=candidate_columns)
    if raw.empty:
        return pd.DataFrame(columns=candidate_columns)

    df = raw.copy()
    for col in ["match_id", "date_utc", "competition", "group", "home", "away", "score_semantics", "provider_kickoff_utc"]:
        if col not in df.columns:
            df[col] = pd.NA
    df["home_goals"] = df.apply(lambda row: _first_present(row, HOME_REGULAR_TIME_GOAL_COLUMNS), axis=1)
    df["away_goals"] = df.apply(lambda row: _first_present(row, AWAY_REGULAR_TIME_GOAL_COLUMNS), axis=1)
    df = df.loc[df["home_goals"].notna() & df["away_goals"].notna()].copy()
    df = df.loc[(df["home_goals"].astype(str).str.strip() != "") & (df["away_goals"].astype(str).str.strip() != "")].copy()
    if df.empty:
        return pd.DataFrame(columns=candidate_columns)

    dates = pd.to_datetime(df["date_utc"], errors="coerce", utc=True)
    if start_date:
        start = pd.to_datetime(start_date, errors="coerce", utc=True)
        if pd.notna(start):
            df = df.loc[dates >= start].copy()
            dates = dates.loc[df.index]
    if end_date:
        end = pd.to_datetime(end_date, errors="coerce", utc=True)
        if pd.notna(end):
            df = df.loc[dates < end + pd.Timedelta(days=1)].copy()
            dates = dates.loc[df.index]
    if teams:
        wanted = {_team_key(team) for team in teams if str(team or "").strip()}
        wanted.discard("")
        if wanted:
            df = df.loc[
                df["home"].map(_team_key).isin(wanted)
                | df["away"].map(_team_key).isin(wanted)
            ].copy()
    if df.empty:
        return pd.DataFrame(columns=candidate_columns)

    now = utc_now_iso()
    out = pd.DataFrame(
        {
            "match_id": df["match_id"].fillna("").astype(str),
            "provider_match_id": df["match_id"].fillna("").astype(str),
            "provider": "manual_missing_results",
            "date_utc": df["date_utc"],
            "competition": df["competition"],
            "group": df["group"],
            "home": df["home"],
            "away": df["away"],
            "home_goals": df["home_goals"],
            "away_goals": df["away_goals"],
            "home_goals_90": df["home_goals"],
            "away_goals_90": df["away_goals"],
            "score_source": "data/manual_missing_results.csv",
            "score_semantics": df["score_semantics"].fillna("90-minute regular time"),
            "provider_kickoff_utc": df["provider_kickoff_utc"],
            "result_source": "manual_verified_missing_results",
            "last_updated": now,
        }
    )
    out["score_semantics"] = out["score_semantics"].replace("", "90-minute regular time")
    return out[candidate_columns].copy()


def append_prediction_snapshots(
    snapshots_df: pd.DataFrame | None,
    path: str | Path = PREDICTION_LEDGER_PATH,
) -> pd.DataFrame:
    """Append prediction snapshots without overwriting prior ledger rows."""
    path = Path(path)
    new_rows = _ensure_columns(snapshots_df, PREDICTION_LEDGER_COLUMNS)
    existing = load_prediction_ledger(path)
    combined = pd.concat([existing, new_rows[PREDICTION_LEDGER_COLUMNS]], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    return combined[PREDICTION_LEDGER_COLUMNS].copy()


def snapshot_predictions_for_fixtures(
    fixtures_df: pd.DataFrame | None,
    team_ratings_df: pd.DataFrame | None = None,
    venues_df: pd.DataFrame | None = None,
    market_odds_df: pd.DataFrame | None = None,
    cfg: ModelConfig | None = None,
    model_version: str = "baseline_external_calibrated_v1",
    parameter_set_id: str = "baseline_current",
    primary_model_mode: str = "baseline_manual",
    notes: str = "",
    path: str | Path = PREDICTION_LEDGER_PATH,
    append: bool = True,
) -> pd.DataFrame:
    """Snapshot current pre-match predictions for fixture rows.

    The function is append-only by default. Re-snapshotting the same match and
    model creates a new row with a new `snapshot_utc`.
    """
    snapshots, _diagnostics = snapshot_predictions_for_fixtures_with_diagnostics(
        fixtures_df,
        team_ratings_df=team_ratings_df,
        venues_df=venues_df,
        market_odds_df=market_odds_df,
        cfg=cfg,
        model_version=model_version,
        parameter_set_id=parameter_set_id,
        primary_model_mode=primary_model_mode,
        notes=notes,
        path=path,
        append=append,
        include_past=True,
    )
    return snapshots


def snapshot_predictions_for_fixtures_with_diagnostics(
    fixtures_df: pd.DataFrame | None,
    team_ratings_df: pd.DataFrame | None = None,
    venues_df: pd.DataFrame | None = None,
    market_odds_df: pd.DataFrame | None = None,
    cfg: ModelConfig | None = None,
    model_version: str = "baseline_external_calibrated_v1",
    parameter_set_id: str = "baseline_current",
    primary_model_mode: str = "baseline_manual",
    notes: str = "",
    path: str | Path = PREDICTION_LEDGER_PATH,
    append: bool = True,
    include_past: bool = False,
    selected_match_ids: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Snapshot predictions with explicit write/skip diagnostics.

    `include_past=False` skips snapshots at or after kickoff so dashboard saves
    are valid for honest post-mortem by default. Passing `include_past=True`
    still writes the row but marks `prediction_before_kickoff=False`.
    """
    fixtures = _normalise_fixtures(fixtures_df)
    fixtures_available = int(len(fixtures))
    if selected_match_ids is not None:
        wanted = {str(match_id) for match_id in selected_match_ids if str(match_id).strip()}
        fixtures = fixtures.loc[fixtures["match_id"].astype(str).isin(wanted)].copy() if wanted else fixtures.iloc[0:0].copy()
    fixtures_selected = int(len(fixtures))
    path = Path(path)
    output_path = _display_path(path)
    skip_rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    def diagnostics(rows_written: int, rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        rows = rows or skip_rows
        skipped = int(len(rows))
        return {
            "fixtures_available": fixtures_available,
            "fixtures_selected": fixtures_selected,
            "fixtures_attempted": int(fixtures_selected - sum(1 for row in rows if row.get("reason_code") in {"past_kickoff", "missing_match_id"})),
            "predictions_written": int(rows_written),
            "predictions_skipped": skipped,
            "skipped_past_kickoff": sum(1 for row in rows if row.get("reason_code") == "past_kickoff"),
            "skipped_missing_model_result": sum(1 for row in rows if row.get("reason_code") == "missing_model_result"),
            "skipped_missing_match_id": sum(1 for row in rows if row.get("reason_code") == "missing_match_id"),
            "skipped_duplicate_policy": 0,
            "output_path": output_path,
            "warnings": warnings.copy(),
            "skip_reasons": rows.copy(),
        }

    if fixtures.empty:
        warning = "No fixtures were selected." if fixtures_available else "No fixtures are available for the requested snapshot."
        warnings.append(warning)
        return pd.DataFrame(columns=PREDICTION_LEDGER_COLUMNS), diagnostics(0)

    snapshot_utc = utc_now_iso()
    teams = team_ratings_df.copy() if team_ratings_df is not None else _load_primary_ratings()
    venues = venues_df.copy() if venues_df is not None else load_venues()
    odds = market_odds_df.copy() if market_odds_df is not None else pd.DataFrame()
    model_cfg = cfg or ModelConfig()
    policy = get_current_model_policy()
    rows: list[dict[str, Any]] = []

    for _, fixture in fixtures.iterrows():
        match_id_raw = str(fixture.get("match_id", "") or "").strip()
        if not match_id_raw:
            skip_rows.append(_skip_row(fixture, "missing_match_id", "Fixture has no match_id, so it was not written to the prediction ledger."))
            continue
        kickoff = _kickoff_utc(fixture)
        before_kickoff = _before_kickoff(snapshot_utc, kickoff)
        if not include_past and not before_kickoff:
            skip_rows.append(_skip_row(fixture, "past_kickoff", "Fixture kickoff is not after the snapshot time."))
            continue
        try:
            result = run_match_model(fixture, teams, venues, odds, model_cfg, model_mode=primary_model_mode)
        except TypeError:
            try:
                result = run_match_model(fixture, teams, venues, odds, model_cfg)
            except Exception as exc:
                skip_rows.append(_skip_row(fixture, "missing_model_result", f"Model result could not be generated: {exc}"))
                continue
        except Exception as exc:
            skip_rows.append(_skip_row(fixture, "missing_model_result", f"Model result could not be generated: {exc}"))
            continue
        if not isinstance(result, dict) or not result.get("probs"):
            skip_rows.append(_skip_row(fixture, "missing_model_result", "Model result was missing probabilities."))
            continue
        probs = result.get("probs", {})
        matrix = result.get("score_matrix")
        over_0_5 = _over_probability(matrix, 0)
        over_1_5 = _over_probability(matrix, 1)
        components = result.get("components", {}) if isinstance(result.get("components", {}), dict) else {}
        environment = result.get("environment", {}) if isinstance(result.get("environment", {}), dict) else {}
        confidence = result.get("confidence", {}) if isinstance(result.get("confidence", {}), dict) else {}
        market_home_prob, market_draw_prob, market_away_prob = _market_1x2_probabilities(
            odds,
            match_id_raw,
            fixture.get("home", ""),
            fixture.get("away", ""),
        )
        venue_country = str(components.get("venue_country", "") or environment.get("country", "") or "")
        neutral_site = components.get("neutral_site", environment.get("neutral_site", pd.NA))
        venue_context = str(
            components.get("venue_context", "")
            or classify_venue_context(
                fixture.get("home", ""),
                fixture.get("away", ""),
                {"country": venue_country, "neutral_site": neutral_site},
            )
        )
        match_id = match_id_raw
        prediction_id = _prediction_id(snapshot_utc, match_id, model_version, parameter_set_id)
        rows.append(
            {
                "prediction_id": prediction_id,
                "snapshot_utc": snapshot_utc,
                "match_id": match_id,
                "kickoff_utc": kickoff,
                "competition": fixture.get("competition", ""),
                "group": fixture.get("group", ""),
                "home": fixture.get("home", ""),
                "away": fixture.get("away", ""),
                "venue": fixture.get("venue", ""),
                "primary_model_mode": primary_model_mode,
                "model_version": model_version,
                "parameter_set_id": parameter_set_id,
                "home_win_prob": float(probs.get("home_win", 0.0)),
                "draw_prob": float(probs.get("draw", 0.0)),
                "away_win_prob": float(probs.get("away_win", 0.0)),
                "home_xg": float(result.get("hxg", 0.0)),
                "away_xg": float(result.get("axg", 0.0)),
                "over_0_5_prob": over_0_5,
                "over_1_5_prob": over_1_5,
                "over_2_5_prob": float(probs.get("over_2_5", 0.0)),
                "over_3_5_prob": float(probs.get("over_3_5", 0.0)),
                "btts_yes_prob": float(probs.get("btts_yes", 0.0)),
                "model_confidence": confidence.get("label", ""),
                "behavior_home_prob": pd.NA,
                "behavior_draw_prob": pd.NA,
                "behavior_away_prob": pd.NA,
                "market_home_prob": market_home_prob,
                "market_draw_prob": market_draw_prob,
                "market_away_prob": market_away_prob,
                "venue_country": venue_country,
                "neutral_site": neutral_site,
                "venue_context": venue_context,
                "altitude": components.get("altitude_m", environment.get("altitude_m", pd.NA)),
                "temperature": components.get("effective_temp_c", environment.get("effective_temp_c", environment.get("temp_c", pd.NA))),
                "humidity": components.get("effective_humidity_pct", environment.get("effective_humidity_pct", environment.get("humidity_pct", pd.NA))),
                "wind": components.get("effective_wind_kmh", environment.get("effective_wind_kmh", environment.get("wind_kmh", pd.NA))),
                "source_status": f"policy={policy.get('primary_model_mode', primary_model_mode)}; offline_csv_safe",
                "market_snapshot_available": bool(_market_snapshot_available(odds, match_id)),
                "prediction_before_kickoff": bool(before_kickoff),
                "notes": notes,
            }
        )

    snapshots = pd.DataFrame(rows, columns=PREDICTION_LEDGER_COLUMNS)
    if append:
        append_prediction_snapshots(snapshots, path=path)
    return snapshots, diagnostics(len(snapshots) if append else 0)


def dashboard_parameter_set_id(cfg: ModelConfig | None, baseline_id: str = "baseline_current") -> str:
    """Create a stable parameter id for dashboard snapshots."""
    model_cfg = cfg or ModelConfig()
    default_cfg = ModelConfig()
    if _config_payload(model_cfg) == _config_payload(default_cfg):
        return baseline_id
    raw = "|".join(f"{key}={value}" for key, value in sorted(_config_payload(model_cfg).items()))
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"dashboard_custom_{digest}"


def snapshot_selected_match_if_needed(
    fixture_row: pd.Series | dict[str, Any] | pd.DataFrame | None,
    team_ratings_df: pd.DataFrame | None = None,
    venues_df: pd.DataFrame | None = None,
    market_odds_df: pd.DataFrame | None = None,
    cfg: ModelConfig | None = None,
    model_version: str = "baseline_external_calibrated_v1",
    parameter_set_id: str | None = None,
    primary_model_mode: str = "baseline_manual",
    notes: str = "Auto-saved by dashboard selected-match analysis",
    path: str | Path = PREDICTION_LEDGER_PATH,
    min_interval_minutes: int = DEFAULT_AUTO_SNAPSHOT_MIN_INTERVAL_MINUTES,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Append a pre-kickoff selected-match snapshot unless a fresh one exists."""
    fixture = _single_fixture_frame(fixture_row)
    parameter_id = parameter_set_id or dashboard_parameter_set_id(cfg)
    match_id = _first_match_id(fixture)
    duplicate = _fresh_existing_prediction(
        match_id,
        model_version=model_version,
        parameter_set_id=parameter_id,
        primary_model_mode=primary_model_mode,
        path=path,
        min_interval_minutes=min_interval_minutes,
    )
    if duplicate:
        diagnostics = {
            "fixtures_available": int(len(fixture)),
            "fixtures_selected": int(len(fixture)),
            "fixtures_attempted": 0,
            "predictions_written": 0,
            "predictions_skipped": 1,
            "skipped_past_kickoff": 0,
            "skipped_missing_model_result": 0,
            "skipped_missing_match_id": 0,
            "skipped_duplicate_policy": 1,
            "output_path": _display_path(Path(path)),
            "warnings": [],
            "skip_reasons": [
                {
                    "match_id": match_id,
                    "reason_code": "duplicate_policy",
                    "reason": f"A fresh pre-kickoff snapshot already exists within {int(min_interval_minutes)} minute(s).",
                }
            ],
        }
        return pd.DataFrame(columns=PREDICTION_LEDGER_COLUMNS), diagnostics
    return snapshot_predictions_for_fixtures_with_diagnostics(
        fixture,
        team_ratings_df=team_ratings_df,
        venues_df=venues_df,
        market_odds_df=market_odds_df,
        cfg=cfg,
        model_version=model_version,
        parameter_set_id=parameter_id,
        primary_model_mode=primary_model_mode,
        notes=notes,
        path=path,
        append=True,
        include_past=False,
        selected_match_ids=[match_id] if match_id else None,
    )


def import_completed_result_for_fixture_if_ready(
    fixture_row: pd.Series | dict[str, Any],
    completed_matches_df: pd.DataFrame | None = None,
    path: str | Path = RESULTS_LEDGER_PATH,
    now_utc: str | pd.Timestamp | None = None,
    result_ready_delay_hours: float = DEFAULT_RESULT_READY_DELAY_HOURS,
    result_source: str = "auto_dashboard_completed_matches",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Import one selected fixture result after a conservative post-game delay.

    The app waits four hours after kickoff by default. Imported market outcomes
    use regulation time plus stoppage when a completed-result source exposes
    90-minute score columns, because Polymarket match markets settle on the
    regular-time result rather than extra time or penalties.
    """
    fixture = pd.Series(fixture_row).copy()
    path = Path(path)
    existing = load_results_ledger(path)
    match_id = str(fixture.get("match_id", "") or "").strip()
    kickoff = pd.to_datetime(_kickoff_utc(fixture), errors="coerce", utc=True)
    now = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    if not match_id:
        return existing, _result_import_diagnostics("skipped_missing_match_id", match_id, "", 0, "Fixture has no match_id.")
    if pd.isna(kickoff):
        return existing, _result_import_diagnostics("skipped_invalid_kickoff", match_id, "", 0, "Fixture kickoff could not be parsed.")
    ready_at = kickoff + pd.Timedelta(hours=float(result_ready_delay_hours))
    if now < ready_at:
        return existing, _result_import_diagnostics(
            "not_ready",
            match_id,
            ready_at.isoformat(),
            0,
            "Result import waits until kickoff plus the post-game delay.",
        )
    if not existing.empty and "match_id" in existing.columns and (existing["match_id"].astype(str) == match_id).any():
        return existing, _result_import_diagnostics("already_imported", match_id, ready_at.isoformat(), 0, "Result already exists in the ledger.")

    if completed_matches_df is not None:
        completed = completed_matches_df.copy()
        completed_source_diagnostics: dict[str, Any] = {
            "status": "provided_completed_matches",
            "provider_checked": "provided dataframe",
            "completed_rows_loaded": len(completed),
        }
    else:
        completed, completed_source_diagnostics = _completed_candidates_for_fixture(fixture)
    candidate = _completed_result_for_fixture(fixture, completed)
    if candidate.empty:
        completed_rows_loaded = int(completed_source_diagnostics.get("completed_rows_loaded", len(completed)) or 0)
        status = "completed_source_empty" if completed_rows_loaded <= 0 else "completed_rows_exist_no_fixture_match"
        message = (
            "No completed-result source rows were available for this fixture window."
            if status == "completed_source_empty"
            else "Completed-result rows were loaded, but none matched this fixture."
        )
        return existing, _result_import_diagnostics(
            status,
            match_id,
            ready_at.isoformat(),
            0,
            message,
            source_diagnostics=completed_source_diagnostics,
            candidate_rows=_nearest_completed_candidates(fixture, completed),
        )

    result_row = candidate.iloc[0].copy()
    result_row["match_id"] = match_id
    for col in ["date_utc", "competition", "home", "away"]:
        if col in fixture.index:
            result_row[col] = fixture.get(col, result_row.get(col, ""))
    combined = import_completed_results(pd.DataFrame([result_row]), path=path, result_source=result_source)
    imported_score = f"{int(coerce_float(result_row.get('home_goals'), 0))}-{int(coerce_float(result_row.get('away_goals'), 0))}"
    return combined, _result_import_diagnostics(
        "imported",
        match_id,
        ready_at.isoformat(),
        1,
        "Completed result imported.",
        source_diagnostics=completed_source_diagnostics,
        candidate_rows=_nearest_completed_candidates(fixture, candidate),
        matched_row={
            "home": result_row.get("home", ""),
            "away": result_row.get("away", ""),
            "score": imported_score,
            "score_semantics": str(result_row.get("score_semantics", "") or "90-minute regular time"),
            "score_source": str(result_row.get("score_source", "") or ""),
        },
    )


def sync_all_completed_results(
    fixtures: pd.DataFrame | None,
    competition: str = "FIFA World Cup",
    date_min: str | None = None,
    date_max: str | None = None,
    force_refresh: bool = False,
    path: str | Path = RESULTS_LEDGER_PATH,
    now_utc: str | pd.Timestamp | None = None,
    result_ready_delay_hours: float = DEFAULT_RESULT_READY_DELAY_HOURS,
    result_source: str = "bulk_completed_results_sync",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Refresh and import completed results for every finished local fixture."""
    path = Path(path)
    existing = load_results_ledger(path)
    all_fixtures = _normalise_fixtures(fixtures)
    if competition and not all_fixtures.empty and "competition" in all_fixtures.columns:
        needle = str(competition).strip().lower()
        all_fixtures = all_fixtures.loc[all_fixtures["competition"].astype(str).str.lower().str.contains(needle, na=False)].copy()

    now = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    prepared, skipped_rows = _bulk_sync_ready_fixtures(
        all_fixtures,
        now=now,
        date_min=date_min,
        date_max=date_max,
        result_ready_delay_hours=result_ready_delay_hours,
    )
    if prepared.empty:
        diagnostics = _bulk_sync_diagnostics(
            status="no_past_fixtures",
            provider_diagnostics={},
            fixture_rows=[],
            skipped_rows=skipped_rows,
            total_fixtures=len(all_fixtures),
            past_fixtures=0,
            existing_rows=len(existing),
            final_rows=len(existing),
            manual_candidate_rows=0,
            manual_path_checked=str(MANUAL_MISSING_RESULTS_PATH),
        )
        return existing, diagnostics

    start = prepared["_kickoff_utc"].dt.date.min().isoformat()
    end = prepared["_kickoff_utc"].dt.date.max().isoformat()
    completed_results, provider_diagnostics = load_completed_results(
        start,
        end,
        teams=None,
        force_refresh=force_refresh,
        persist=True,
    )
    provider_candidates = _completed_results_to_match_candidates(completed_results)
    fallback_candidates = load_completed_matches_for_backtest(start_date=start, end_date=end, teams=None)
    manual_candidates = load_manual_missing_results(start_date=start, end_date=end)
    provider_and_fallback = _combine_completed_candidate_frames([
        provider_candidates,
        fallback_candidates,
        manual_candidates,
    ])
    existing_candidates = _results_ledger_to_completed_candidates(existing)
    full_candidate_pool = _combine_completed_candidate_frames([provider_and_fallback, existing_candidates])

    working = existing.copy()
    rows_to_append: list[dict[str, Any]] = []
    fixture_rows: list[dict[str, Any]] = []
    for _, fixture in prepared.iterrows():
        candidate = _completed_result_for_fixture(fixture, provider_and_fallback)
        matched_from_existing_only = False
        if candidate.empty:
            candidate = _completed_result_for_fixture(fixture, existing_candidates)
            matched_from_existing_only = not candidate.empty
        if candidate.empty:
            fixture_rows.append(
                _bulk_fixture_diagnostic(
                    fixture,
                    "unmatched",
                    reason="No completed-result row matched this fixture.",
                    nearest=_nearest_completed_candidates(fixture, full_candidate_pool),
                )
            )
            continue

        result_row = _ledger_result_row_from_candidate(fixture, candidate.iloc[0], result_source)
        match_id = str(result_row.get("match_id", "") or "").strip()
        existing_row = _existing_result_for_match_id(working, match_id)
        if existing_row is not None:
            old_score = _score_pair(existing_row)
            new_score = _score_pair(result_row)
            if old_score == new_score:
                fixture_rows.append(
                    _bulk_fixture_diagnostic(
                        fixture,
                        "already_present",
                        candidate.iloc[0],
                        result_row,
                        reason="Result already exists in data/results_ledger.csv.",
                    )
                )
                continue
            fixture_rows.append(
                _bulk_fixture_diagnostic(
                    fixture,
                    "conflict_needs_review",
                    candidate.iloc[0],
                    result_row,
                    reason=f"Existing score {old_score[0]}-{old_score[1]} conflicts with provider score {new_score[0]}-{new_score[1]}.",
                )
            )
            continue

        rows_to_append.append(result_row)
        fixture_rows.append(
            _bulk_fixture_diagnostic(
                fixture,
                "imported_from_existing_result" if matched_from_existing_only else "imported",
                candidate.iloc[0],
                result_row,
            )
        )

    if rows_to_append:
        new_results = pd.DataFrame(rows_to_append, columns=RESULTS_LEDGER_COLUMNS)
        working = pd.concat([working, new_results], ignore_index=True)
        working = working.drop_duplicates(subset=["match_id"], keep="last")
        working = working.sort_values(["date_utc", "match_id"]).reset_index(drop=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        working[RESULTS_LEDGER_COLUMNS].to_csv(path, index=False)

    status = "completed_source_empty" if provider_and_fallback.empty and existing_candidates.empty else "success"
    diagnostics = _bulk_sync_diagnostics(
        status=status,
        provider_diagnostics=provider_diagnostics,
        fixture_rows=fixture_rows,
        skipped_rows=skipped_rows,
        total_fixtures=len(all_fixtures),
        past_fixtures=len(prepared),
        existing_rows=len(existing),
        final_rows=len(working),
        provider_rows=len(completed_results),
        provider_candidate_rows=len(provider_candidates),
        fallback_candidate_rows=len(fallback_candidates) if fallback_candidates is not None else 0,
        manual_candidate_rows=len(manual_candidates),
        manual_path_checked=str(MANUAL_MISSING_RESULTS_PATH),
        existing_candidate_rows=len(existing_candidates),
        date_window=f"{start} to {end}",
    )
    return working[RESULTS_LEDGER_COLUMNS].copy(), diagnostics


def auto_sync_completed_results_on_launch(
    fixtures_df: pd.DataFrame | None,
    competition: str = "FIFA World Cup",
    result_ready_delay_minutes: float = 15,
    force_refresh: bool = True,
    path: str | Path = RESULTS_LEDGER_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Best-effort app-launch import of completed 90-minute fixture results."""
    path = Path(path)
    try:
        results, diagnostics = sync_all_completed_results(
            fixtures_df,
            competition=competition,
            force_refresh=force_refresh,
            path=path,
            result_ready_delay_hours=float(result_ready_delay_minutes) / 60.0,
            result_source="auto_app_launch_completed_results_sync",
        )
    except Exception as exc:
        try:
            results = load_results_ledger(path)
        except Exception:
            results = pd.DataFrame(columns=RESULTS_LEDGER_COLUMNS)
        diagnostics = {
            "status": "auto_sync_failed",
            "message": f"Completed-result auto-sync failed without stopping the app: {exc}",
            "fixture_diagnostics": [],
            "unmatched_rows": [],
            "conflict_rows_detail": [],
            "final_ledger_rows": len(results),
        }
    diagnostics = _app_launch_sync_diagnostics(
        diagnostics,
        results,
        result_ready_delay_minutes=result_ready_delay_minutes,
        force_refresh=force_refresh,
        path=path,
    )
    return results[RESULTS_LEDGER_COLUMNS].copy(), diagnostics


def import_completed_results(
    completed_matches_df: pd.DataFrame | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    competition: str | None = None,
    path: str | Path = RESULTS_LEDGER_PATH,
    result_source: str = "completed_matches",
) -> pd.DataFrame:
    """Create or update result ledger rows from completed match data."""
    matches = completed_matches_df.copy() if completed_matches_df is not None else load_completed_matches_for_backtest(start_date, end_date)
    if matches is None or matches.empty:
        out = load_results_ledger(path)
        if not Path(path).exists():
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(path, index=False)
        return out
    matches = matches.copy()
    if competition and "competition" in matches.columns:
        needle = str(competition).strip().lower()
        matches = matches.loc[matches["competition"].astype(str).str.lower().str.contains(needle, na=False)].copy()
    rows = []
    now = utc_now_iso()
    for _, match in matches.iterrows():
        home_goals = coerce_float(_first_present(match, HOME_REGULAR_TIME_GOAL_COLUMNS), 0.0)
        away_goals = coerce_float(_first_present(match, AWAY_REGULAR_TIME_GOAL_COLUMNS), 0.0)
        total = home_goals + away_goals
        rows.append(
            {
                "match_id": str(match.get("match_id", "") or _fallback_match_id(match)),
                "date_utc": match.get("date_utc", ""),
                "competition": match.get("competition", ""),
                "home": match.get("home", ""),
                "away": match.get("away", ""),
                "home_goals": int(home_goals),
                "away_goals": int(away_goals),
                "actual_result": classify_actual_result(home_goals, away_goals),
                "total_goals": int(total),
                "btts_actual": int(home_goals > 0 and away_goals > 0),
                "over_0_5_actual": int(total > 0.5),
                "over_1_5_actual": int(total > 1.5),
                "over_2_5_actual": int(total > 2.5),
                "over_3_5_actual": int(total > 3.5),
                "actual_advancing_team": match.get("actual_advancing_team", pd.NA),
                "result_semantics": match.get("result_semantics", match.get("score_semantics", "90-minute regular time")),
                "evaluation_eligible_1x2": True,
                "had_extra_time": False,
                "had_penalties": False,
                "penalties_home": pd.NA,
                "penalties_away": pd.NA,
                "result_source": result_source,
                "last_updated": now,
            }
        )
    new_results = validate_results_ledger_semantics(pd.DataFrame(rows, columns=RESULTS_LEDGER_COLUMNS))
    existing = load_results_ledger(path)
    combined = pd.concat([existing, new_results], ignore_index=True)
    if not combined.empty:
        combined = combined.drop_duplicates(subset=["match_id"], keep="last")
        combined = combined.sort_values(["date_utc", "match_id"]).reset_index(drop=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined[RESULTS_LEDGER_COLUMNS].to_csv(path, index=False)
    return combined[RESULTS_LEDGER_COLUMNS].copy()


def _load_primary_ratings() -> pd.DataFrame:
    try:
        return get_team_ratings(model_mode="baseline_manual")
    except TypeError:
        return read_csv_with_columns(DATA_DIR / "team_ratings.csv", TEAM_RATING_COLUMNS)


def _config_payload(cfg: ModelConfig) -> dict[str, Any]:
    if is_dataclass(cfg):
        raw = asdict(cfg)
    else:
        raw = dict(getattr(cfg, "__dict__", {}))
    payload: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, float):
            payload[key] = round(value, 8)
        else:
            payload[key] = value
    return payload


def _single_fixture_frame(fixture_row: pd.Series | dict[str, Any] | pd.DataFrame | None) -> pd.DataFrame:
    if fixture_row is None:
        return pd.DataFrame()
    if isinstance(fixture_row, pd.DataFrame):
        return fixture_row.head(1).copy()
    return pd.DataFrame([pd.Series(fixture_row)])


def _first_match_id(fixtures: pd.DataFrame) -> str:
    if fixtures is None or fixtures.empty or "match_id" not in fixtures.columns:
        return ""
    return str(fixtures["match_id"].iloc[0] or "").strip()


def _fresh_existing_prediction(
    match_id: str,
    model_version: str,
    parameter_set_id: str,
    primary_model_mode: str,
    path: str | Path,
    min_interval_minutes: int,
) -> bool:
    if not match_id:
        return False
    existing = load_prediction_ledger(path)
    if existing.empty:
        return False
    frame = existing.copy()
    mask = (
        frame["match_id"].astype(str).eq(str(match_id))
        & frame["model_version"].astype(str).eq(str(model_version))
        & frame["parameter_set_id"].astype(str).eq(str(parameter_set_id))
        & frame["primary_model_mode"].astype(str).eq(str(primary_model_mode))
        & frame["prediction_before_kickoff"].astype(str).str.lower().isin({"true", "1", "yes"})
    )
    matched = frame.loc[mask].copy()
    if matched.empty:
        return False
    snapshots = pd.to_datetime(matched["snapshot_utc"], errors="coerce", utc=True).dropna()
    if snapshots.empty:
        return False
    latest = snapshots.max()
    now = pd.to_datetime(utc_now_iso(), errors="coerce", utc=True)
    if pd.isna(now):
        now = pd.Timestamp.now(tz="UTC")
    return bool(latest >= now - pd.Timedelta(minutes=int(min_interval_minutes)))


def _completed_candidates_for_fixture(fixture: pd.Series) -> tuple[pd.DataFrame, dict[str, Any]]:
    date_label = _date_label(fixture.get("date_utc"))
    fixture_date = pd.to_datetime(date_label, errors="coerce")
    if pd.isna(fixture_date):
        return pd.DataFrame(), {"status": "invalid_fixture_date", "completed_rows_loaded": 0}
    start = (fixture_date.date() - timedelta(days=RESULT_IMPORT_DATE_TOLERANCE_DAYS)).isoformat()
    end = (fixture_date.date() + timedelta(days=RESULT_IMPORT_DATE_TOLERANCE_DAYS)).isoformat()
    teams = [str(fixture.get("home", "") or ""), str(fixture.get("away", "") or "")]
    teams = [team for team in teams if team.strip()]
    refreshed, diagnostics = refresh_completed_results_for_fixture(
        fixture,
        window_days=RESULT_IMPORT_DATE_TOLERANCE_DAYS,
        force_refresh=True,
        path=COMPLETED_RESULTS_PATH,
    )
    completed_matches = completed_results_to_backtest_matches(refreshed)
    fallback_matches = load_completed_matches_for_backtest(start_date=start, end_date=end, teams=teams or None)
    frames = [df for df in [completed_matches, fallback_matches] if df is not None and not df.empty]
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined = combined.drop_duplicates(subset=["match_id", "date_utc", "home", "away"], keep="first")
    else:
        combined = pd.DataFrame()
    diagnostics["completed_rows_loaded"] = int(len(combined))
    diagnostics["completed_rows_from_refresh"] = int(len(completed_matches))
    diagnostics["completed_rows_from_fallback"] = int(len(fallback_matches))
    return combined, diagnostics


def _completed_result_for_fixture(fixture: pd.Series, completed_matches_df: pd.DataFrame | None) -> pd.DataFrame:
    completed = completed_matches_df.copy() if completed_matches_df is not None else pd.DataFrame()
    if completed.empty:
        return pd.DataFrame()
    for col in ["match_id", "date_utc", "home", "away", "home_goals", "away_goals"]:
        if col not in completed.columns:
            completed[col] = pd.NA
    fixture_match_id = str(fixture.get("match_id", "") or "").strip()
    if fixture_match_id:
        by_id = completed.loc[completed["match_id"].astype(str) == fixture_match_id].copy()
        if not by_id.empty:
            return _orient_completed_match_to_fixture(fixture, by_id).head(1)
    date_label = _date_label(fixture.get("date_utc"))
    fixture_date = pd.to_datetime(date_label, errors="coerce")
    home_key = _team_key(fixture.get("home", ""))
    away_key = _team_key(fixture.get("away", ""))
    home = completed["home"].map(_team_key)
    away = completed["away"].map(_team_key)
    same_order = (home == home_key) & (away == away_key)
    reverse_order = (home == away_key) & (away == home_key)
    if not home_key or not away_key:
        return pd.DataFrame()
    date_delta = _date_delta_days(completed["date_utc"], fixture_date)
    competition_ok = completed.apply(lambda row: _competition_compatible(fixture.get("competition", ""), row.get("competition", "")), axis=1)
    by_fixture = completed.loc[(same_order | reverse_order) & competition_ok & (date_delta <= RESULT_IMPORT_DATE_TOLERANCE_DAYS)].copy()
    if by_fixture.empty:
        return by_fixture
    by_fixture["_date_delta"] = date_delta.loc[by_fixture.index]
    by_fixture["_same_order"] = same_order.loc[by_fixture.index].astype(int)
    by_fixture = by_fixture.sort_values(["_date_delta", "_same_order"], ascending=[True, False])
    by_fixture = by_fixture.drop(columns=["_date_delta", "_same_order"], errors="ignore")
    return _orient_completed_match_to_fixture(fixture, by_fixture).head(1)


def _result_import_diagnostics(
    status: str,
    match_id: str,
    ready_after_utc: str,
    rows_imported: int,
    message: str,
    source_diagnostics: dict[str, Any] | None = None,
    candidate_rows: list[dict[str, Any]] | None = None,
    matched_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_diagnostics = source_diagnostics or {}
    return {
        "status": status,
        "match_id": match_id,
        "ready_after_utc": ready_after_utc,
        "rows_imported": int(rows_imported),
        "message": message,
        "provider_checked": source_diagnostics.get("provider_checked", ""),
        "date_window_checked": source_diagnostics.get("date_window_checked", ""),
        "cache_path_checked": source_diagnostics.get("cache_path_checked", ""),
        "local_path_checked": source_diagnostics.get("local_path_checked", ""),
        "completed_rows_loaded": int(source_diagnostics.get("completed_rows_loaded", 0) or 0),
        "completed_rows_from_refresh": int(source_diagnostics.get("completed_rows_from_refresh", 0) or 0),
        "completed_rows_from_fallback": int(source_diagnostics.get("completed_rows_from_fallback", 0) or 0),
        "completed_source_status": source_diagnostics.get("status", ""),
        "completed_last_updated": source_diagnostics.get("last_updated", ""),
        "score_semantics": (matched_row or {}).get("score_semantics", ""),
        "imported_score": (matched_row or {}).get("score", ""),
        "matched_row": matched_row or {},
        "nearest_candidate_rows": candidate_rows or [],
    }


def _date_label(value: Any) -> str:
    ts = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(ts) else ts.date().isoformat()


def _team_key(value: Any) -> str:
    canonical = normalize_team_name(str(value or ""))
    return " ".join(str(canonical or "").strip().lower().replace("-", " ").split())


def _competition_compatible(fixture_competition: Any, candidate_competition: Any) -> bool:
    fixture_key = _team_key(fixture_competition)
    candidate_key = _team_key(candidate_competition)
    if not fixture_key or not candidate_key:
        return True
    if fixture_key == candidate_key:
        return True
    if "world cup" in fixture_key and "world cup" in candidate_key:
        return True
    return fixture_key in candidate_key or candidate_key in fixture_key


def _date_delta_days(values: pd.Series, fixture_date: pd.Timestamp) -> pd.Series:
    if pd.isna(fixture_date):
        return pd.Series([9999] * len(values), index=values.index)
    dates = pd.to_datetime(values.map(_date_label), errors="coerce")
    return (dates - fixture_date).abs().dt.days.fillna(9999).astype(int)


def _nearest_completed_candidates(fixture: pd.Series, completed: pd.DataFrame | None, limit: int = 5) -> list[dict[str, Any]]:
    if completed is None or completed.empty:
        return []
    out = completed.copy()
    for col in ["date_utc", "home", "away", "home_goals", "away_goals", "competition"]:
        if col not in out.columns:
            out[col] = ""
    fixture_date = pd.to_datetime(_date_label(fixture.get("date_utc")), errors="coerce")
    out["_date_delta"] = _date_delta_days(out["date_utc"], fixture_date)
    fixture_teams = {_team_key(fixture.get("home", "")), _team_key(fixture.get("away", ""))}
    out["_team_overlap"] = out.apply(
        lambda row: len(fixture_teams & {_team_key(row.get("home", "")), _team_key(row.get("away", ""))}),
        axis=1,
    )
    out = out.sort_values(["_team_overlap", "_date_delta"], ascending=[False, True]).head(limit)
    rows = []
    for _, row in out.iterrows():
        rows.append(
            {
                "date_utc": str(row.get("date_utc", "") or ""),
                "competition": str(row.get("competition", "") or ""),
                "home": str(row.get("home", "") or ""),
                "away": str(row.get("away", "") or ""),
                "score": f"{int(coerce_float(row.get('home_goals'), 0))}-{int(coerce_float(row.get('away_goals'), 0))}",
                "date_delta_days": int(row.get("_date_delta", 9999)),
                "team_overlap": int(row.get("_team_overlap", 0)),
            }
        )
    return rows


def _orient_completed_match_to_fixture(fixture: pd.Series, completed: pd.DataFrame) -> pd.DataFrame:
    if completed is None or completed.empty:
        return pd.DataFrame()
    out = completed.copy()
    home_key = _team_key(fixture.get("home", ""))
    away_key = _team_key(fixture.get("away", ""))
    if not home_key or not away_key:
        return out
    for idx, row in out.iterrows():
        row_home = _team_key(row.get("home", ""))
        row_away = _team_key(row.get("away", ""))
        home_goals = _first_present(row, HOME_REGULAR_TIME_GOAL_COLUMNS)
        away_goals = _first_present(row, AWAY_REGULAR_TIME_GOAL_COLUMNS)
        if row_home == away_key and row_away == home_key:
            out.at[idx, "home"] = fixture.get("home", row.get("away", ""))
            out.at[idx, "away"] = fixture.get("away", row.get("home", ""))
            out.at[idx, "home_goals"] = away_goals
            out.at[idx, "away_goals"] = home_goals
            out.at[idx, "home_goals_90"] = away_goals
            out.at[idx, "away_goals_90"] = home_goals
            out.at[idx, "home_score_90"] = away_goals
            out.at[idx, "away_score_90"] = home_goals
        else:
            out.at[idx, "home_goals"] = home_goals
            out.at[idx, "away_goals"] = away_goals
            out.at[idx, "home_goals_90"] = home_goals
            out.at[idx, "away_goals_90"] = away_goals
            out.at[idx, "home_score_90"] = home_goals
            out.at[idx, "away_score_90"] = away_goals
    return out


def _first_present(row: pd.Series | dict[str, Any], columns: list[str]) -> Any:
    for col in columns:
        if col not in row:
            continue
        value = row.get(col)
        if pd.isna(value) or str(value).strip() == "":
            continue
        return value
    return pd.NA


def _bulk_sync_ready_fixtures(
    fixtures: pd.DataFrame,
    now: pd.Timestamp,
    date_min: str | None,
    date_max: str | None,
    result_ready_delay_hours: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if fixtures is None or fixtures.empty:
        return pd.DataFrame(), []
    out = fixtures.copy()
    out["_kickoff_utc"] = pd.to_datetime(out.apply(_kickoff_utc, axis=1), errors="coerce", utc=True)
    if date_min:
        out = out.loc[out["_kickoff_utc"] >= pd.to_datetime(date_min, errors="coerce", utc=True)].copy()
    if date_max:
        max_ts = pd.to_datetime(date_max, errors="coerce", utc=True)
        if pd.notna(max_ts):
            out = out.loc[out["_kickoff_utc"] <= max_ts + pd.Timedelta(days=1)].copy()
    skipped_rows: list[dict[str, Any]] = []
    invalid = out.loc[out["_kickoff_utc"].isna()].copy()
    for _, row in invalid.iterrows():
        skipped_rows.append(_bulk_skipped_fixture(row, "invalid_kickoff", "Fixture kickoff could not be parsed."))
    out = out.loc[out["_kickoff_utc"].notna()].copy()
    unresolved = out.loc[out.apply(_fixture_has_unresolved_team, axis=1)].copy()
    for _, row in unresolved.iterrows():
        skipped_rows.append(_bulk_skipped_fixture(row, "unresolved_team", "Fixture still contains an unresolved bracket slot."))
    out = out.loc[~out.apply(_fixture_has_unresolved_team, axis=1)].copy()
    ready_at = out["_kickoff_utc"] + pd.Timedelta(hours=float(result_ready_delay_hours))
    not_ready = out.loc[ready_at > now].copy()
    for _, row in not_ready.iterrows():
        skipped_rows.append(_bulk_skipped_fixture(row, "not_ready", "Fixture has not reached kickoff plus result-ready delay."))
    ready = out.loc[ready_at <= now].copy()
    return ready.sort_values(["_kickoff_utc", "match_id"]).reset_index(drop=True), skipped_rows


def _fixture_has_unresolved_team(row: pd.Series) -> bool:
    home = str(row.get("home", "") or "").strip().lower()
    away = str(row.get("away", "") or "").strip().lower()
    unresolved_terms = ("winner ", "loser ", "winner match", "loser match", "tbd", "to be determined")
    return any(term in home for term in unresolved_terms) or any(term in away for term in unresolved_terms)


def _bulk_skipped_fixture(row: pd.Series, status: str, reason: str) -> dict[str, Any]:
    return {
        "fixture_id": str(row.get("match_id", "") or ""),
        "kickoff_utc": str(row.get("_kickoff_utc", "") or ""),
        "home": str(row.get("home", "") or ""),
        "away": str(row.get("away", "") or ""),
        "import_status": status,
        "matched_provider_id": "",
        "imported_score_90": "",
        "score_semantics": "",
        "reason": reason,
    }


def _completed_results_to_match_candidates(completed_results: pd.DataFrame | None) -> pd.DataFrame:
    if completed_results is None or completed_results.empty:
        return pd.DataFrame()
    df = completed_results.copy()
    for col in COMPLETED_RESULT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    out = pd.DataFrame(
        {
            "match_id": df["provider_match_id"].fillna("").astype(str),
            "provider_match_id": df["provider_match_id"].fillna("").astype(str),
            "provider": df["provider"].fillna("").astype(str),
            "date_utc": df["date_utc"],
            "competition": df["competition"],
            "group": df["group"],
            "home": df["home"],
            "away": df["away"],
            "home_goals": df["home_score_90"],
            "away_goals": df["away_score_90"],
            "home_goals_90": df["home_score_90"],
            "away_goals_90": df["away_score_90"],
            "score_source": df["score_source"],
            "score_semantics": df["score_semantics"],
            "provider_kickoff_utc": df["provider_kickoff_utc"],
            "result_source": df["source"],
            "last_updated": df["last_updated"],
        }
    )
    return out


def _results_ledger_to_completed_candidates(results_ledger: pd.DataFrame | None) -> pd.DataFrame:
    if results_ledger is None or results_ledger.empty:
        return pd.DataFrame()
    df = _ensure_columns(results_ledger, RESULTS_LEDGER_COLUMNS)
    out = pd.DataFrame(
        {
            "match_id": df["match_id"].fillna("").astype(str),
            "provider_match_id": df["match_id"].fillna("").astype(str),
            "provider": "results_ledger",
            "date_utc": df["date_utc"],
            "competition": df["competition"],
            "group": "",
            "home": df["home"],
            "away": df["away"],
            "home_goals": df["home_goals"],
            "away_goals": df["away_goals"],
            "home_goals_90": df["home_goals"],
            "away_goals_90": df["away_goals"],
            "score_source": "data/results_ledger.csv",
            "score_semantics": df["result_semantics"].fillna("90-minute regular time"),
            "provider_kickoff_utc": "",
            "result_source": df["result_source"],
            "last_updated": df["last_updated"],
        }
    )
    return out


def _combine_completed_candidate_frames(frames: list[pd.DataFrame | None]) -> pd.DataFrame:
    valid = [frame.copy() for frame in frames if frame is not None and not frame.empty]
    if not valid:
        return pd.DataFrame()
    combined = pd.concat(valid, ignore_index=True, sort=False)
    for col in ["match_id", "date_utc", "competition", "home", "away", "home_goals", "away_goals"]:
        if col not in combined.columns:
            combined[col] = pd.NA
    combined["_dedupe"] = combined.apply(
        lambda row: "|".join(
            [
                str(row.get("match_id", "") or ""),
                _date_label(row.get("date_utc")),
                _team_key(row.get("home", "")),
                _team_key(row.get("away", "")),
                str(int(coerce_float(row.get("home_goals"), 0))),
                str(int(coerce_float(row.get("away_goals"), 0))),
            ]
        ),
        axis=1,
    )
    return combined.drop_duplicates(subset=["_dedupe"], keep="first").drop(columns=["_dedupe"]).reset_index(drop=True)


def _ledger_result_row_from_candidate(fixture: pd.Series, candidate: pd.Series, result_source: str) -> dict[str, Any]:
    home_goals = coerce_float(_first_present(candidate, HOME_REGULAR_TIME_GOAL_COLUMNS), 0.0)
    away_goals = coerce_float(_first_present(candidate, AWAY_REGULAR_TIME_GOAL_COLUMNS), 0.0)
    total = home_goals + away_goals
    return {
        "match_id": str(fixture.get("match_id", "") or "").strip(),
        "date_utc": fixture.get("date_utc", candidate.get("date_utc", "")),
        "competition": fixture.get("competition", candidate.get("competition", "")),
        "home": fixture.get("home", candidate.get("home", "")),
        "away": fixture.get("away", candidate.get("away", "")),
        "home_goals": int(home_goals),
        "away_goals": int(away_goals),
        "actual_result": classify_actual_result(home_goals, away_goals),
        "total_goals": int(total),
        "btts_actual": int(home_goals > 0 and away_goals > 0),
        "over_0_5_actual": int(total > 0.5),
        "over_1_5_actual": int(total > 1.5),
        "over_2_5_actual": int(total > 2.5),
        "over_3_5_actual": int(total > 3.5),
        "actual_advancing_team": candidate.get("actual_advancing_team", pd.NA),
        "result_semantics": candidate.get("result_semantics", candidate.get("score_semantics", "90-minute regular time")),
        "evaluation_eligible_1x2": True,
        "had_extra_time": False,
        "had_penalties": False,
        "penalties_home": pd.NA,
        "penalties_away": pd.NA,
        "result_source": result_source,
        "last_updated": utc_now_iso(),
    }


def _existing_result_for_match_id(results: pd.DataFrame, match_id: str) -> pd.Series | None:
    if results is None or results.empty or "match_id" not in results.columns:
        return None
    matched = results.loc[results["match_id"].astype(str) == str(match_id)]
    if matched.empty:
        return None
    return matched.iloc[-1]


def _score_pair(row: pd.Series | dict[str, Any]) -> tuple[int, int]:
    return int(coerce_float(row.get("home_goals"), 0)), int(coerce_float(row.get("away_goals"), 0))


def _bulk_fixture_diagnostic(
    fixture: pd.Series,
    status: str,
    candidate: pd.Series | None = None,
    result_row: dict[str, Any] | None = None,
    reason: str = "",
    nearest: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    candidate = candidate if candidate is not None else pd.Series(dtype="object")
    result_row = result_row or {}
    home_goals = result_row.get("home_goals", "")
    away_goals = result_row.get("away_goals", "")
    score = f"{home_goals}-{away_goals}" if str(home_goals) != "" and str(away_goals) != "" else ""
    return {
        "fixture_id": str(fixture.get("match_id", "") or ""),
        "kickoff_utc": str(fixture.get("_kickoff_utc", "") or _kickoff_utc(fixture)),
        "home": str(fixture.get("home", "") or ""),
        "away": str(fixture.get("away", "") or ""),
        "import_status": status,
        "matched_provider_id": str(candidate.get("provider_match_id", "") or candidate.get("match_id", "") or ""),
        "imported_score_90": score,
        "score_semantics": str(candidate.get("score_semantics", "") or "90-minute regular time") if not candidate.empty else "",
        "reason": reason,
        "nearest_candidates": nearest or [],
    }


def _bulk_sync_diagnostics(
    status: str,
    provider_diagnostics: dict[str, Any],
    fixture_rows: list[dict[str, Any]],
    skipped_rows: list[dict[str, Any]],
    total_fixtures: int,
    past_fixtures: int,
    existing_rows: int,
    final_rows: int,
    provider_rows: int = 0,
    provider_candidate_rows: int = 0,
    fallback_candidate_rows: int = 0,
    manual_candidate_rows: int = 0,
    manual_path_checked: str = str(MANUAL_MISSING_RESULTS_PATH),
    existing_candidate_rows: int = 0,
    date_window: str = "",
) -> dict[str, Any]:
    statuses = [row.get("import_status", "") for row in fixture_rows]
    imported = sum(1 for value in statuses if value in {"imported", "imported_from_existing_result", "updated"})
    summary = {
        "status": status,
        "total_fixtures": int(total_fixtures),
        "total_past_fixtures": int(past_fixtures),
        "provider_completed_rows_fetched": int(provider_diagnostics.get("normalized_rows_fetched", provider_rows) or provider_rows),
        "completed_rows_loaded": int(provider_diagnostics.get("completed_rows_loaded", provider_rows) or provider_rows),
        "provider_candidate_rows": int(provider_candidate_rows),
        "fallback_candidate_rows": int(fallback_candidate_rows),
        "manual_candidate_rows": int(manual_candidate_rows),
        "manual_path_checked": str(manual_path_checked or MANUAL_MISSING_RESULTS_PATH),
        "existing_candidate_rows": int(existing_candidate_rows),
        "matched_fixtures": sum(1 for value in statuses if value in {"imported", "imported_from_existing_result", "already_present", "updated", "conflict_needs_review"}),
        "imported_or_updated_rows": imported,
        "already_present_rows": statuses.count("already_present"),
        "conflict_rows": statuses.count("conflict_needs_review"),
        "unmatched_fixtures": statuses.count("unmatched"),
        "skipped_fixtures": len(skipped_rows),
        "existing_ledger_rows": int(existing_rows),
        "final_ledger_rows": int(final_rows),
        "provider_checked": provider_diagnostics.get("provider_checked", ""),
        "date_window_checked": provider_diagnostics.get("date_window_checked", date_window) or date_window,
        "cache_path_checked": provider_diagnostics.get("cache_path_checked", ""),
        "local_path_checked": provider_diagnostics.get("local_path_checked", ""),
        "last_refresh_timestamp": provider_diagnostics.get("last_updated", ""),
        "completed_source_status": provider_diagnostics.get("status", ""),
    }
    summary["fixture_diagnostics"] = fixture_rows + skipped_rows
    summary["unmatched_rows"] = [row for row in fixture_rows if row.get("import_status") == "unmatched"]
    summary["conflict_rows_detail"] = [row for row in fixture_rows if row.get("import_status") == "conflict_needs_review"]
    return summary


def _app_launch_sync_diagnostics(
    diagnostics: dict[str, Any],
    results: pd.DataFrame,
    result_ready_delay_minutes: float,
    force_refresh: bool,
    path: Path,
) -> dict[str, Any]:
    out = dict(diagnostics or {})
    fixture_rows = out.get("fixture_diagnostics", []) or []
    skipped_not_ready = sum(1 for row in fixture_rows if row.get("import_status") == "not_ready")
    imported = int(out.get("imported_or_updated_rows", out.get("rows_imported", 0)) or 0)
    already_present = int(out.get("already_present_rows", 0) or 0)
    unmatched = int(out.get("unmatched_fixtures", 0) or 0)
    conflicts = int(out.get("conflict_rows", 0) or 0)
    skipped = int(out.get("skipped_fixtures", 0) or 0)
    provider_rows = int(out.get("completed_rows_loaded", out.get("provider_completed_rows_fetched", 0)) or 0)
    out.update(
        {
            "auto_sync_ran": True,
            "force_refresh": bool(force_refresh),
            "result_ready_delay_minutes": float(result_ready_delay_minutes),
            "provider_rows_loaded": provider_rows,
            "imported_rows": imported,
            "already_present_rows": already_present,
            "unmatched_rows_count": unmatched,
            "conflict_rows": conflicts,
            "skipped_rows": skipped,
            "skipped_not_ready_rows": int(skipped_not_ready),
            "results_ledger_final_row_count": int(len(results)),
            "final_ledger_rows": int(len(results)),
            "output_path": _display_path(path),
        }
    )
    return out


def _normalise_fixtures(fixtures_df: pd.DataFrame | None) -> pd.DataFrame:
    required = [
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
        "neutral_site",
    ]
    out = _ensure_columns(fixtures_df, required)
    if out.empty:
        return out
    out = out.dropna(subset=["date_utc", "home", "away"]).copy()
    out["home"] = out["home"].astype(str).str.strip()
    out["away"] = out["away"].astype(str).str.strip()
    out = out.loc[(out["home"] != "") & (out["away"] != "")]
    if "neutral_site" in out.columns:
        out["neutral_site"] = out["neutral_site"].map(coerce_bool).astype(int)
    return out.reset_index(drop=True)


def _ensure_columns(df: pd.DataFrame | None, columns: list[str]) -> pd.DataFrame:
    out = df.copy() if df is not None else pd.DataFrame(columns=columns)
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out[columns].copy()


def _over_probability(matrix: pd.DataFrame | None, threshold_goals: int) -> float:
    if matrix is None or matrix.empty:
        return 0.0
    mat = matrix.copy()
    if not {"home_goals", "away_goals", "prob"}.issubset(mat.columns):
        return 0.0
    total = pd.to_numeric(mat["home_goals"], errors="coerce") + pd.to_numeric(mat["away_goals"], errors="coerce")
    return float(pd.to_numeric(mat["prob"], errors="coerce").fillna(0.0).loc[total > float(threshold_goals)].sum())


def _market_1x2_probabilities(market_odds_df: pd.DataFrame, match_id: Any, home: Any, away: Any) -> tuple[Any, Any, Any]:
    odds = market_odds_df.copy() if market_odds_df is not None else pd.DataFrame()
    if odds.empty or not {"match_id", "market", "selection", "odds"}.issubset(odds.columns):
        return pd.NA, pd.NA, pd.NA
    selected = odds.loc[
        (odds["match_id"].astype(str) == str(match_id))
        & (odds["market"].astype(str).str.lower() == "1x2")
    ].copy()
    if selected.empty:
        return pd.NA, pd.NA, pd.NA
    lookup = {
        str(row.get("selection", "")).strip().lower(): coerce_float(row.get("odds"), float("nan"))
        for _, row in selected.iterrows()
    }
    decimal_odds = [
        lookup.get(str(home).strip().lower(), float("nan")),
        lookup.get("draw", float("nan")),
        lookup.get(str(away).strip().lower(), float("nan")),
    ]
    if not all(pd.notna(value) and value > 1.0 for value in decimal_odds):
        return pd.NA, pd.NA, pd.NA
    implied = [1.0 / value for value in decimal_odds]
    total = sum(implied)
    if total <= 0:
        return pd.NA, pd.NA, pd.NA
    return tuple(value / total for value in implied)


def _kickoff_utc(row: pd.Series) -> str:
    date_value = str(row.get("date_utc", "") or "").strip()
    time_value = str(row.get("time_utc", "") or "00:00").strip() or "00:00"
    ts = pd.to_datetime(f"{date_value} {time_value}", errors="coerce", utc=True)
    if pd.isna(ts):
        ts = pd.to_datetime(date_value, errors="coerce", utc=True)
    return "" if pd.isna(ts) else ts.replace(microsecond=0).isoformat()


def _fallback_match_id(row: Any) -> str:
    series = pd.Series(row)
    return "|".join(
        [
            str(series.get("date_utc", "")),
            str(series.get("home", "")),
            str(series.get("away", "")),
        ]
    ).strip("|")


def _prediction_id(snapshot_utc: str, match_id: str, model_version: str, parameter_set_id: str) -> str:
    raw = "|".join([snapshot_utc, match_id, model_version, parameter_set_id])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _skip_row(row: Any, reason_code: str, reason: str) -> dict[str, Any]:
    series = pd.Series(row)
    return {
        "match_id": str(series.get("match_id", "") or ""),
        "date_utc": str(series.get("date_utc", "") or ""),
        "time_utc": str(series.get("time_utc", "") or ""),
        "home": str(series.get("home", "") or ""),
        "away": str(series.get("away", "") or ""),
        "reason_code": reason_code,
        "reason": reason,
    }


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(DATA_DIR.parent))
    except ValueError:
        return str(path)


def _market_snapshot_available(market_odds_df: pd.DataFrame, match_id: str) -> bool:
    if market_odds_df is None or market_odds_df.empty or "match_id" not in market_odds_df.columns:
        return False
    return bool((market_odds_df["match_id"].astype(str) == str(match_id)).any())


def _before_kickoff(snapshot_utc: str, kickoff_utc: str) -> bool:
    snapshot = pd.to_datetime(snapshot_utc, errors="coerce", utc=True)
    kickoff = pd.to_datetime(kickoff_utc, errors="coerce", utc=True)
    if pd.isna(snapshot) or pd.isna(kickoff):
        return False
    return bool(snapshot < kickoff)
