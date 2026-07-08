from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.calibration import (
    OUTCOMES,
    PROBABILITY_BIN_LABELS,
    PROBABILITY_BINS,
    brier_1x2,
    gap_band,
    market_type_semantics,
    safe_log_loss,
)
from src.confidence import (
    HIGH_CONFIDENCE_MIN_SAMPLE,
    calibration_status_label,
    model_market_benchmark_status,
    score_forecast_confidence,
)
from src.match_identity import (
    align_result_to_prediction,
    build_result_fixture_crosswalk,
    build_match_key,
    normalize_match_date,
    resolve_prediction_to_result,
    validate_results_ledger_semantics,
)
from src.prediction_ledger import select_latest_valid_snapshots
from src.utils import coerce_float, today_iso


EVALUATION_DATASET_COLUMNS = [
    "prediction_id",
    "prediction_source",
    "generated_at_utc",
    "match_id",
    "kickoff_utc",
    "home_team",
    "away_team",
    "competition",
    "round",
    "venue",
    "model_version",
    "home_win_prob_raw",
    "draw_prob_raw",
    "away_win_prob_raw",
    "home_win_prob_calibrated",
    "draw_prob_calibrated",
    "away_win_prob_calibrated",
    "home_win_prob_market",
    "draw_prob_market",
    "away_win_prob_market",
    "behavior_home_prob",
    "behavior_draw_prob",
    "behavior_away_prob",
    "behavior_disagreement_pp",
    "market_disagreement_pp",
    "model_confidence",
    "confidence_score",
    "confidence_label",
    "confidence_reason",
    "confidence_flags",
    "market_join_status",
    "mapping_confidence",
    "data_quality_status",
    "actual_home_goals_90",
    "actual_away_goals_90",
    "actual_result_1x2",
    "advancing_team",
    "result_semantics",
    "result_match_id_original",
    "fixture_match_id",
    "result_join_method",
    "result_join_confidence",
    "result_join_reason",
    "team_order_status",
    "actual_result_from_prediction_perspective",
    "top_pick_raw",
    "top_pick_calibrated",
    "probability_assigned_to_actual_raw",
    "probability_assigned_to_actual_calibrated",
    "brier_raw",
    "brier_calibrated",
    "log_loss_raw",
    "log_loss_calibrated",
    "market_probability_assigned_to_actual",
    "brier_market",
    "log_loss_market",
    "raw_model_was_correct",
    "calibrated_model_was_correct",
    "market_was_closer_than_model",
    "notes",
]

RESULT_PREDICTION_JOIN_AUDIT_COLUMNS = [
    "prediction_id",
    "prediction_source",
    "generated_at_utc",
    "match_id",
    "kickoff_utc",
    "home_team",
    "away_team",
    "competition",
    "round",
    "prediction_before_kickoff",
    "valid_probabilities",
    "resolved_result_match_id",
    "fixture_match_id",
    "result_join_method",
    "result_join_confidence",
    "result_join_reason",
    "home_away_order",
    "team_order_status",
    "result_semantics",
    "evaluation_eligible_1x2",
    "join_status",
    "usable_for_evaluation",
    "exclusion_reason",
]

MODEL_COMPARISON_METHODS = [
    "raw_model",
    "temperature_scaling",
    "shrinkage_to_base_rates",
    "market_prior_blend",
    "behavior_gated_shrinkage",
    "confidence_gated_shrinkage",
    "combined_conservative_calibrated_model",
]

RECENT_DIAGNOSTIC_CASES = [
    {
        "home": "Mexico",
        "away": "Ecuador",
        "actual_home_goals_90": 2,
        "actual_away_goals_90": 0,
        "actual_result_1x2": "home_win",
        "advancing_team": "Mexico",
        "behavior_disagreement_pp": 32.8,
        "note": "Milestone 2 diagnostic case: model strongly favored Ecuador; behavior layer warned against that side.",
    },
    {
        "home": "Germany",
        "away": "Paraguay",
        "actual_home_goals_90": 1,
        "actual_away_goals_90": 1,
        "actual_result_1x2": "draw",
        "advancing_team": "Paraguay",
        "note": "Milestone 2 diagnostic case: penalty advancement is stored separately from the 90-minute draw.",
    },
    {
        "home": "Colombia",
        "away": "Portugal",
        "actual_home_goals_90": 0,
        "actual_away_goals_90": 0,
        "actual_result_1x2": "draw",
        "advancing_team": "",
        "note": "Milestone 2 diagnostic case: strong favorite call missed a 90-minute draw.",
    },
    {
        "home": "Netherlands",
        "away": "Morocco",
        "actual_home_goals_90": 1,
        "actual_away_goals_90": 1,
        "actual_result_1x2": "draw",
        "advancing_team": "Morocco",
        "note": "Milestone 2 diagnostic case: advancement is separate from the 90-minute draw.",
    },
    {
        "home": "France",
        "away": "Sweden",
        "actual_home_goals_90": pd.NA,
        "actual_away_goals_90": pd.NA,
        "actual_result_1x2": "home_win",
        "advancing_team": "France",
        "note": "Milestone 2 diagnostic case: exact score not present in local result ledger.",
    },
    {
        "home": "England",
        "away": "Panama",
        "actual_home_goals_90": pd.NA,
        "actual_away_goals_90": pd.NA,
        "actual_result_1x2": "home_win",
        "advancing_team": "England",
        "note": "Milestone 2 diagnostic case: no matching local pre-match snapshot found in current ledgers.",
    },
    {
        "home": "Uruguay",
        "away": "Spain",
        "actual_home_goals_90": pd.NA,
        "actual_away_goals_90": pd.NA,
        "actual_result_1x2": "away_win",
        "advancing_team": "Spain",
        "note": "Milestone 2 diagnostic case from prediction_log; exact score not present in local result ledger.",
    },
    {
        "home": "South Africa",
        "away": "Canada",
        "actual_home_goals_90": pd.NA,
        "actual_away_goals_90": pd.NA,
        "actual_result_1x2": "away_win",
        "advancing_team": "Canada",
        "note": "Milestone 2 diagnostic case: exact score not present in local result ledger.",
    },
    {
        "home": "Brazil",
        "away": "Japan",
        "actual_home_goals_90": 1,
        "actual_away_goals_90": 0,
        "actual_result_1x2": "home_win",
        "advancing_team": "Brazil",
        "note": "Local result ledger contains this match.",
    },
    {
        "home": "Ivory Coast",
        "away": "Norway",
        "actual_home_goals_90": pd.NA,
        "actual_away_goals_90": pd.NA,
        "actual_result_1x2": "away_win",
        "advancing_team": "Norway",
        "note": "Milestone 2 diagnostic case: exact score not present in local result ledger.",
    },
]


def build_formal_evaluation_dataset(
    prediction_ledger_df: pd.DataFrame | None,
    results_ledger_df: pd.DataFrame | None,
    *,
    completed_results_df: pd.DataFrame | None = None,
    fixtures_df: pd.DataFrame | None = None,
    result_fixture_crosswalk_df: pd.DataFrame | None = None,
    prediction_log_df: pd.DataFrame | None = None,
    market_odds_df: pd.DataFrame | None = None,
    min_calibration_sample: int = HIGH_CONFIDENCE_MIN_SAMPLE,
    latest_snapshot_only: bool = True,
) -> pd.DataFrame:
    predictions = _prediction_rows(
        prediction_ledger_df,
        prediction_log_df,
        market_odds_df,
        latest_snapshot_only=latest_snapshot_only,
    )
    results = _result_rows(results_ledger_df, completed_results_df)
    crosswalk = _evaluation_crosswalk(result_fixture_crosswalk_df, results_ledger_df, fixtures_df)
    if predictions.empty or results.empty:
        return pd.DataFrame(columns=EVALUATION_DATASET_COLUMNS)

    rows: list[dict[str, Any]] = []
    calibration_status = calibration_status_label(0)
    for _, prediction in predictions.iterrows():
        if not _valid_prediction_probabilities(prediction):
            continue
        resolution = resolve_prediction_to_result(prediction, results, fixtures_df, crosswalk)
        result = resolution.get("result")
        if result is None:
            continue
        result = align_result_to_prediction(result, str(resolution.get("home_away_order", "same")))
        if not bool(result.get("evaluation_eligible_1x2", True)):
            continue
        scope = market_type_semantics("1X2", str(result.get("result_semantics", "")))
        if scope != "90-minute":
            continue
        actual = str(result.get("actual_result_1x2", "") or "")
        if actual not in OUTCOMES:
            continue
        row = _scored_dataset_row(prediction, result, calibration_status, min_calibration_sample)
        row["_resolved_result_match_id"] = str(resolution.get("resolved_result_match_id", "") or "")
        row["result_match_id_original"] = row["_resolved_result_match_id"]
        row["fixture_match_id"] = str(resolution.get("fixture_match_id", "") or "")
        row["result_join_method"] = str(resolution.get("result_join_method", "") or "")
        row["result_join_confidence"] = resolution.get("result_join_confidence", pd.NA)
        row["result_join_reason"] = str(resolution.get("result_join_reason", "") or "")
        row["team_order_status"] = str(resolution.get("team_order_status", resolution.get("home_away_order", "same")) or "same")
        row["actual_result_from_prediction_perspective"] = actual
        row["notes"] = (
            f"{row.get('notes', '')}; result_join_method={resolution.get('result_join_method', '')}; "
            f"result_join_confidence={resolution.get('result_join_confidence', '')}"
        )
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=EVALUATION_DATASET_COLUMNS)
    out = pd.DataFrame(rows)
    if latest_snapshot_only and not out.empty:
        out["_generated_ts"] = pd.to_datetime(out["generated_at_utc"], errors="coerce", utc=True)
        group_key = out["_resolved_result_match_id"].where(
            out["_resolved_result_match_id"].astype(str).str.strip().ne(""),
            out.apply(lambda row: "|".join(str(row.get(col, "")) for col in ["match_id", "home_team", "away_team", "kickoff_utc"]), axis=1),
        )
        out["_latest_group_key"] = group_key
        out = (
            out.sort_values(["_latest_group_key", "_generated_ts", "prediction_id"])
            .drop_duplicates(subset=["_latest_group_key"], keep="last")
            .reset_index(drop=True)
        )
    return out[EVALUATION_DATASET_COLUMNS].copy()


def build_prediction_source_dataset(
    prediction_ledger_df: pd.DataFrame | None,
    prediction_log_df: pd.DataFrame | None = None,
    market_odds_df: pd.DataFrame | None = None,
    latest_snapshot_only: bool = True,
) -> pd.DataFrame:
    """Return normalized pre-kickoff prediction rows before result joining."""
    return _prediction_rows(
        prediction_ledger_df,
        prediction_log_df,
        market_odds_df,
        latest_snapshot_only=latest_snapshot_only,
    )


def build_result_prediction_join_audit(
    prediction_ledger_df: pd.DataFrame | None,
    results_ledger_df: pd.DataFrame | None,
    *,
    completed_results_df: pd.DataFrame | None = None,
    fixtures_df: pd.DataFrame | None = None,
    result_fixture_crosswalk_df: pd.DataFrame | None = None,
    prediction_log_df: pd.DataFrame | None = None,
    market_odds_df: pd.DataFrame | None = None,
    now_utc: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    predictions = _prediction_rows(
        prediction_ledger_df,
        prediction_log_df,
        market_odds_df,
        filter_pre_kickoff=False,
        latest_snapshot_only=False,
    )
    results = _result_rows(results_ledger_df, completed_results_df)
    crosswalk = _evaluation_crosswalk(result_fixture_crosswalk_df, results_ledger_df, fixtures_df)
    if predictions.empty:
        return pd.DataFrame(columns=RESULT_PREDICTION_JOIN_AUDIT_COLUMNS)

    now = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    rows: list[dict[str, Any]] = []
    for _, prediction in predictions.iterrows():
        generated_ts = pd.to_datetime(prediction.get("generated_at_utc"), errors="coerce", utc=True)
        kickoff_ts = pd.to_datetime(prediction.get("kickoff_utc"), errors="coerce", utc=True)
        resolution = resolve_prediction_to_result(prediction, results, fixtures_df, crosswalk)
        result = resolution.get("result")
        result_found = result is not None
        aligned = align_result_to_prediction(result, str(resolution.get("home_away_order", "same"))) if result_found else pd.Series(dtype="object")
        valid_probs = _valid_prediction_probabilities(prediction)
        status, usable, reason = _join_audit_status(
            prediction,
            aligned,
            resolution,
            generated_ts,
            kickoff_ts,
            now,
            valid_probs,
        )
        rows.append(
            {
                "prediction_id": prediction.get("prediction_id", ""),
                "prediction_source": prediction.get("prediction_source", ""),
                "generated_at_utc": prediction.get("generated_at_utc", ""),
                "match_id": prediction.get("match_id", ""),
                "kickoff_utc": prediction.get("kickoff_utc", ""),
                "home_team": prediction.get("home_team", ""),
                "away_team": prediction.get("away_team", ""),
                "competition": prediction.get("competition", ""),
                "round": prediction.get("round", ""),
                "prediction_before_kickoff": bool(pd.notna(generated_ts) and pd.notna(kickoff_ts) and generated_ts < kickoff_ts),
                "valid_probabilities": valid_probs,
                "resolved_result_match_id": resolution.get("resolved_result_match_id", ""),
                "fixture_match_id": resolution.get("fixture_match_id", ""),
                "result_join_method": resolution.get("result_join_method", ""),
                "result_join_confidence": resolution.get("result_join_confidence", 0.0),
                "result_join_reason": resolution.get("result_join_reason", ""),
                "home_away_order": resolution.get("home_away_order", "same"),
                "team_order_status": resolution.get("team_order_status", resolution.get("home_away_order", "same")),
                "result_semantics": aligned.get("result_semantics", ""),
                "evaluation_eligible_1x2": aligned.get("evaluation_eligible_1x2", False) if result_found else False,
                "join_status": status,
                "usable_for_evaluation": usable,
                "exclusion_reason": reason,
            }
        )
    audit = pd.DataFrame(rows, columns=RESULT_PREDICTION_JOIN_AUDIT_COLUMNS)
    return _mark_duplicate_unselected_snapshots(audit)


def evaluation_coverage_summary(audit_df: pd.DataFrame | None, previous_usable_rows: int | None = None) -> pd.DataFrame:
    audit = audit_df.copy() if audit_df is not None else pd.DataFrame(columns=RESULT_PREDICTION_JOIN_AUDIT_COLUMNS)
    rows = [
        {"metric": "prediction_snapshots", "value": int(len(audit))},
        {"metric": "unique_prediction_match_ids", "value": int(audit["match_id"].astype(str).nunique()) if not audit.empty else 0},
        {"metric": "exact_match_id_joins_possible", "value": int((audit["result_join_method"] == "exact_match_id").sum()) if not audit.empty else 0},
        {"metric": "schedule_bridge_joins_possible", "value": int((audit["result_join_method"] == "result_fixture_crosswalk").sum()) if not audit.empty else 0},
        {"metric": "fixture_bridge_joins_possible", "value": int((audit["result_join_method"] == "fixture_bridge").sum()) if not audit.empty else 0},
        {"metric": "normalized_team_date_joins_possible", "value": int(audit["result_join_method"].isin(["normalized_team_date", "fuzzy_team_date", "symmetric_team_date"]).sum()) if not audit.empty else 0},
        {"metric": "home_away_order_mismatch_matches", "value": int((audit["home_away_order"] == "reversed").sum()) if not audit.empty else 0},
        {"metric": "usable_evaluated_predictions_after_latest_selection", "value": int(audit["usable_for_evaluation"].astype(bool).sum()) if not audit.empty else 0},
        {"metric": "excluded_predictions", "value": int((~audit["usable_for_evaluation"].astype(bool)).sum()) if not audit.empty else 0},
    ]
    if previous_usable_rows is not None:
        rows.insert(0, {"metric": "usable_evaluated_predictions_before_fix", "value": int(previous_usable_rows)})
    if audit.empty:
        return pd.DataFrame(rows)
    status_counts = audit["join_status"].value_counts(dropna=False).rename_axis("join_status").reset_index(name="value")
    status_counts["metric"] = "join_status:" + status_counts["join_status"].astype(str)
    rows.extend(status_counts[["metric", "value"]].to_dict("records"))
    return pd.DataFrame(rows)


def render_result_prediction_join_audit(
    audit_df: pd.DataFrame,
    *,
    prediction_rows: int,
    unique_prediction_match_ids: int,
    result_rows: int,
    unique_result_match_ids: int,
    completed_result_rows: int,
    previous_usable_rows: int | None = None,
) -> str:
    audit = audit_df.copy() if audit_df is not None else pd.DataFrame(columns=RESULT_PREDICTION_JOIN_AUDIT_COLUMNS)
    usable = int(audit["usable_for_evaluation"].astype(bool).sum()) if not audit.empty else 0
    exact = int((audit["result_join_method"] == "exact_match_id").sum()) if not audit.empty else 0
    schedule_bridge = int((audit["result_join_method"] == "result_fixture_crosswalk").sum()) if not audit.empty else 0
    fixture_bridge = int((audit["result_join_method"] == "fixture_bridge").sum()) if not audit.empty else 0
    normalized = int(audit["result_join_method"].isin(["normalized_team_date", "fuzzy_team_date", "symmetric_team_date"]).sum()) if not audit.empty else 0
    top_exclusions = _top_exclusion_markdown(audit)
    before = "Unavailable" if previous_usable_rows is None else str(int(previous_usable_rows))
    lines = [
        f"# Result/Prediction Join Audit - {today_iso()}",
        "",
        "This audit reconciles saved pre-match prediction snapshots with the existing completed-results ledger. It does not scrape PDFs or build a separate result-ingestion path.",
        "",
        "## Coverage",
        "",
        f"- Prediction snapshots: `{prediction_rows}`",
        f"- Unique prediction match_ids: `{unique_prediction_match_ids}`",
        f"- Result ledger rows: `{result_rows}`",
        f"- Unique result match_ids: `{unique_result_match_ids}`",
        f"- Completed-results source rows: `{completed_result_rows}`",
        f"- Exact match_id joins possible: `{exact}`",
        f"- Schedule bridge joins possible: `{schedule_bridge}`",
        f"- Fixture-bridge joins possible: `{fixture_bridge}`",
        f"- Normalized team/date joins possible: `{normalized}`",
        f"- Usable evaluated predictions before this fix: `{before}`",
        f"- Usable evaluated predictions after latest-snapshot selection: `{usable}`",
        "",
        "## Exclusions",
        "",
        top_exclusions,
        "",
        "## Do we still need PDF backfill?",
        "",
        _pdf_backfill_sentence(usable),
        "",
        "PDF import remains secondary. Structured CSV joins must be correct first, and PDF timestamps need separate provenance review before they can support calibration claims.",
    ]
    return "\n".join(lines)


def render_evaluation_coverage_audit(audit_df: pd.DataFrame, previous_usable_rows: int | None = None) -> str:
    audit = audit_df.copy() if audit_df is not None else pd.DataFrame(columns=RESULT_PREDICTION_JOIN_AUDIT_COLUMNS)
    summary = evaluation_coverage_summary(audit, previous_usable_rows)
    usable = int(audit["usable_for_evaluation"].astype(bool).sum()) if not audit.empty else 0
    lines = [
        f"# Evaluation Coverage Audit - {today_iso()}",
        "",
        "Evaluation uses the latest valid pre-kickoff prediction snapshot per resolved completed match and scores only 90-minute 1X2 outcomes.",
        "",
        _to_markdown(_printable(summary)),
        "",
        f"The previous Milestone 2 dataset had `{previous_usable_rows if previous_usable_rows is not None else 'Unavailable'}` usable row(s). The regenerated dataset has `{usable}` usable row(s) after duplicate-snapshot selection.",
        "",
        "Top exclusion reasons:",
        "",
        _top_exclusion_markdown(audit),
    ]
    return "\n".join(lines)


def calibration_summary(evaluation_df: pd.DataFrame | None) -> pd.DataFrame:
    df = _valid_scored_rows(evaluation_df)
    columns = [
        "model",
        "n",
        "top_pick_accuracy",
        "brier_score",
        "multiclass_log_loss",
        "expected_calibration_error",
        "mean_actual_result_probability",
        "calibration_slope",
        "calibration_intercept",
        "status",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)
    rows = [_metric_row(df, "raw_model", ["home_win_prob_raw", "draw_prob_raw", "away_win_prob_raw"])]
    if df[["home_win_prob_calibrated", "draw_prob_calibrated", "away_win_prob_calibrated"]].notna().all(axis=1).any():
        rows.append(_metric_row(df, "calibrated_model", ["home_win_prob_calibrated", "draw_prob_calibrated", "away_win_prob_calibrated"]))
    if df[["home_win_prob_market", "draw_prob_market", "away_win_prob_market"]].notna().all(axis=1).any():
        rows.append(_metric_row(df, "market_implied", ["home_win_prob_market", "draw_prob_market", "away_win_prob_market"]))
    return pd.DataFrame(rows, columns=columns)


def calibration_by_bin(evaluation_df: pd.DataFrame | None) -> pd.DataFrame:
    df = _valid_scored_rows(evaluation_df)
    rows: list[dict[str, Any]] = []
    for outcome, col in zip(OUTCOMES, ["home_win_prob_raw", "draw_prob_raw", "away_win_prob_raw"]):
        if df.empty or col not in df.columns:
            continue
        frame = pd.DataFrame(
            {
                "probability": pd.to_numeric(df[col], errors="coerce"),
                "actual": (df["actual_result_1x2"].astype(str) == outcome).astype(float),
            }
        ).dropna()
        if frame.empty:
            continue
        frame["probability_bin"] = pd.cut(
            frame["probability"].clip(0, 1),
            PROBABILITY_BINS,
            labels=PROBABILITY_BIN_LABELS,
            include_lowest=True,
        )
        grouped = frame.groupby("probability_bin", observed=False).agg(
            n=("actual", "size"),
            avg_predicted_probability=("probability", "mean"),
            observed_frequency=("actual", "mean"),
        )
        for bucket, row in grouped.iterrows():
            if int(row["n"]) <= 0:
                continue
            rows.append(
                {
                    "model": "raw_model",
                    "outcome": outcome,
                    "probability_bin": str(bucket),
                    "n": int(row["n"]),
                    "avg_predicted_probability": float(row["avg_predicted_probability"]),
                    "observed_frequency": float(row["observed_frequency"]),
                    "calibration_error": abs(float(row["avg_predicted_probability"]) - float(row["observed_frequency"])),
                }
            )
    return pd.DataFrame(rows)


def performance_by_dimension(evaluation_df: pd.DataFrame | None, dimension: str) -> pd.DataFrame:
    df = _valid_scored_rows(evaluation_df)
    if df.empty or dimension not in df.columns:
        return pd.DataFrame(columns=[dimension, "n", "top_pick_accuracy", "brier_score", "multiclass_log_loss"])
    rows = []
    for value, group in df.groupby(dimension, dropna=False):
        metrics = calibration_summary(group)
        raw = metrics.loc[metrics["model"] == "raw_model"]
        if raw.empty:
            continue
        rec = raw.iloc[0].to_dict()
        rows.append(
            {
                dimension: value,
                "n": int(rec["n"]),
                "top_pick_accuracy": rec["top_pick_accuracy"],
                "brier_score": rec["brier_score"],
                "multiclass_log_loss": rec["multiclass_log_loss"],
            }
        )
    return pd.DataFrame(rows)


def calibration_by_match_context(evaluation_df: pd.DataFrame | None) -> pd.DataFrame:
    df = _valid_scored_rows(evaluation_df)
    if df.empty:
        return pd.DataFrame()
    working = df.copy()
    working["favorite_probability"] = working[["home_win_prob_raw", "away_win_prob_raw"]].max(axis=1)
    working["favorite_probability_band"] = working["favorite_probability"].map(_favorite_probability_band)
    working["draw_probability_band"] = working["draw_prob_raw"].map(_probability_band)
    working["behavior_disagreement_band"] = working["behavior_disagreement_pp"].map(_pp_gap_band)
    working["market_disagreement_band"] = working["market_disagreement_pp"].map(_pp_gap_band)
    working["match_context"] = working["round"].map(lambda value: "knockout" if "round of" in str(value).lower() or "final" in str(value).lower() else "group_stage")
    frames = []
    for dimension in [
        "favorite_probability_band",
        "draw_probability_band",
        "behavior_disagreement_band",
        "market_disagreement_band",
        "market_join_status",
        "match_context",
    ]:
        part = performance_by_dimension(working, dimension)
        if not part.empty:
            part.insert(0, "dimension", dimension)
            part = part.rename(columns={dimension: "segment"})
            frames.append(part)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def worst_model_misses(evaluation_df: pd.DataFrame | None, limit: int = 20) -> pd.DataFrame:
    df = _valid_scored_rows(evaluation_df)
    if df.empty:
        return pd.DataFrame()
    return (
        df.sort_values(["log_loss_raw", "brier_raw"], ascending=[False, False])
        .head(limit)
        .reset_index(drop=True)
    )


def walk_forward_calibration_comparison(
    evaluation_df: pd.DataFrame | None,
    min_train: int = HIGH_CONFIDENCE_MIN_SAMPLE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _valid_scored_rows(evaluation_df)
    columns = [
        "model",
        "n",
        "brier_score",
        "log_loss",
        "top_pick_accuracy",
        "expected_calibration_error",
        "brier_delta_vs_raw",
        "log_loss_delta_vs_raw",
        "production_status",
        "notes",
    ]
    prediction_columns = [
        "model",
        "prediction_id",
        "match_id",
        "generated_at_utc",
        "kickoff_utc",
        "home_team",
        "away_team",
        "actual_result_1x2",
        "home_prob",
        "draw_prob",
        "away_prob",
        "brier_score",
        "log_loss",
        "top_pick_correct",
        "probability_assigned_to_actual",
    ]
    if len(df) <= min_train:
        rows = [
            {
                "model": method,
                "n": 0,
                "brier_score": pd.NA,
                "log_loss": pd.NA,
                "top_pick_accuracy": pd.NA,
                "expected_calibration_error": pd.NA,
                "brier_delta_vs_raw": pd.NA,
                "log_loss_delta_vs_raw": pd.NA,
                "production_status": "not_promoted_sample_too_small",
                "notes": f"Need more than {min_train} completed pre-kickoff predictions for walk-forward scoring.",
            }
            for method in MODEL_COMPARISON_METHODS
        ]
        return pd.DataFrame(rows, columns=columns), pd.DataFrame(columns=prediction_columns)

    working = df.copy()
    working["_kickoff_ts"] = pd.to_datetime(working["kickoff_utc"], errors="coerce", utc=True)
    working["_generated_ts"] = pd.to_datetime(working["generated_at_utc"], errors="coerce", utc=True)
    working = working.sort_values(["_kickoff_ts", "_generated_ts", "prediction_id"]).reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for idx in range(min_train, len(working)):
        train = working.iloc[:idx].copy()
        row = working.iloc[idx]
        raw = _row_probabilities(row, "raw")
        base = _base_rates(train)
        fitted = {
            "temperature": _fit_temperature(train),
            "shrinkage": _fit_shrinkage(train, base),
            "market_weight": _fit_market_weight(train),
            "behavior_gate": _fit_behavior_gate(train, base),
            "confidence_shrinkage": _fit_confidence_shrinkage(train, base),
        }
        variants = _walk_forward_variants(row, raw, base, fitted)
        for model_name, probs in variants.items():
            actual = str(row.get("actual_result_1x2", ""))
            rows.append(
                {
                    "model": model_name,
                    "prediction_id": row.get("prediction_id", ""),
                    "match_id": row.get("match_id", ""),
                    "generated_at_utc": row.get("generated_at_utc", ""),
                    "kickoff_utc": row.get("kickoff_utc", ""),
                    "home_team": row.get("home_team", ""),
                    "away_team": row.get("away_team", ""),
                    "actual_result_1x2": actual,
                    "home_prob": probs[0],
                    "draw_prob": probs[1],
                    "away_prob": probs[2],
                    "brier_score": brier_1x2(probs, actual),
                    "log_loss": safe_log_loss(_prob_for_actual(probs, actual)),
                    "top_pick_correct": _top_pick(probs) == actual,
                    "probability_assigned_to_actual": _prob_for_actual(probs, actual),
                }
            )
    predictions = pd.DataFrame(rows, columns=prediction_columns)
    if predictions.empty:
        return pd.DataFrame(columns=columns), predictions
    metrics = _walk_forward_metrics(predictions)
    return metrics[columns], predictions


def model_vs_market_benchmark(evaluation_df: pd.DataFrame | None, min_sample: int = 5) -> pd.DataFrame:
    df = _valid_scored_rows(evaluation_df)
    complete = df.loc[df[["home_win_prob_market", "draw_prob_market", "away_win_prob_market"]].notna().all(axis=1)].copy()
    columns = [
        "segment",
        "n",
        "brier_raw_model",
        "brier_calibrated_model",
        "brier_market",
        "log_loss_raw_model",
        "log_loss_calibrated_model",
        "log_loss_market",
        "model_higher_actual_probability_share",
        "status",
    ]
    if complete.empty:
        return pd.DataFrame(
            [
                {
                    "segment": "all_market_rows",
                    "n": 0,
                    "status": "Market benchmark sample too small for reliable conclusions.",
                }
            ],
            columns=columns,
        )
    complete["agreement_segment"] = complete["market_disagreement_pp"].map(_market_agreement_segment)
    frames = [("all_market_rows", complete)]
    frames.extend((segment, group) for segment, group in complete.groupby("agreement_segment", dropna=False))
    rows = []
    for segment, group in frames:
        raw = _metric_row(group, "raw_model", ["home_win_prob_raw", "draw_prob_raw", "away_win_prob_raw"])
        market = _metric_row(group, "market_implied", ["home_win_prob_market", "draw_prob_market", "away_win_prob_market"])
        calibrated = (
            _metric_row(group, "calibrated_model", ["home_win_prob_calibrated", "draw_prob_calibrated", "away_win_prob_calibrated"])
            if group[["home_win_prob_calibrated", "draw_prob_calibrated", "away_win_prob_calibrated"]].notna().all(axis=1).any()
            else {}
        )
        rows.append(
            {
                "segment": segment,
                "n": int(len(group)),
                "brier_raw_model": raw.get("brier_score", pd.NA),
                "brier_calibrated_model": calibrated.get("brier_score", pd.NA),
                "brier_market": market.get("brier_score", pd.NA),
                "log_loss_raw_model": raw.get("multiclass_log_loss", pd.NA),
                "log_loss_calibrated_model": calibrated.get("multiclass_log_loss", pd.NA),
                "log_loss_market": market.get("multiclass_log_loss", pd.NA),
                "model_higher_actual_probability_share": float(group["market_was_closer_than_model"].map(lambda value: not bool(value)).mean()),
                "status": (
                    "Market benchmark sample too small for reliable conclusions."
                    if len(group) < min_sample
                    else model_market_benchmark_status(
                        len(group),
                        raw.get("brier_score", pd.NA),
                        market.get("brier_score", pd.NA),
                        min_sample_size=min_sample,
                    )
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def recent_match_postmortem(evaluation_df: pd.DataFrame | None, prediction_sources_df: pd.DataFrame | None = None) -> pd.DataFrame:
    eval_df = evaluation_df.copy() if evaluation_df is not None else pd.DataFrame(columns=EVALUATION_DATASET_COLUMNS)
    predictions = prediction_sources_df.copy() if prediction_sources_df is not None else pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for case in RECENT_DIAGNOSTIC_CASES:
        scored = _find_scored_case(eval_df, case)
        if scored is None:
            prediction = _find_prediction_case(predictions, case)
            scored = _postmortem_row_from_prediction(prediction, case)
        response = _recommended_model_response(scored)
        scored["recommended_model_system_response"] = response
        scored["what_warning_should_have_appeared"] = _warning_for_postmortem(scored)
        rows.append(scored)
    return pd.DataFrame(rows)


def write_milestone2_reports(
    evaluation_df: pd.DataFrame,
    *,
    reports_dir: str | Path = "reports",
    min_train: int = HIGH_CONFIDENCE_MIN_SAMPLE,
    prediction_sources_df: pd.DataFrame | None = None,
) -> dict[str, Path]:
    out = Path(reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = calibration_summary(evaluation_df)
    bins = calibration_by_bin(evaluation_df)
    by_confidence = performance_by_dimension(evaluation_df, "confidence_label")
    by_context = calibration_by_match_context(evaluation_df)
    misses = worst_model_misses(evaluation_df)
    comparison, wf_rows = walk_forward_calibration_comparison(evaluation_df, min_train=min_train)
    market = model_vs_market_benchmark(evaluation_df)
    postmortem = recent_match_postmortem(evaluation_df, prediction_sources_df)

    paths = {
        "evaluation_dataset": out / "evaluation_dataset.csv",
        "calibration_summary": out / "calibration_summary.csv",
        "calibration_by_bin": out / "calibration_by_bin.csv",
        "calibration_by_confidence": out / "calibration_by_confidence.csv",
        "calibration_by_match_context": out / "calibration_by_match_context.csv",
        "worst_model_misses": out / "worst_model_misses.csv",
        "calibration_report": out / "calibration_report.md",
        "calibration_model_comparison": out / "calibration_model_comparison.csv",
        "walk_forward_calibration_results": out / "walk_forward_calibration_results.csv",
        "model_vs_market_benchmark": out / "model_vs_market_benchmark.csv",
        "model_vs_market_report": out / "model_vs_market_report.md",
        "recent_match_postmortem": out / "recent_match_postmortem.csv",
        "recent_match_postmortem_report": out / "recent_match_postmortem.md",
    }
    evaluation_df.to_csv(paths["evaluation_dataset"], index=False)
    summary.to_csv(paths["calibration_summary"], index=False)
    bins.to_csv(paths["calibration_by_bin"], index=False)
    by_confidence.to_csv(paths["calibration_by_confidence"], index=False)
    by_context.to_csv(paths["calibration_by_match_context"], index=False)
    misses.to_csv(paths["worst_model_misses"], index=False)
    comparison.to_csv(paths["calibration_model_comparison"], index=False)
    wf_rows.to_csv(paths["walk_forward_calibration_results"], index=False)
    market.to_csv(paths["model_vs_market_benchmark"], index=False)
    postmortem.to_csv(paths["recent_match_postmortem"], index=False)
    paths["calibration_report"].write_text(
        render_calibration_report(evaluation_df, summary, bins, by_confidence, by_context, misses, comparison, market),
        encoding="utf-8",
    )
    paths["model_vs_market_report"].write_text(render_model_vs_market_report(market), encoding="utf-8")
    paths["recent_match_postmortem_report"].write_text(render_recent_match_postmortem(postmortem), encoding="utf-8")
    return paths


def render_calibration_report(
    evaluation_df: pd.DataFrame,
    summary: pd.DataFrame,
    bins: pd.DataFrame,
    by_confidence: pd.DataFrame,
    by_context: pd.DataFrame,
    misses: pd.DataFrame,
    comparison: pd.DataFrame,
    market: pd.DataFrame,
) -> str:
    usable = len(_valid_scored_rows(evaluation_df))
    excluded = _excluded_prediction_summary(evaluation_df)
    raw = _summary_row(summary, "raw_model")
    calibrated = _summary_row(summary, "calibrated_model")
    market_row = _summary_row(summary, "market_implied")
    comparison_status = _comparison_status(comparison)
    lines = [
        f"# Calibration Report - {today_iso()}",
        "",
        "This report evaluates saved pre-kickoff 1X2 predictions against regular-time results only. It is evaluation-only and does not provide staking, trade execution, wallet, Kelly sizing, or investment advice.",
        "",
        f"Sample size: `{usable}` usable evaluated prediction snapshot(s).",
        f"Usable evaluated predictions: `{usable}`.",
        f"Excluded predictions and reasons: {excluded}",
        "",
        "## Raw Model Performance",
        "",
        _metric_sentence(raw, "Raw model"),
        "",
        "## Calibrated Model Performance",
        "",
        _metric_sentence(calibrated, "Calibrated model") if calibrated else "No production-eligible calibrated probabilities are available in the evaluated dataset.",
        "",
        "## Market Performance",
        "",
        _metric_sentence(market_row, "Market-implied probabilities") if market_row else "Market benchmark sample too small for reliable conclusions.",
        "",
        "## Walk-Forward Candidate Methods",
        "",
        comparison_status,
        "",
        _to_markdown(_printable(comparison)),
        "",
        "## Reliability By Probability Bin",
        "",
        _to_markdown(_printable(bins)),
        "",
        "## Performance By Confidence",
        "",
        _to_markdown(_printable(by_confidence)),
        "",
        "## Performance By Match Context",
        "",
        _to_markdown(_printable(by_context)),
        "",
        "## Worst Model Misses",
        "",
        _to_markdown(_printable(misses.head(10))),
        "",
        "## Conclusions",
        "",
        f"- Calibration improves Brier score: {_improves(summary, 'brier_score')}",
        f"- Calibration improves log loss: {_improves(summary, 'multiclass_log_loss')}",
        f"- Model beats market-implied probabilities: {_model_beats_market(summary)}",
        "- Where the model is overconfident: review high favorite-probability bands, high behavior disagreement, and knockout draw rows above.",
        "- Ready to claim calibrated predictive probabilities: No. The current evaluated sample is too small and no candidate is production-eligible.",
    ]
    return "\n".join(lines)


def render_model_vs_market_report(market: pd.DataFrame) -> str:
    lines = [
        f"# Model vs Market Benchmark - {today_iso()}",
        "",
        "This benchmark only uses semantically matched 90-minute 1X2 market probabilities. Advancement or to-qualify markets are excluded.",
        "",
        _to_markdown(_printable(market)),
        "",
    ]
    if market.empty or int(coerce_float(market.iloc[0].get("n", 0), 0)) < 5:
        lines.append("Market benchmark sample too small for reliable conclusions.")
    else:
        lines.append(str(market.iloc[0].get("status", "")))
    lines.append("")
    lines.append("No staking, trade execution, wallet, Kelly sizing, or investment advice is provided.")
    return "\n".join(lines)


def render_recent_match_postmortem(postmortem: pd.DataFrame) -> str:
    mexico = postmortem.loc[
        postmortem["home_team"].astype(str).str.lower().eq("mexico")
        & postmortem["away_team"].astype(str).str.lower().eq("ecuador")
    ]
    lines = [
        f"# Recent Match Post-Mortem - {today_iso()}",
        "",
        "This report separates 90-minute 1X2 results from advancement. Penalty advancement is never treated as a 1X2 win.",
        "",
        _to_markdown(_printable(postmortem)),
        "",
        "## Mexico vs Ecuador",
        "",
    ]
    if mexico.empty:
        lines.append("Mexico vs Ecuador is missing from local prediction sources.")
    else:
        row = mexico.iloc[0]
        lines.extend(
            [
                "Mexico vs Ecuador was a major diagnostic miss: the raw model top pick was Ecuador while the 90-minute result was Mexico.",
                f"Probability assigned to the actual result: `{_fmt(row.get('probability_assigned_to_actual_raw'))}`.",
                "The behavior layer disagreement should have downgraded confidence before kickoff.",
                "Venue/home/altitude logic was insufficient for this case because the raw model still overstated Ecuador despite Mexico's contextual flags.",
                "A market prior would only be considered if a semantically matched 90-minute 1X2 market was joined; otherwise the correct status is model-only/no market comparison.",
                "Future similar cases should trigger low-confidence forecast, behavior-disagreement caution, venue-context caution, and calibration caution.",
            ]
        )
    lines.extend(
        [
            "",
            "This post-mortem is diagnostic-only. It must not be used to hard-code a Mexico/Ecuador correction or to tune only the latest misses.",
        ]
    )
    return "\n".join(lines)


def _prediction_rows(
    prediction_ledger_df: pd.DataFrame | None,
    prediction_log_df: pd.DataFrame | None,
    market_odds_df: pd.DataFrame | None,
    *,
    filter_pre_kickoff: bool = True,
    latest_snapshot_only: bool = True,
) -> pd.DataFrame:
    frames = [
        _prediction_rows_from_ledger(prediction_ledger_df, market_odds_df, latest_snapshot_only=latest_snapshot_only),
        _prediction_rows_from_prediction_log(prediction_log_df),
    ]
    out = pd.concat([frame for frame in frames if frame is not None and not frame.empty], ignore_index=True) if any(not frame.empty for frame in frames) else pd.DataFrame()
    if out.empty:
        return out
    out["_generated_ts"] = _parse_utc_series(out["generated_at_utc"])
    out["_kickoff_ts"] = _parse_utc_series(out["kickoff_utc"])
    if filter_pre_kickoff:
        out = out.loc[out["_generated_ts"].notna() & out["_kickoff_ts"].notna() & (out["_generated_ts"] < out["_kickoff_ts"])].copy()
    return out.reset_index(drop=True)


def _prediction_rows_from_ledger(
    prediction_ledger_df: pd.DataFrame | None,
    market_odds_df: pd.DataFrame | None,
    *,
    latest_snapshot_only: bool = True,
) -> pd.DataFrame:
    df = prediction_ledger_df.copy() if prediction_ledger_df is not None else pd.DataFrame()
    if df.empty:
        return pd.DataFrame()
    if latest_snapshot_only:
        df = select_latest_valid_snapshots(df, require_pre_kickoff=True, exclude_unresolved_teams=True)
        if df.empty:
            return pd.DataFrame()
    rows = []
    for _, row in df.iterrows():
        market_probs = _market_probs_from_prediction(row, market_odds_df)
        raw = _normalise_or_missing(
            [
                row.get("home_win_prob", row.get("home_win_prob_raw")),
                row.get("draw_prob", row.get("draw_prob_raw")),
                row.get("away_win_prob", row.get("away_win_prob_raw")),
            ]
        )
        behavior = _optional_probability_triplet([row.get("behavior_home_prob"), row.get("behavior_draw_prob"), row.get("behavior_away_prob")])
        rows.append(
            {
                "prediction_id": row.get("prediction_id", ""),
                "generated_at_utc": row.get("snapshot_utc", row.get("generated_at_utc", "")),
                "match_id": row.get("match_id", ""),
                "kickoff_utc": row.get("kickoff_utc", ""),
                "home_team": row.get("home", row.get("home_team", "")),
                "away_team": row.get("away", row.get("away_team", "")),
                "competition": row.get("competition", ""),
                "round": row.get("group", row.get("round", "")),
                "venue": row.get("venue", ""),
                "model_version": row.get("model_version", ""),
                "home_win_prob_raw": raw[0],
                "draw_prob_raw": raw[1],
                "away_win_prob_raw": raw[2],
                "home_win_prob_market": market_probs[0],
                "draw_prob_market": market_probs[1],
                "away_win_prob_market": market_probs[2],
                "behavior_home_prob": behavior[0],
                "behavior_draw_prob": behavior[1],
                "behavior_away_prob": behavior[2],
                "model_confidence": row.get("model_confidence", ""),
                "market_join_status": _market_join_status(market_probs, row.get("market_snapshot_available", False)),
                "mapping_confidence": "",
                "prediction_source": row.get("source_type", "prediction_ledger") or "prediction_ledger",
            }
        )
    return pd.DataFrame(rows)


def _prediction_rows_from_prediction_log(prediction_log_df: pd.DataFrame | None) -> pd.DataFrame:
    df = prediction_log_df.copy() if prediction_log_df is not None else pd.DataFrame()
    if df.empty or "match_id" not in df.columns:
        return pd.DataFrame()
    keys = ["timestamp_utc", "match_id", "home", "away", "kickoff_utc"]
    for key in keys:
        if key not in df.columns:
            df[key] = ""
    rows = []
    for key, group in df.groupby(keys, dropna=False):
        timestamp, match_id, home, away, kickoff = key
        raw = _side_probabilities(group, "primary_model_probability", "model_probability")
        behavior = _side_probabilities(group, "behavior_diagnostic_probability", None, missing_ok=True)
        market = _market_probabilities_from_prediction_log(group)
        if not all(pd.notna(value) for value in raw):
            continue
        rows.append(
            {
                "prediction_id": _stable_prediction_id(timestamp, match_id, "prediction_log"),
                "generated_at_utc": timestamp,
                "match_id": match_id,
                "kickoff_utc": kickoff,
                "home_team": home,
                "away_team": away,
                "competition": "",
                "round": "",
                "venue": "",
                "model_version": group["model_policy"].iloc[0] if "model_policy" in group.columns else "",
                "home_win_prob_raw": raw[0],
                "draw_prob_raw": raw[1],
                "away_win_prob_raw": raw[2],
                "home_win_prob_market": market[0],
                "draw_prob_market": market[1],
                "away_win_prob_market": market[2],
                "behavior_home_prob": behavior[0],
                "behavior_draw_prob": behavior[1],
                "behavior_away_prob": behavior[2],
                "model_confidence": "",
                "market_join_status": _market_join_status(market, True),
                "mapping_confidence": _lowest_mapping_confidence(group),
                "prediction_source": "prediction_log",
            }
        )
    return pd.DataFrame(rows)


def _result_rows(results_ledger_df: pd.DataFrame | None, completed_results_df: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    ledger = validate_results_ledger_semantics(results_ledger_df)
    for _, row in ledger.iterrows():
        home_goals = coerce_float(row.get("home_goals"), float("nan"))
        away_goals = coerce_float(row.get("away_goals"), float("nan"))
        rows.append(
            {
                "match_id": str(row.get("match_id", "") or ""),
                "date_utc": row.get("date_utc", ""),
                "home_team": row.get("home", ""),
                "away_team": row.get("away", ""),
                "actual_home_goals_90": home_goals,
                "actual_away_goals_90": away_goals,
                "actual_result_1x2": str(row.get("actual_result", "") or _result_from_goals(home_goals, away_goals)),
                "advancing_team": row.get("actual_advancing_team", ""),
                "result_semantics": _text_or_default(row.get("result_semantics", pd.NA), "90-minute regular time"),
                "evaluation_eligible_1x2": bool(row.get("evaluation_eligible_1x2", True)),
                "had_extra_time": bool(row.get("had_extra_time", False)),
                "had_penalties": bool(row.get("had_penalties", False)),
                "penalties_home": row.get("penalties_home", pd.NA),
                "penalties_away": row.get("penalties_away", pd.NA),
                "result_source": row.get("result_source", ""),
            }
        )
    completed = completed_results_df.copy() if completed_results_df is not None else pd.DataFrame()
    for _, row in completed.iterrows():
        if str(row.get("is_completed", "1")).strip().lower() not in {"1", "true", "yes"}:
            continue
        home_goals = coerce_float(row.get("home_score_90"), float("nan"))
        away_goals = coerce_float(row.get("away_score_90"), float("nan"))
        rows.append(
            {
                "match_id": str(row.get("provider_match_id", "") or ""),
                "date_utc": row.get("date_utc", ""),
                "home_team": row.get("home", ""),
                "away_team": row.get("away", ""),
                "actual_home_goals_90": home_goals,
                "actual_away_goals_90": away_goals,
                "actual_result_1x2": _result_from_goals(home_goals, away_goals),
                "advancing_team": "",
                "result_semantics": _text_or_default(row.get("score_semantics", pd.NA), "90-minute regular time"),
                "evaluation_eligible_1x2": True,
                "had_extra_time": False,
                "had_penalties": False,
                "penalties_home": pd.NA,
                "penalties_away": pd.NA,
                "result_source": row.get("source", row.get("provider", "")),
            }
        )
    return validate_results_ledger_semantics(pd.DataFrame(rows))


def _evaluation_crosswalk(
    result_fixture_crosswalk_df: pd.DataFrame | None,
    results_ledger_df: pd.DataFrame | None,
    fixtures_df: pd.DataFrame | None,
) -> pd.DataFrame:
    if result_fixture_crosswalk_df is not None and not result_fixture_crosswalk_df.empty:
        return result_fixture_crosswalk_df.copy()
    if results_ledger_df is None or fixtures_df is None:
        return pd.DataFrame()
    return build_result_fixture_crosswalk(results_ledger_df, fixtures_df)


def _build_result_lookup(results: pd.DataFrame) -> dict[str, Any]:
    by_id = {}
    by_pair_date = {}
    for _, row in results.iterrows():
        match_id = str(row.get("match_id", "") or "").strip()
        if match_id and match_id not in by_id:
            by_id[match_id] = row
        key = _pair_date_key(row.get("home_team"), row.get("away_team"), row.get("date_utc"))
        if key not in by_pair_date:
            by_pair_date[key] = row
    return {"by_id": by_id, "by_pair_date": by_pair_date}


def _find_result_for_prediction(prediction: pd.Series, lookup: dict[str, Any]) -> pd.Series | None:
    match_id = str(prediction.get("match_id", "") or "").strip()
    if match_id in lookup["by_id"]:
        return lookup["by_id"][match_id]
    kickoff = pd.to_datetime(prediction.get("kickoff_utc"), errors="coerce", utc=True)
    date_text = "" if pd.isna(kickoff) else kickoff.date().isoformat()
    key = _pair_date_key(prediction.get("home_team"), prediction.get("away_team"), date_text)
    return lookup["by_pair_date"].get(key)


def _scored_dataset_row(prediction: pd.Series, result: pd.Series, calibration_status: str, min_calibration_sample: int) -> dict[str, Any]:
    raw = _row_probabilities(prediction, "raw")
    calibrated = _row_probabilities(prediction, "calibrated", allow_missing=True)
    market = _row_probabilities(prediction, "market", allow_missing=True)
    behavior = _row_probabilities(prediction, "behavior", allow_missing=True)
    actual = str(result.get("actual_result_1x2", "") or "")
    behavior_gap = _max_gap_pp(raw, behavior)
    market_gap = _max_gap_pp(raw, market)
    conf = score_forecast_confidence(
        calibration_sample_size=0,
        calibration_status=calibration_status,
        market_join_status=str(prediction.get("market_join_status", "")),
        behavior_disagreement_pp=behavior_gap,
        market_disagreement_pp=market_gap,
        venue_context="",
        team_sample_size=pd.NA,
        match_round=str(prediction.get("round", "")),
        draw_probability=raw[1],
        favorite_probability=max(raw[0], raw[2]),
    )
    raw_actual_prob = _prob_for_actual(raw, actual)
    calibrated_actual_prob = _prob_for_actual(calibrated, actual) if all(pd.notna(value) for value in calibrated) else pd.NA
    market_actual_prob = _prob_for_actual(market, actual) if all(pd.notna(value) for value in market) else pd.NA
    market_closer = (
        bool(pd.notna(market_actual_prob) and pd.notna(raw_actual_prob) and float(market_actual_prob) > float(raw_actual_prob))
        if pd.notna(market_actual_prob)
        else pd.NA
    )
    return {
        "prediction_id": prediction.get("prediction_id", ""),
        "prediction_source": prediction.get("prediction_source", ""),
        "generated_at_utc": prediction.get("generated_at_utc", ""),
        "match_id": prediction.get("match_id", ""),
        "kickoff_utc": prediction.get("kickoff_utc", ""),
        "home_team": prediction.get("home_team", ""),
        "away_team": prediction.get("away_team", ""),
        "competition": prediction.get("competition", ""),
        "round": prediction.get("round", ""),
        "venue": prediction.get("venue", ""),
        "model_version": prediction.get("model_version", ""),
        "home_win_prob_raw": raw[0],
        "draw_prob_raw": raw[1],
        "away_win_prob_raw": raw[2],
        "home_win_prob_calibrated": calibrated[0],
        "draw_prob_calibrated": calibrated[1],
        "away_win_prob_calibrated": calibrated[2],
        "home_win_prob_market": market[0],
        "draw_prob_market": market[1],
        "away_win_prob_market": market[2],
        "behavior_home_prob": behavior[0],
        "behavior_draw_prob": behavior[1],
        "behavior_away_prob": behavior[2],
        "behavior_disagreement_pp": behavior_gap,
        "market_disagreement_pp": market_gap,
        "model_confidence": prediction.get("model_confidence", ""),
        "confidence_score": conf["confidence_score"],
        "confidence_label": conf["confidence_label"],
        "confidence_reason": conf["confidence_reason"],
        "confidence_flags": conf["confidence_flags"],
        "market_join_status": prediction.get("market_join_status", ""),
        "mapping_confidence": prediction.get("mapping_confidence", ""),
        "data_quality_status": "scored_90_minute",
        "actual_home_goals_90": result.get("actual_home_goals_90", pd.NA),
        "actual_away_goals_90": result.get("actual_away_goals_90", pd.NA),
        "actual_result_1x2": actual,
        "advancing_team": result.get("advancing_team", ""),
        "result_semantics": result.get("result_semantics", "90-minute regular time"),
        "top_pick_raw": _top_pick(raw),
        "top_pick_calibrated": _top_pick(calibrated) if all(pd.notna(value) for value in calibrated) else pd.NA,
        "probability_assigned_to_actual_raw": raw_actual_prob,
        "probability_assigned_to_actual_calibrated": calibrated_actual_prob,
        "brier_raw": brier_1x2(raw, actual),
        "brier_calibrated": brier_1x2(calibrated, actual) if all(pd.notna(value) for value in calibrated) else pd.NA,
        "log_loss_raw": safe_log_loss(raw_actual_prob),
        "log_loss_calibrated": safe_log_loss(calibrated_actual_prob) if pd.notna(calibrated_actual_prob) else pd.NA,
        "market_probability_assigned_to_actual": market_actual_prob,
        "brier_market": brier_1x2(market, actual) if all(pd.notna(value) for value in market) else pd.NA,
        "log_loss_market": safe_log_loss(market_actual_prob) if pd.notna(market_actual_prob) else pd.NA,
        "raw_model_was_correct": _top_pick(raw) == actual,
        "calibrated_model_was_correct": (_top_pick(calibrated) == actual if all(pd.notna(value) for value in calibrated) else pd.NA),
        "market_was_closer_than_model": market_closer,
        "notes": f"source={prediction.get('prediction_source', '')}; calibration_min_sample={min_calibration_sample}",
    }


def _valid_scored_rows(evaluation_df: pd.DataFrame | None) -> pd.DataFrame:
    df = evaluation_df.copy() if evaluation_df is not None else pd.DataFrame(columns=EVALUATION_DATASET_COLUMNS)
    for col in EVALUATION_DATASET_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df.loc[df["data_quality_status"].astype(str).eq("scored_90_minute") & df["actual_result_1x2"].astype(str).isin(OUTCOMES)].copy()


def _metric_row(df: pd.DataFrame, model: str, columns: list[str]) -> dict[str, Any]:
    complete = df.loc[df[columns].notna().all(axis=1)].copy()
    if complete.empty:
        return {
            "model": model,
            "n": 0,
            "top_pick_accuracy": pd.NA,
            "brier_score": pd.NA,
            "multiclass_log_loss": pd.NA,
            "expected_calibration_error": pd.NA,
            "mean_actual_result_probability": pd.NA,
            "calibration_slope": pd.NA,
            "calibration_intercept": pd.NA,
            "status": "no_scored_rows",
        }
    probs = complete[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    actual = complete["actual_result_1x2"].astype(str).tolist()
    actual_probs = [_prob_for_actual(prob, outcome) for prob, outcome in zip(probs, actual)]
    top = [_top_pick(prob) for prob in probs]
    slope, intercept = _calibration_slope_intercept(actual_probs, top, actual)
    return {
        "model": model,
        "n": int(len(complete)),
        "top_pick_accuracy": float(np.mean([pred == outcome for pred, outcome in zip(top, actual)])),
        "brier_score": float(np.mean([brier_1x2(prob, outcome) for prob, outcome in zip(probs, actual)])),
        "multiclass_log_loss": float(np.mean([safe_log_loss(prob) for prob in actual_probs])),
        "expected_calibration_error": _expected_calibration_error(complete, columns),
        "mean_actual_result_probability": float(np.nanmean(actual_probs)),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "status": "ok",
    }


def _expected_calibration_error(df: pd.DataFrame, columns: list[str]) -> float:
    rows = []
    for outcome, col in zip(OUTCOMES, columns):
        part = pd.DataFrame(
            {
                "prob": pd.to_numeric(df[col], errors="coerce"),
                "actual": (df["actual_result_1x2"].astype(str) == outcome).astype(float),
            }
        ).dropna()
        rows.append(part)
    frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if frame.empty:
        return float("nan")
    frame["bucket"] = pd.cut(frame["prob"].clip(0, 1), PROBABILITY_BINS, labels=PROBABILITY_BIN_LABELS, include_lowest=True)
    grouped = frame.groupby("bucket", observed=False).agg(n=("actual", "size"), avg=("prob", "mean"), actual=("actual", "mean"))
    grouped = grouped.loc[grouped["n"] > 0].copy()
    if grouped.empty:
        return float("nan")
    return float((grouped["n"] * (grouped["avg"] - grouped["actual"]).abs()).sum() / grouped["n"].sum())


def _calibration_slope_intercept(actual_probs: list[float], top_predictions: list[str], actual: list[str]) -> tuple[Any, Any]:
    if len(actual_probs) < 3:
        return pd.NA, pd.NA
    x = np.array([_logit(prob) for prob in actual_probs if pd.notna(prob)], dtype=float)
    y = np.array([1.0 if pred == outcome else 0.0 for pred, outcome in zip(top_predictions, actual)], dtype=float)
    if len(x) != len(y) or len(x) < 3 or np.nanstd(x) <= 1e-9:
        return pd.NA, pd.NA
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def _walk_forward_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in predictions.groupby("model", dropna=False):
        rows.append(
            {
                "model": model,
                "n": int(len(group)),
                "brier_score": float(pd.to_numeric(group["brier_score"], errors="coerce").mean()),
                "log_loss": float(pd.to_numeric(group["log_loss"], errors="coerce").mean()),
                "top_pick_accuracy": float(group["top_pick_correct"].astype(bool).mean()),
                "expected_calibration_error": _prediction_ece(group),
            }
        )
    out = pd.DataFrame(rows)
    raw = out.loc[out["model"] == "raw_model"]
    raw_brier = float(raw.iloc[0]["brier_score"]) if not raw.empty else float("nan")
    raw_log = float(raw.iloc[0]["log_loss"]) if not raw.empty else float("nan")
    out["brier_delta_vs_raw"] = out["brier_score"] - raw_brier
    out["log_loss_delta_vs_raw"] = out["log_loss"] - raw_log
    out["production_status"] = out.apply(
        lambda row: (
            "production_eligible"
            if row["model"] != "raw_model" and row["brier_delta_vs_raw"] < 0 and row["log_loss_delta_vs_raw"] < 0
            else ("raw_baseline" if row["model"] == "raw_model" else "review_only_not_promoted")
        ),
        axis=1,
    )
    out["notes"] = out["production_status"].map(
        {
            "production_eligible": "Candidate beat raw on both Brier score and log loss in walk-forward validation.",
            "raw_baseline": "Raw model baseline.",
            "review_only_not_promoted": "Candidate did not beat raw on both promotion metrics.",
        }
    )
    return out.sort_values(["log_loss", "brier_score", "model"]).reset_index(drop=True)


def _walk_forward_variants(row: pd.Series, raw: list[float], base: list[float], fitted: dict[str, Any]) -> dict[str, list[float]]:
    shrink = _blend(raw, base, fitted["shrinkage"])
    market = _market_prior_variant(row, raw, base, fitted)
    behavior = _behavior_variant(row, raw, base, fitted)
    confidence = _confidence_variant(row, raw, base, fitted)
    combined = _blend(_behavior_variant(row, market, base, fitted), base, min(fitted["shrinkage"], 0.25))
    return {
        "raw_model": raw,
        "temperature_scaling": _temperature_scale(raw, fitted["temperature"]),
        "shrinkage_to_base_rates": shrink,
        "market_prior_blend": market,
        "behavior_gated_shrinkage": behavior,
        "confidence_gated_shrinkage": confidence,
        "combined_conservative_calibrated_model": combined,
    }


def _fit_temperature(train: pd.DataFrame) -> float:
    candidates = [0.7, 0.85, 1.0, 1.15, 1.3, 1.6, 2.0]
    return min(candidates, key=lambda temp: _mean_log_loss(train, lambda row: _temperature_scale(_row_probabilities(row, "raw"), temp)))


def _fit_shrinkage(train: pd.DataFrame, base: list[float]) -> float:
    weights = [idx / 20 for idx in range(0, 11)]
    return min(weights, key=lambda weight: _mean_log_loss(train, lambda row: _blend(_row_probabilities(row, "raw"), base, weight)))


def _fit_market_weight(train: pd.DataFrame) -> float:
    rows = train.loc[train[["home_win_prob_market", "draw_prob_market", "away_win_prob_market"]].notna().all(axis=1)]
    if len(rows) < 3:
        return 0.0
    weights = [idx / 10 for idx in range(0, 8)]
    return min(
        weights,
        key=lambda weight: _mean_log_loss(
            rows,
            lambda row: _blend(_row_probabilities(row, "raw"), _row_probabilities(row, "market"), weight),
        ),
    )


def _fit_behavior_gate(train: pd.DataFrame, base: list[float]) -> dict[str, float]:
    thresholds = [10.0, 20.0, 30.0]
    weights = [0.0, 0.10, 0.20, 0.30]
    best = {"threshold": float("inf"), "weight": 0.0}
    best_loss = float("inf")
    for threshold in thresholds:
        for weight in weights:
            loss = _mean_log_loss(train, lambda row: _blend(_row_probabilities(row, "raw"), base, weight) if coerce_float(row.get("behavior_disagreement_pp"), 0.0) >= threshold else _row_probabilities(row, "raw"))
            if loss < best_loss:
                best_loss = loss
                best = {"threshold": threshold, "weight": weight}
    return best


def _fit_confidence_shrinkage(train: pd.DataFrame, base: list[float]) -> float:
    weights = [idx / 20 for idx in range(0, 9)]
    return min(
        weights,
        key=lambda weight: _mean_log_loss(
            train,
            lambda row: _blend(_row_probabilities(row, "raw"), base, weight)
            if str(row.get("confidence_label", "")).lower() != "high"
            else _row_probabilities(row, "raw"),
        ),
    )


def _mean_log_loss(df: pd.DataFrame, probability_fn) -> float:
    losses = []
    for _, row in df.iterrows():
        probs = probability_fn(row)
        losses.append(safe_log_loss(_prob_for_actual(probs, str(row.get("actual_result_1x2", "")))))
    return float(np.mean(losses)) if losses else float("inf")


def _market_prior_variant(row: pd.Series, raw: list[float], base: list[float], fitted: dict[str, Any]) -> list[float]:
    market = _row_probabilities(row, "market", allow_missing=True)
    if all(pd.notna(value) for value in market):
        return _blend(raw, market, fitted["market_weight"])
    return _blend(raw, base, min(fitted["shrinkage"], 0.20))


def _behavior_variant(row: pd.Series, raw: list[float], base: list[float], fitted: dict[str, Any]) -> list[float]:
    gate = fitted["behavior_gate"]
    if coerce_float(row.get("behavior_disagreement_pp"), 0.0) >= gate["threshold"]:
        return _blend(raw, base, gate["weight"])
    return raw


def _confidence_variant(row: pd.Series, raw: list[float], base: list[float], fitted: dict[str, Any]) -> list[float]:
    if str(row.get("confidence_label", "")).lower() == "high":
        return raw
    return _blend(raw, base, fitted["confidence_shrinkage"])


def _prediction_ece(predictions: pd.DataFrame) -> float:
    rows = []
    for outcome, column in zip(OUTCOMES, ["home_prob", "draw_prob", "away_prob"]):
        part = pd.DataFrame(
            {
                "prob": pd.to_numeric(predictions[column], errors="coerce"),
                "actual": (predictions["actual_result_1x2"].astype(str) == outcome).astype(float),
            }
        ).dropna()
        rows.append(part)
    frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if frame.empty:
        return float("nan")
    frame["bucket"] = pd.cut(frame["prob"].clip(0, 1), PROBABILITY_BINS, labels=PROBABILITY_BIN_LABELS, include_lowest=True)
    table = frame.groupby("bucket", observed=False).agg(n=("actual", "size"), avg=("prob", "mean"), actual=("actual", "mean"))
    table = table.loc[table["n"] > 0].copy()
    if table.empty:
        return float("nan")
    return float((table["n"] * (table["avg"] - table["actual"]).abs()).sum() / table["n"].sum())


def _find_scored_case(df: pd.DataFrame, case: dict[str, Any]) -> dict[str, Any] | None:
    if df.empty:
        return None
    pair = {str(case["home"]).lower(), str(case["away"]).lower()}
    matches = df.loc[
        df.apply(lambda row: {str(row.get("home_team", "")).lower(), str(row.get("away_team", "")).lower()} == pair, axis=1)
    ].copy()
    if matches.empty:
        return None
    row = matches.sort_values("generated_at_utc").iloc[-1]
    return _postmortem_output_row(row, case.get("note", "Local evaluated dataset row."))


def _find_prediction_case(predictions: pd.DataFrame, case: dict[str, Any]) -> pd.Series | None:
    if predictions is None or predictions.empty:
        return None
    pair = {str(case["home"]).lower(), str(case["away"]).lower()}
    matches = predictions.loc[
        predictions.apply(lambda row: {str(row.get("home_team", "")).lower(), str(row.get("away_team", "")).lower()} == pair, axis=1)
    ].copy()
    if matches.empty:
        return None
    return matches.sort_values("generated_at_utc").iloc[-1]


def _postmortem_row_from_prediction(prediction: pd.Series | None, case: dict[str, Any]) -> dict[str, Any]:
    if prediction is None:
        row = {col: pd.NA for col in EVALUATION_DATASET_COLUMNS}
        row.update(
            {
                "home_team": case["home"],
                "away_team": case["away"],
                "actual_home_goals_90": case.get("actual_home_goals_90", pd.NA),
                "actual_away_goals_90": case.get("actual_away_goals_90", pd.NA),
                "actual_result_1x2": case.get("actual_result_1x2", ""),
                "advancing_team": case.get("advancing_team", ""),
                "data_quality_status": "diagnostic_case_no_local_prediction",
                "notes": case.get("note", ""),
            }
        )
        return _postmortem_output_row(pd.Series(row), case.get("note", ""))
    raw = _row_probabilities(prediction, "raw")
    actual = _case_actual_for_prediction(prediction, case)
    row = prediction.to_dict()
    row.update(
        {
            "actual_home_goals_90": case.get("actual_home_goals_90", pd.NA),
            "actual_away_goals_90": case.get("actual_away_goals_90", pd.NA),
            "actual_result_1x2": actual,
            "advancing_team": case.get("advancing_team", ""),
            "probability_assigned_to_actual_raw": _prob_for_actual(raw, actual),
            "brier_raw": brier_1x2(raw, actual),
            "log_loss_raw": safe_log_loss(_prob_for_actual(raw, actual)),
            "top_pick_raw": _top_pick(raw),
            "raw_model_was_correct": _top_pick(raw) == actual,
            "behavior_disagreement_pp": case.get("behavior_disagreement_pp", _max_gap_pp(raw, _row_probabilities(prediction, "behavior", allow_missing=True))),
            "market_disagreement_pp": _max_gap_pp(raw, _row_probabilities(prediction, "market", allow_missing=True)),
            "data_quality_status": "diagnostic_case_prompt_result_not_in_results_ledger",
            "notes": case.get("note", ""),
        }
    )
    return _postmortem_output_row(pd.Series(row), case.get("note", ""))


def _postmortem_output_row(row: pd.Series, note: str) -> dict[str, Any]:
    return {
        "match_id": row.get("match_id", ""),
        "home_team": row.get("home_team", ""),
        "away_team": row.get("away_team", ""),
        "raw_probabilities": _probability_text(row, "raw"),
        "calibrated_probabilities": _probability_text(row, "calibrated"),
        "market_probabilities": _probability_text(row, "market"),
        "actual_90_min_result": row.get("actual_result_1x2", ""),
        "actual_home_goals_90": row.get("actual_home_goals_90", pd.NA),
        "actual_away_goals_90": row.get("actual_away_goals_90", pd.NA),
        "advancement_result": row.get("advancing_team", ""),
        "top_pick": row.get("top_pick_raw", ""),
        "probability_assigned_to_actual_raw": row.get("probability_assigned_to_actual_raw", pd.NA),
        "brier_raw": row.get("brier_raw", pd.NA),
        "log_loss_raw": row.get("log_loss_raw", pd.NA),
        "confidence_label": row.get("confidence_label", row.get("model_confidence", "")),
        "confidence_score": row.get("confidence_score", pd.NA),
        "behavior_disagreement_pp": row.get("behavior_disagreement_pp", pd.NA),
        "market_disagreement_pp": row.get("market_disagreement_pp", pd.NA),
        "market_join_status": row.get("market_join_status", ""),
        "venue_context_flags": _venue_flags(row),
        "data_quality_status": row.get("data_quality_status", ""),
        "notes": note or row.get("notes", ""),
    }


def _recommended_model_response(row: dict[str, Any]) -> str:
    responses = []
    if str(row.get("data_quality_status", "")).endswith("no_local_prediction"):
        responses.append("model-only forecast")
    if str(row.get("market_join_status", "")) != "joined_price":
        responses.append("no market comparison")
    if coerce_float(row.get("behavior_disagreement_pp"), 0.0) >= 20.0:
        responses.append("low-confidence forecast")
        responses.append("behavior-disagreement caution")
    if str(row.get("venue_context_flags", "")).strip():
        responses.append("venue-context caution")
    responses.append("calibration caution")
    return "; ".join(dict.fromkeys(responses or ["normal forecast"]))


def _warning_for_postmortem(row: dict[str, Any]) -> str:
    warnings = []
    if coerce_float(row.get("behavior_disagreement_pp"), 0.0) >= 20.0:
        warnings.append("Behavior-disagreement caution")
    if str(row.get("market_join_status", "")) != "joined_price":
        warnings.append("No market comparison")
    if str(row.get("confidence_label", "")).lower() != "high":
        warnings.append("Calibration caution")
    return "; ".join(warnings) if warnings else "Normal forecast"


def _case_actual_for_prediction(prediction: pd.Series, case: dict[str, Any]) -> str:
    actual = str(case.get("actual_result_1x2", ""))
    if actual == "draw":
        return "draw"
    winner = case.get("home") if actual == "home_win" else case.get("away")
    if str(prediction.get("home_team", "")).lower() == str(winner).lower():
        return "home_win"
    if str(prediction.get("away_team", "")).lower() == str(winner).lower():
        return "away_win"
    return actual


def _probability_text(row: pd.Series, prefix: str) -> str:
    probs = _row_probabilities(row, prefix, allow_missing=True)
    if not all(pd.notna(value) for value in probs):
        return "Unavailable"
    return f"H {probs[0]:.1%} / D {probs[1]:.1%} / A {probs[2]:.1%}"


def _venue_flags(row: pd.Series) -> str:
    flags = []
    venue = str(row.get("venue_context", "") or "")
    if venue:
        flags.append(venue)
    return "; ".join(flags)


def _row_probabilities(row: pd.Series, prefix: str, allow_missing: bool = False) -> list[Any]:
    mapping = {
        "raw": ["home_win_prob_raw", "draw_prob_raw", "away_win_prob_raw"],
        "calibrated": ["home_win_prob_calibrated", "draw_prob_calibrated", "away_win_prob_calibrated"],
        "market": ["home_win_prob_market", "draw_prob_market", "away_win_prob_market"],
        "behavior": ["behavior_home_prob", "behavior_draw_prob", "behavior_away_prob"],
    }
    cols = mapping[prefix]
    values = [row.get(col, pd.NA) for col in cols]
    if allow_missing and not all(_is_probability(value) for value in values):
        return [pd.NA, pd.NA, pd.NA]
    return _normalise_or_missing(values)


def _valid_prediction_probabilities(row: pd.Series) -> bool:
    return all(pd.notna(value) for value in _row_probabilities(row, "raw", allow_missing=True))


def _join_audit_status(
    prediction: pd.Series,
    result: pd.Series,
    resolution: dict[str, Any],
    generated_ts: pd.Timestamp | pd.NaT,
    kickoff_ts: pd.Timestamp | pd.NaT,
    now: pd.Timestamp,
    valid_probs: bool,
) -> tuple[str, bool, str]:
    result_found = result is not None and not result.empty
    if pd.isna(generated_ts):
        return "prediction_missing_generated_at", False, "Prediction snapshot has no parseable generated_at timestamp."
    if pd.isna(kickoff_ts):
        return "prediction_missing_generated_at", False, "Prediction snapshot has no parseable kickoff timestamp."
    if generated_ts >= kickoff_ts:
        if result_found:
            return "result_found_but_after_kickoff_issue", False, "Prediction was generated at or after kickoff."
        return "prediction_generated_after_kickoff", False, "Prediction was generated at or after kickoff."
    if not valid_probs:
        return "prediction_missing_probabilities", False, "Prediction is missing a valid home/draw/away probability triplet."
    if not result_found:
        if kickoff_ts > now:
            return "result_not_completed", False, "Fixture has not reached kickoff/result availability yet."
        return "no_result_found", False, "No completed result row matched this prediction."

    scope = market_type_semantics("1X2", str(result.get("result_semantics", "")))
    actual = str(result.get("actual_result_1x2", result.get("actual_result", "")) or "")
    eligible = bool(result.get("evaluation_eligible_1x2", False))
    if not eligible or scope != "90-minute" or actual not in OUTCOMES:
        return "result_found_but_ambiguous_semantics", False, "A result was found but it is not explicitly eligible for 90-minute 1X2 evaluation."

    order = str(resolution.get("home_away_order", "same"))
    if order == "reversed":
        return "result_found_but_team_order_mismatch", True, "Result matched with reversed home-away order and was aligned for scoring."
    method = str(resolution.get("result_join_method", ""))
    if method == "exact_match_id":
        return "exact_match_id_join", True, "Prediction match_id matched result match_id."
    if method == "result_fixture_crosswalk":
        return "schedule_bridge_join", True, "Prediction fixture_id matched a completed result through the schedule result bridge."
    if method == "fixture_bridge":
        return "fixture_bridge_join", True, "Prediction matched via fixtures.csv bridge."
    if method in {"normalized_team_date", "fuzzy_team_date"}:
        return "normalized_team_date_join", True, "Prediction matched by normalized teams and kickoff/result date."
    return "no_result_found", False, str(resolution.get("result_join_reason", "No completed result matched."))


def _mark_duplicate_unselected_snapshots(audit: pd.DataFrame) -> pd.DataFrame:
    if audit.empty:
        return audit
    out = audit.copy()
    usable_statuses = {
        "exact_match_id_join",
        "schedule_bridge_join",
        "fixture_bridge_join",
        "normalized_team_date_join",
        "result_found_but_team_order_mismatch",
    }
    candidates = out.loc[out["join_status"].isin(usable_statuses)].copy()
    if candidates.empty:
        return out
    candidates["_generated_ts"] = pd.to_datetime(candidates["generated_at_utc"], errors="coerce", utc=True)
    candidates["_group"] = candidates["resolved_result_match_id"].where(
        candidates["resolved_result_match_id"].astype(str).str.strip().ne(""),
        candidates.apply(
            lambda row: "|".join(
                [
                    str(row.get("match_id", "")),
                    str(row.get("home_team", "")),
                    str(row.get("away_team", "")),
                    normalize_match_date(row.get("kickoff_utc", "")),
                ]
            ),
            axis=1,
        ),
    )
    keep_idx = set(
        candidates.sort_values(["_group", "_generated_ts", "prediction_id"])
        .drop_duplicates(subset=["_group"], keep="last")
        .index
        .tolist()
    )
    duplicate_idx = [idx for idx in candidates.index.tolist() if idx not in keep_idx]
    if duplicate_idx:
        out.loc[duplicate_idx, "join_status"] = "duplicate_snapshot_not_selected"
        out.loc[duplicate_idx, "usable_for_evaluation"] = False
        out.loc[duplicate_idx, "exclusion_reason"] = "A later valid pre-kickoff snapshot was selected for this resolved match."
    return out[RESULT_PREDICTION_JOIN_AUDIT_COLUMNS].copy()


def _top_exclusion_markdown(audit: pd.DataFrame) -> str:
    if audit.empty:
        return "No prediction rows were available to audit."
    excluded = audit.loc[~audit["usable_for_evaluation"].astype(bool)].copy()
    if excluded.empty:
        return "No exclusions after latest-snapshot selection."
    counts = (
        excluded["join_status"]
        .value_counts(dropna=False)
        .rename_axis("join_status")
        .reset_index(name="count")
        .head(10)
    )
    return _to_markdown(counts)


def _pdf_backfill_sentence(structured_usable_rows: int) -> str:
    if structured_usable_rows >= HIGH_CONFIDENCE_MIN_SAMPLE:
        return (
            f"Structured CSVs currently provide `{structured_usable_rows}` usable evaluated row(s), "
            "so PDF backfill is not the first calibration blocker."
        )
    return (
        f"Structured CSVs currently provide `{structured_usable_rows}` usable evaluated row(s), below the "
        f"`{HIGH_CONFIDENCE_MIN_SAMPLE}`-row high-confidence calibration threshold. PDFs may materially increase "
        "the sample only if their generated-before-kickoff timestamps can be verified."
    )


def _normalise_or_missing(values: list[Any]) -> list[Any]:
    nums = [coerce_float(value, float("nan")) for value in values]
    if not all(pd.notna(value) and 0.0 <= value <= 1.0 for value in nums):
        return [pd.NA, pd.NA, pd.NA]
    total = sum(nums)
    if total <= 0:
        return [pd.NA, pd.NA, pd.NA]
    return [float(value / total) for value in nums]


def _optional_probability_triplet(values: list[Any]) -> list[Any]:
    return _normalise_or_missing(values) if all(_is_probability(value) for value in values) else [pd.NA, pd.NA, pd.NA]


def _is_probability(value: Any) -> bool:
    number = coerce_float(value, float("nan"))
    return bool(pd.notna(number) and 0.0 <= number <= 1.0)


def _prob_for_actual(probs: list[Any] | np.ndarray, actual: str) -> Any:
    if not all(pd.notna(value) for value in probs):
        return pd.NA
    idx = {"home_win": 0, "draw": 1, "away_win": 2}.get(str(actual))
    return pd.NA if idx is None else float(probs[idx])


def _top_pick(probs: list[Any] | np.ndarray) -> Any:
    if not all(pd.notna(value) for value in probs):
        return pd.NA
    return OUTCOMES[int(np.argmax(np.array(probs, dtype=float)))]


def _blend(source: list[float], target: list[float], target_weight: float) -> list[float]:
    weight = min(max(float(target_weight), 0.0), 1.0)
    return _normalise_or_missing([(1 - weight) * source[idx] + weight * target[idx] for idx in range(3)])


def _temperature_scale(probs: list[float], temperature: float) -> list[float]:
    temp = max(float(temperature), 0.05)
    logits = np.log(np.clip(np.array(probs, dtype=float), 1e-6, 1.0))
    scaled = logits / temp
    exp = np.exp(scaled - np.max(scaled))
    return (exp / exp.sum()).tolist()


def _base_rates(df: pd.DataFrame) -> list[float]:
    actual = df["actual_result_1x2"].astype(str)
    counts = [float((actual == outcome).sum()) + 1.0 for outcome in OUTCOMES]
    total = sum(counts)
    return [count / total for count in counts]


def _market_probs_from_prediction(row: pd.Series, market_odds_df: pd.DataFrame | None) -> list[Any]:
    direct = _optional_probability_triplet([row.get("market_home_prob"), row.get("market_draw_prob"), row.get("market_away_prob")])
    if all(pd.notna(value) for value in direct):
        return direct
    odds = market_odds_df.copy() if market_odds_df is not None else pd.DataFrame()
    if odds.empty or not {"match_id", "market", "selection", "odds"}.issubset(odds.columns):
        return [pd.NA, pd.NA, pd.NA]
    selected = odds.loc[(odds["match_id"].astype(str) == str(row.get("match_id", ""))) & (odds["market"].astype(str).str.lower() == "1x2")]
    if selected.empty:
        return [pd.NA, pd.NA, pd.NA]
    lookup = {str(item.get("selection", "")).strip().lower(): coerce_float(item.get("odds"), float("nan")) for _, item in selected.iterrows()}
    decimal = [
        lookup.get(str(row.get("home", "")).strip().lower(), float("nan")),
        lookup.get("draw", float("nan")),
        lookup.get(str(row.get("away", "")).strip().lower(), float("nan")),
    ]
    if not all(pd.notna(value) and value > 1 for value in decimal):
        return [pd.NA, pd.NA, pd.NA]
    return _normalise_or_missing([1 / value for value in decimal])


def _side_probabilities(group: pd.DataFrame, primary_col: str, fallback_col: str | None, missing_ok: bool = False) -> list[Any]:
    out = {"home_win": pd.NA, "draw": pd.NA, "away_win": pd.NA}
    for _, row in group.iterrows():
        side = str(row.get("model_side", "") or "").lower()
        if side not in out:
            continue
        value = row.get(primary_col, pd.NA)
        if not _is_probability(value) and fallback_col:
            value = row.get(fallback_col, pd.NA)
        out[side] = value
    probs = [out["home_win"], out["draw"], out["away_win"]]
    if missing_ok and not all(_is_probability(value) for value in probs):
        return [pd.NA, pd.NA, pd.NA]
    return _normalise_or_missing(probs)


def _market_probabilities_from_prediction_log(group: pd.DataFrame) -> list[Any]:
    out = {"home_win": pd.NA, "draw": pd.NA, "away_win": pd.NA}
    for _, row in group.iterrows():
        side = str(row.get("model_side", "") or "").lower()
        if side not in out:
            continue
        cents = coerce_float(row.get("polymarket_price_cents"), float("nan"))
        if pd.isna(cents):
            continue
        prob = cents / 100.0
        if str(row.get("polymarket_side", "YES")).upper() == "NO":
            prob = 1.0 - prob
        out[side] = prob
    return _normalise_or_missing([out["home_win"], out["draw"], out["away_win"]])


def _lowest_mapping_confidence(group: pd.DataFrame) -> str:
    if "mapping_confidence" not in group.columns:
        return ""
    order = {"low": 0, "medium": 1, "high": 2}
    values = [str(value).lower() for value in group["mapping_confidence"].dropna().tolist()]
    if not values:
        return ""
    return min(values, key=lambda value: order.get(value, 99))


def _market_join_status(market_probs: list[Any], market_snapshot_available: Any) -> str:
    if all(pd.notna(value) for value in market_probs):
        return "joined_price"
    if str(market_snapshot_available).strip().lower() in {"1", "true", "yes"}:
        return "event_found_no_usable_price"
    return "no_event_resolved"


def _max_gap_pp(left: list[Any], right: list[Any]) -> Any:
    values = [abs(float(a) - float(b)) * 100.0 for a, b in zip(left, right) if pd.notna(a) and pd.notna(b)]
    return max(values) if values else pd.NA


def _result_from_goals(home_goals: Any, away_goals: Any) -> str:
    home = coerce_float(home_goals, float("nan"))
    away = coerce_float(away_goals, float("nan"))
    if pd.isna(home) or pd.isna(away):
        return ""
    if home > away:
        return "home_win"
    if home < away:
        return "away_win"
    return "draw"


def _pair_date_key(home: Any, away: Any, date_utc: Any) -> tuple[str, str, str]:
    return (_team_key(home), _team_key(away), str(date_utc or "")[:10])


def _team_key(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("-", " ").split())


def _stable_prediction_id(*parts: Any) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _favorite_probability_band(value: Any) -> str:
    prob = coerce_float(value, float("nan"))
    if pd.isna(prob):
        return "missing"
    if prob < 0.4:
        return "0.00-0.40"
    if prob < 0.6:
        return "0.40-0.60"
    if prob < 0.8:
        return "0.60-0.80"
    return "0.80-1.00"


def _probability_band(value: Any) -> str:
    prob = coerce_float(value, float("nan"))
    if pd.isna(prob):
        return "missing"
    for idx in range(len(PROBABILITY_BINS) - 1):
        if PROBABILITY_BINS[idx] <= prob <= PROBABILITY_BINS[idx + 1]:
            return PROBABILITY_BIN_LABELS[idx]
    return "missing"


def _pp_gap_band(value: Any) -> str:
    gap = coerce_float(value, float("nan"))
    if pd.isna(gap):
        return "missing"
    return gap_band(gap / 100.0)


def _market_agreement_segment(value: Any) -> str:
    gap = coerce_float(value, float("nan"))
    if pd.isna(gap):
        return "missing"
    if gap < 5:
        return "agree_under_5pp"
    if gap < 10:
        return "disagree_5_10pp"
    if gap < 20:
        return "disagree_10_20pp"
    return "disagree_over_20pp"


def _logit(prob: float) -> float:
    clipped = min(max(float(prob), 0.001), 0.999)
    return math.log(clipped / (1 - clipped))


def _excluded_prediction_summary(evaluation_df: pd.DataFrame) -> str:
    return "See reports/evaluation_coverage_audit.md and reports/result_prediction_join_audit.csv for per-snapshot exclusion reasons."


def _summary_row(summary: pd.DataFrame, model: str) -> dict[str, Any] | None:
    if summary.empty or "model" not in summary.columns:
        return None
    row = summary.loc[summary["model"].astype(str) == model]
    return None if row.empty else row.iloc[0].to_dict()


def _metric_sentence(row: dict[str, Any] | None, label: str) -> str:
    if not row:
        return f"{label}: unavailable."
    return (
        f"{label}: n={int(coerce_float(row.get('n'), 0))}, "
        f"Brier={_fmt(row.get('brier_score'))}, "
        f"log loss={_fmt(row.get('multiclass_log_loss'))}, "
        f"top-pick accuracy={_fmt(row.get('top_pick_accuracy'))}, "
        f"ECE={_fmt(row.get('expected_calibration_error'))}."
    )


def _comparison_status(comparison: pd.DataFrame) -> str:
    if comparison.empty:
        return "No walk-forward candidate rows were scored."
    if (comparison.get("production_status", pd.Series(dtype=str)).astype(str) == "production_eligible").any():
        return "At least one candidate beat the raw model on both Brier score and log loss in walk-forward validation."
    if (comparison.get("production_status", pd.Series(dtype=str)).astype(str) == "not_promoted_sample_too_small").all():
        return "Calibration sample too small; using conservative shrinkage."
    return "No candidate is production-eligible; variants are review-only."


def _improves(summary: pd.DataFrame, metric: str) -> str:
    raw = _summary_row(summary, "raw_model")
    cal = _summary_row(summary, "calibrated_model")
    if not raw or not cal:
        return "Unavailable; no production-eligible calibrated rows."
    return "Yes" if coerce_float(cal.get(metric), float("inf")) < coerce_float(raw.get(metric), float("inf")) else "No"


def _model_beats_market(summary: pd.DataFrame) -> str:
    raw = _summary_row(summary, "raw_model")
    market = _summary_row(summary, "market_implied")
    if not raw or not market:
        return "Market benchmark sample too small for reliable conclusions."
    raw_brier = coerce_float(raw.get("brier_score"), float("nan"))
    market_brier = coerce_float(market.get("brier_score"), float("nan"))
    raw_log = coerce_float(raw.get("multiclass_log_loss"), float("nan"))
    market_log = coerce_float(market.get("multiclass_log_loss"), float("nan"))
    if pd.isna(raw_brier) or pd.isna(market_brier) or pd.isna(raw_log) or pd.isna(market_log):
        return "Market benchmark sample too small for reliable conclusions."
    return "Yes" if raw_brier < market_brier and raw_log < market_log else "No"


def _printable(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda value: "Unavailable" if pd.isna(value) else round(float(value), 4))
    return out


def _to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        values = [_display_cell(row.get(col, "")).replace("|", "\\|") for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    number = coerce_float(value, float("nan"))
    return "Unavailable" if pd.isna(number) else f"{number:.4f}"


def _text_or_default(value: Any, default: str) -> str:
    if value is None or pd.isna(value):
        return default
    text = str(value).strip()
    return text if text else default


def _display_cell(value: Any) -> str:
    if value is None:
        return "Unavailable"
    try:
        if pd.isna(value):
            return "Unavailable"
    except (TypeError, ValueError):
        pass
    text = str(value)
    if text.strip().lower() in {"", "nan", "<na>", "nat"}:
        return "Unavailable"
    return text


def _parse_utc_series(series: pd.Series) -> pd.Series:
    parsed = series.map(lambda value: pd.to_datetime(value, errors="coerce", utc=True))
    return pd.Series(parsed, index=series.index)
