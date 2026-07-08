from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.prediction_ledger import PREDICTION_LEDGER_COLUMNS, RESULTS_LEDGER_COLUMNS, select_latest_valid_snapshots
from src.utils import coerce_bool, coerce_float, today_iso
from src.venue_features import altitude_category, classify_venue_context


OUTCOMES = ["home_win", "draw", "away_win"]
OUTCOME_LABELS = {"home_win": "home", "draw": "draw", "away_win": "away"}
PROBABILITY_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
PROBABILITY_BIN_LABELS = ["0.00-0.20", "0.20-0.40", "0.40-0.60", "0.60-0.80", "0.80-1.00"]
DEFAULT_DIAGNOSTIC_MATCHES = [
    ("Uruguay", "Spain"),
    ("Panama", "England"),
    ("Colombia", "Portugal"),
    ("South Africa", "Canada"),
    ("Brazil", "Japan"),
    ("Germany", "Paraguay"),
    ("Netherlands", "Morocco"),
    ("Ivory Coast", "Norway"),
    ("France", "Sweden"),
    ("Mexico", "Ecuador"),
]


EVALUATION_COLUMNS = [
    "match_id",
    "prediction_id",
    "snapshot_utc",
    "kickoff_utc",
    "home_team",
    "away_team",
    "predicted_home_prob",
    "predicted_draw_prob",
    "predicted_away_prob",
    "predicted_home_xg",
    "predicted_away_xg",
    "model_confidence",
    "behavior_home_prob",
    "behavior_draw_prob",
    "behavior_away_prob",
    "market_home_prob",
    "market_draw_prob",
    "market_away_prob",
    "venue",
    "venue_context",
    "venue_country",
    "neutral_site",
    "altitude",
    "altitude_category",
    "temperature",
    "humidity",
    "wind",
    "actual_home_goals_90",
    "actual_away_goals_90",
    "actual_1x2_result",
    "actual_advancing_team",
    "result_source",
    "result_semantics",
    "market_type",
    "evaluation_scope",
    "predicted_top_class",
    "probability_assigned_to_actual",
    "top_pick_correct",
    "brier_1x2",
    "log_loss_1x2",
    "favorite_probability",
    "favorite_probability_band",
    "market_gap_size",
    "market_gap_band",
    "behavior_disagreement_magnitude",
    "behavior_disagreement_band",
    "market_prior_closer",
    "behavior_warning",
    "venue_environment_note",
]


VARIANT_NAMES = [
    "baseline_model",
    "model_plus_shrinkage",
    "model_plus_market_prior",
    "behavior_gated_adjustment",
    "venue_calibration",
    "full_calibrated_model",
]


def build_calibration_evaluation_dataset(
    prediction_ledger_df: pd.DataFrame | None,
    results_ledger_df: pd.DataFrame | None,
    venues_df: pd.DataFrame | None = None,
    market_odds_df: pd.DataFrame | None = None,
    only_pre_kickoff: bool = True,
    latest_snapshot_only: bool = True,
) -> pd.DataFrame:
    predictions = _ensure_columns(prediction_ledger_df, PREDICTION_LEDGER_COLUMNS)
    if latest_snapshot_only:
        predictions = select_latest_valid_snapshots(
            predictions,
            require_pre_kickoff=only_pre_kickoff,
            exclude_unresolved_teams=True,
        )
    results = _ensure_columns(results_ledger_df, RESULTS_LEDGER_COLUMNS + ["actual_advancing_team", "result_semantics"])
    if predictions.empty or results.empty:
        return pd.DataFrame(columns=EVALUATION_COLUMNS)

    merged = predictions.merge(results, on="match_id", how="inner", suffixes=("", "_result"))
    if merged.empty:
        return pd.DataFrame(columns=EVALUATION_COLUMNS)

    snapshot_ts = pd.to_datetime(merged["snapshot_utc"], errors="coerce", utc=True)
    kickoff_ts = pd.to_datetime(merged["kickoff_utc"], errors="coerce", utc=True)
    before_kickoff = snapshot_ts.notna() & kickoff_ts.notna() & (snapshot_ts < kickoff_ts)
    if "prediction_before_kickoff" in merged.columns:
        before_kickoff = before_kickoff & merged["prediction_before_kickoff"].map(coerce_bool)
    merged = merged.assign(_snapshot_ts=snapshot_ts, _kickoff_ts=kickoff_ts, prediction_before_kickoff=before_kickoff)
    if only_pre_kickoff:
        merged = merged.loc[merged["prediction_before_kickoff"]].copy()
    if merged.empty:
        return pd.DataFrame(columns=EVALUATION_COLUMNS)

    venue_lookup = _venue_lookup(venues_df)
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        home = _first_text(row, ["home", "home_result"])
        away = _first_text(row, ["away", "away_result"])
        venue_name = _first_text(row, ["venue"])
        venue_row = venue_lookup.get(venue_name.strip().lower(), {})
        altitude = _first_numeric(row, ["altitude", "altitude_m"], venue_row.get("altitude_m", pd.NA))
        temperature = _first_numeric(row, ["temperature", "temp_c"], venue_row.get("temp_c", pd.NA))
        humidity = _first_numeric(row, ["humidity", "humidity_pct"], venue_row.get("humidity_pct", pd.NA))
        wind = _first_numeric(row, ["wind", "wind_kmh"], venue_row.get("wind_kmh", pd.NA))
        venue_country = _first_text(row, ["venue_country"], venue_row.get("country", ""))
        neutral_site = row.get("neutral_site", venue_row.get("neutral_site", pd.NA))
        venue_context = _first_text(
            row,
            ["venue_context"],
            classify_venue_context(home, away, {**venue_row, "country": venue_country, "neutral_site": neutral_site}),
        )

        home_prob, draw_prob, away_prob = _normalise_probs(
            [
                coerce_float(row.get("home_win_prob"), float("nan")),
                coerce_float(row.get("draw_prob"), float("nan")),
                coerce_float(row.get("away_win_prob"), float("nan")),
            ]
        )
        market_home, market_draw, market_away = _market_probs_from_sources(row, market_odds_df)
        behavior_home = _optional_prob(row.get("behavior_home_prob"))
        behavior_draw = _optional_prob(row.get("behavior_draw_prob"))
        behavior_away = _optional_prob(row.get("behavior_away_prob"))
        actual = _first_text(row, ["actual_result"])
        actual_prob = _prob_for_actual([home_prob, draw_prob, away_prob], actual)
        brier = brier_1x2([home_prob, draw_prob, away_prob], actual)
        log_loss = safe_log_loss(actual_prob)
        top_class = _top_class([home_prob, draw_prob, away_prob])
        favorite_prob = float(max(home_prob, away_prob))
        behavior_disagreement = _max_abs_gap(
            [home_prob, draw_prob, away_prob],
            [behavior_home, behavior_draw, behavior_away],
        )
        market_gap = _max_abs_gap([home_prob, draw_prob, away_prob], [market_home, market_draw, market_away])
        market_actual_prob = _prob_for_actual([market_home, market_draw, market_away], actual)
        market_prior_closer = (
            bool(pd.notna(market_actual_prob) and pd.notna(actual_prob) and float(market_actual_prob) > float(actual_prob))
            if pd.notna(market_actual_prob) and pd.notna(actual_prob)
            else pd.NA
        )
        result_semantics = _first_text(row, ["result_semantics"], "90-minute regular time")
        market_type = _first_text(row, ["market_type"], "1X2")
        evaluation_scope = market_type_semantics(market_type, result_semantics)
        rows.append(
            {
                "match_id": row.get("match_id", ""),
                "prediction_id": row.get("prediction_id", ""),
                "snapshot_utc": row.get("snapshot_utc", ""),
                "kickoff_utc": row.get("kickoff_utc", ""),
                "home_team": home,
                "away_team": away,
                "predicted_home_prob": home_prob,
                "predicted_draw_prob": draw_prob,
                "predicted_away_prob": away_prob,
                "predicted_home_xg": coerce_float(row.get("home_xg"), float("nan")),
                "predicted_away_xg": coerce_float(row.get("away_xg"), float("nan")),
                "model_confidence": _first_text(row, ["model_confidence"], "Unknown"),
                "behavior_home_prob": behavior_home,
                "behavior_draw_prob": behavior_draw,
                "behavior_away_prob": behavior_away,
                "market_home_prob": market_home,
                "market_draw_prob": market_draw,
                "market_away_prob": market_away,
                "venue": venue_name,
                "venue_context": venue_context,
                "venue_country": venue_country,
                "neutral_site": neutral_site,
                "altitude": altitude,
                "altitude_category": altitude_category(altitude),
                "temperature": temperature,
                "humidity": humidity,
                "wind": wind,
                "actual_home_goals_90": coerce_float(row.get("home_goals"), float("nan")),
                "actual_away_goals_90": coerce_float(row.get("away_goals"), float("nan")),
                "actual_1x2_result": actual,
                "actual_advancing_team": row.get("actual_advancing_team", pd.NA),
                "result_source": row.get("result_source", ""),
                "result_semantics": result_semantics,
                "market_type": market_type,
                "evaluation_scope": evaluation_scope,
                "predicted_top_class": top_class,
                "probability_assigned_to_actual": actual_prob,
                "top_pick_correct": bool(top_class == actual),
                "brier_1x2": brier,
                "log_loss_1x2": log_loss,
                "favorite_probability": favorite_prob,
                "favorite_probability_band": favorite_probability_band(favorite_prob),
                "market_gap_size": market_gap,
                "market_gap_band": gap_band(market_gap),
                "behavior_disagreement_magnitude": behavior_disagreement,
                "behavior_disagreement_band": gap_band(behavior_disagreement),
                "market_prior_closer": market_prior_closer,
                "behavior_warning": behavior_warning(behavior_disagreement),
                "venue_environment_note": venue_environment_note(venue_context, altitude),
            }
        )

    return pd.DataFrame(rows, columns=EVALUATION_COLUMNS)


def market_type_semantics(market_type: str, result_semantics: str = "") -> str:
    text = f"{market_type} {result_semantics}".lower()
    if any(token in text for token in ["advance", "advancing", "to qualify", "winner market", "tournament winner"]):
        return "advancement-inclusive"
    if any(token in text for token in ["extra time", "aet", "penalties", "shootout"]):
        return "extra-time-inclusive"
    return "90-minute"


def calibration_metrics(evaluation_df: pd.DataFrame | None, probability_columns: list[str] | None = None, model_label: str = "baseline_model") -> pd.DataFrame:
    df = _valid_1x2_rows(evaluation_df)
    columns = probability_columns or ["predicted_home_prob", "predicted_draw_prob", "predicted_away_prob"]
    if df.empty or not set(columns).issubset(df.columns):
        return pd.DataFrame(columns=["model", "n", "top_pick_accuracy", "brier_score", "multiclass_log_loss", "expected_calibration_error", "mean_actual_result_probability"])
    probs = df[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    actual = df["actual_1x2_result"].astype(str).tolist()
    actual_probs = [_prob_for_actual(prob, outcome) for prob, outcome in zip(probs, actual)]
    top = [OUTCOMES[int(np.nanargmax(prob))] if np.isfinite(prob).all() else "" for prob in probs]
    return pd.DataFrame(
        [
            {
                "model": model_label,
                "n": int(len(df)),
                "top_pick_accuracy": float(np.mean([pred == outcome for pred, outcome in zip(top, actual)])),
                "brier_score": float(np.mean([brier_1x2(prob, outcome) for prob, outcome in zip(probs, actual)])),
                "multiclass_log_loss": float(np.mean([safe_log_loss(prob) for prob in actual_probs])),
                "expected_calibration_error": expected_calibration_error(df, columns),
                "mean_actual_result_probability": float(np.nanmean(actual_probs)),
            }
        ]
    )


def reliability_curve(evaluation_df: pd.DataFrame | None, probability_columns: list[str] | None = None, model_label: str = "baseline_model") -> pd.DataFrame:
    df = _valid_1x2_rows(evaluation_df)
    columns = probability_columns or ["predicted_home_prob", "predicted_draw_prob", "predicted_away_prob"]
    rows: list[dict[str, Any]] = []
    if df.empty or not set(columns).issubset(df.columns):
        return pd.DataFrame(columns=["model", "outcome", "bucket", "n", "mean_predicted_probability", "observed_frequency", "calibration_error"])
    for outcome, column in zip(OUTCOMES, columns):
        frame = pd.DataFrame(
            {
                "prob": pd.to_numeric(df[column], errors="coerce"),
                "actual": (df["actual_1x2_result"].astype(str) == outcome).astype(float),
            }
        ).dropna(subset=["prob", "actual"])
        if frame.empty:
            continue
        frame["bucket"] = pd.cut(frame["prob"].clip(0.0, 1.0), bins=PROBABILITY_BINS, labels=PROBABILITY_BIN_LABELS, include_lowest=True)
        grouped = frame.groupby("bucket", observed=False).agg(
            n=("actual", "size"),
            mean_predicted_probability=("prob", "mean"),
            observed_frequency=("actual", "mean"),
        )
        grouped["calibration_error"] = (grouped["mean_predicted_probability"] - grouped["observed_frequency"]).abs()
        for bucket, row in grouped.reset_index().iterrows():
            if int(row["n"]) <= 0:
                continue
            rows.append(
                {
                    "model": model_label,
                    "outcome": outcome,
                    "bucket": str(row["bucket"]),
                    "n": int(row["n"]),
                    "mean_predicted_probability": float(row["mean_predicted_probability"]),
                    "observed_frequency": float(row["observed_frequency"]),
                    "calibration_error": float(row["calibration_error"]),
                }
            )
    return pd.DataFrame(rows)


def expected_calibration_error(evaluation_df: pd.DataFrame | None, probability_columns: list[str] | None = None) -> float:
    curve = reliability_curve(evaluation_df, probability_columns)
    if curve.empty:
        return float("nan")
    total = pd.to_numeric(curve["n"], errors="coerce").sum()
    if total <= 0:
        return float("nan")
    return float((pd.to_numeric(curve["n"], errors="coerce") * pd.to_numeric(curve["calibration_error"], errors="coerce")).sum() / total)


def performance_by_dimension(evaluation_df: pd.DataFrame | None, dimension: str) -> pd.DataFrame:
    df = _valid_1x2_rows(evaluation_df)
    if df.empty or dimension not in df.columns:
        return pd.DataFrame(columns=[dimension, "n", "top_pick_accuracy", "brier_score", "multiclass_log_loss"])
    rows = []
    for value, group in df.groupby(dimension, dropna=False):
        metrics = calibration_metrics(group)
        if metrics.empty:
            continue
        row = metrics.iloc[0].to_dict()
        rows.append(
            {
                dimension: value,
                "n": row["n"],
                "top_pick_accuracy": row["top_pick_accuracy"],
                "brier_score": row["brier_score"],
                "multiclass_log_loss": row["multiclass_log_loss"],
            }
        )
    return pd.DataFrame(rows).sort_values(["n", dimension], ascending=[False, True]).reset_index(drop=True) if rows else pd.DataFrame()


def build_calibration_backtest_report(evaluation_df: pd.DataFrame | None, min_train: int = 30) -> dict[str, pd.DataFrame | dict[str, Any]]:
    evaluation = _valid_1x2_rows(evaluation_df)
    baseline = calibration_metrics(evaluation)
    reliability = reliability_curve(evaluation)
    variants, variant_predictions, diagnostics = walk_forward_calibration(evaluation, min_train=min_train)
    probability_policy = calibration_probability_policy(diagnostics, variants)
    return {
        "evaluation": evaluation,
        "summary": baseline,
        "reliability": reliability,
        "by_confidence": performance_by_dimension(evaluation, "model_confidence"),
        "by_favorite_probability": performance_by_dimension(evaluation, "favorite_probability_band"),
        "by_market_gap": performance_by_dimension(evaluation, "market_gap_band"),
        "by_venue": performance_by_dimension(evaluation, "venue_context"),
        "by_altitude": performance_by_dimension(evaluation, "altitude_category"),
        "by_behavior_disagreement": performance_by_dimension(evaluation, "behavior_disagreement_band"),
        "variant_metrics": variants,
        "variant_predictions": variant_predictions,
        "diagnostics": diagnostics,
        "probability_policy": probability_policy,
        "diagnostic_matches": diagnostic_match_report(evaluation),
    }


def walk_forward_calibration(evaluation_df: pd.DataFrame | None, min_train: int = 30) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    df = _valid_1x2_rows(evaluation_df)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), {"status": "empty", "warning": "No valid 90-minute 1X2 evaluation rows."}
    df = df.copy()
    df["_kickoff_ts"] = pd.to_datetime(df["kickoff_utc"], errors="coerce", utc=True)
    df = df.sort_values(["_kickoff_ts", "snapshot_utc", "match_id"]).reset_index(drop=True)
    if len(df) <= min_train:
        return (
            pd.DataFrame(columns=["model", "n", "top_pick_accuracy", "brier_score", "multiclass_log_loss", "expected_calibration_error", "mean_actual_result_probability", "brier_delta_vs_baseline", "log_loss_delta_vs_baseline"]),
            pd.DataFrame(),
            {
                "status": "insufficient_history",
                "rows": int(len(df)),
                "minimum_training_rows": int(min_train),
                "warning": "Calibration sample too small; using conservative shrinkage.",
            },
        )

    prediction_rows: list[dict[str, Any]] = []
    for idx in range(int(min_train), len(df)):
        train = df.iloc[:idx].copy()
        row = df.iloc[idx]
        fitted = _fit_layers(train)
        raw = _row_probs(row, "predicted")
        variants = _variant_probabilities(row, raw, fitted)
        for name, probs in variants.items():
            actual = str(row.get("actual_1x2_result", ""))
            prediction_rows.append(
                {
                    "model": name,
                    "match_id": row.get("match_id", ""),
                    "kickoff_utc": row.get("kickoff_utc", ""),
                    "home_team": row.get("home_team", ""),
                    "away_team": row.get("away_team", ""),
                    "actual_1x2_result": actual,
                    "home_prob": probs[0],
                    "draw_prob": probs[1],
                    "away_prob": probs[2],
                    "brier_1x2": brier_1x2(probs, actual),
                    "log_loss_1x2": safe_log_loss(_prob_for_actual(probs, actual)),
                    "top_pick_correct": _top_class(probs) == actual,
                    "probability_assigned_to_actual": _prob_for_actual(probs, actual),
                }
            )
    predictions = pd.DataFrame(prediction_rows)
    metrics = _variant_metrics_from_predictions(predictions)
    diagnostics = {
        "status": "ok" if not metrics.empty else "no_scored_variants",
        "rows": int(len(df)),
        "minimum_training_rows": int(min_train),
        "walk_forward_scored_rows": int(predictions["match_id"].nunique()) if not predictions.empty else 0,
        "warning": "",
    }
    return metrics, predictions, diagnostics


def calibration_probability_policy(diagnostics: dict[str, Any] | None, variants: pd.DataFrame | None) -> dict[str, str]:
    status = str((diagnostics or {}).get("status", "unknown") or "unknown")
    policy = {
        "primary_probability_source": "Raw baseline external-calibrated model probabilities",
        "calibrated_probability_status": "Not active in production tables",
        "production_gate": "Calibrated variants must beat the raw baseline on both Brier score and log loss before promotion.",
    }
    if status == "insufficient_history":
        policy["calibrated_probability_status"] = "Unavailable: calibration sample too small; using conservative shrinkage."
    elif status == "empty":
        policy["calibrated_probability_status"] = "Unavailable: no valid completed pre-kickoff 90-minute 1X2 rows."
    elif status == "ok" and variants is not None and not variants.empty:
        policy["calibrated_probability_status"] = "Candidate walk-forward variants are scored for review only."
    elif status == "no_scored_variants":
        policy["calibrated_probability_status"] = "Unavailable: no walk-forward variant rows were scored."
    return policy


def diagnostic_match_report(evaluation_df: pd.DataFrame | None, matchups: list[tuple[str, str]] | None = None) -> pd.DataFrame:
    df = _valid_1x2_rows(evaluation_df)
    columns = [
        "match",
        "match_id",
        "model_probabilities",
        "actual_90_min_result",
        "predicted_top_class",
        "probability_assigned_to_actual",
        "brier_1x2",
        "log_loss_1x2",
        "market_prior_closer",
        "behavior_warning",
        "venue_environment_note",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)
    wanted = matchups or DEFAULT_DIAGNOSTIC_MATCHES
    rows = []
    for home, away in wanted:
        pair = {home.strip().lower(), away.strip().lower()}
        matches = df.loc[
            df.apply(lambda row: {str(row.get("home_team", "")).lower(), str(row.get("away_team", "")).lower()} == pair, axis=1)
        ].copy()
        if matches.empty:
            rows.append(
                {
                    "match": f"{home} vs {away}",
                    "match_id": "",
                    "model_probabilities": "not in joined completed pre-kickoff predictions",
                    "actual_90_min_result": "",
                    "predicted_top_class": "",
                    "probability_assigned_to_actual": pd.NA,
                    "brier_1x2": pd.NA,
                    "log_loss_1x2": pd.NA,
                    "market_prior_closer": pd.NA,
                    "behavior_warning": "",
                    "venue_environment_note": "",
                }
            )
            continue
        row = matches.sort_values("kickoff_utc").iloc[-1]
        rows.append(
            {
                "match": f"{row.get('home_team', '')} vs {row.get('away_team', '')}",
                "match_id": row.get("match_id", ""),
                "model_probabilities": (
                    f"H {coerce_float(row.get('predicted_home_prob'), 0.0):.1%} / "
                    f"D {coerce_float(row.get('predicted_draw_prob'), 0.0):.1%} / "
                    f"A {coerce_float(row.get('predicted_away_prob'), 0.0):.1%}"
                ),
                "actual_90_min_result": row.get("actual_1x2_result", ""),
                "predicted_top_class": row.get("predicted_top_class", ""),
                "probability_assigned_to_actual": row.get("probability_assigned_to_actual", pd.NA),
                "brier_1x2": row.get("brier_1x2", pd.NA),
                "log_loss_1x2": row.get("log_loss_1x2", pd.NA),
                "market_prior_closer": row.get("market_prior_closer", pd.NA),
                "behavior_warning": row.get("behavior_warning", ""),
                "venue_environment_note": row.get("venue_environment_note", ""),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def write_calibration_artifacts(
    evaluation_df: pd.DataFrame,
    report: dict[str, Any],
    reports_dir: str | Path = "reports",
    report_date: str | None = None,
) -> dict[str, Path]:
    out_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = report_date or today_iso()
    performance_path = out_dir / f"prediction_performance_{label}.csv"
    markdown_path = out_dir / f"calibration_report_{label}.md"
    evaluation_df.to_csv(performance_path, index=False)
    markdown_path.write_text(render_calibration_markdown(evaluation_df, report, label), encoding="utf-8")
    return {"performance_csv": performance_path, "markdown_report": markdown_path}


def render_calibration_markdown(evaluation_df: pd.DataFrame, report: dict[str, Any], report_date: str) -> str:
    summary = _printable(report.get("summary", pd.DataFrame()))
    variants = _printable(report.get("variant_metrics", pd.DataFrame()))
    diagnostics = report.get("diagnostics", {}) if isinstance(report.get("diagnostics"), dict) else {}
    probability_policy = report.get("probability_policy", {}) if isinstance(report.get("probability_policy"), dict) else {}
    diagnostic_matches = _printable(report.get("diagnostic_matches", pd.DataFrame()))
    market_statement = _market_prior_statement(variants)
    probability_policy_df = pd.DataFrame(
        [
            {"item": "Primary probability source", "status": probability_policy.get("primary_probability_source", "Unavailable")},
            {"item": "Calibrated probability status", "status": probability_policy.get("calibrated_probability_status", "Unavailable")},
            {"item": "Production gate", "status": probability_policy.get("production_gate", "Unavailable")},
        ]
    )
    lines = [
        f"# Calibration / Post-Mortem Report - {report_date}",
        "",
        "This report evaluates saved pre-kickoff predictions against completed 90-minute results. It is evaluation-only and does not provide staking, sizing, wallet, order-placement, or investment advice.",
        "",
        f"Evaluation rows: `{len(evaluation_df):,}`",
        f"Walk-forward status: `{diagnostics.get('status', 'unknown')}`",
    ]
    if diagnostics.get("warning"):
        lines.append(f"Warning: {diagnostics['warning']}")
    lines.extend(
        [
            "",
            "## Raw Model Calibration Metrics",
            "",
            _to_markdown(summary),
            "",
            "## Probability Status",
            "",
            _to_markdown(probability_policy_df),
            "",
            "## Walk-Forward Calibration Variants",
            "",
            "These are candidate calibrated probabilities for review only. The primary match tabs stay on raw baseline probabilities unless a variant beats the raw baseline on both Brier score and log loss.",
            "",
            market_statement,
            "",
            _to_markdown(variants),
            "",
            "## Diagnostic Match Set",
            "",
            _to_markdown(diagnostic_matches),
            "",
            "## Notes",
            "",
            "- 1X2 rows use regular time plus stoppage only.",
            "- Advancement, winner, extra-time, and penalty-inclusive semantics are tagged separately and excluded from 1X2 calibration metrics.",
            "- Market probabilities are de-vigged before use in the market-prior blend.",
            "- Behavior disagreement is diagnostic. It gates shrinkage/warnings only when walk-forward history supports the rule.",
        ]
    )
    return "\n".join(lines)


def _fit_layers(train: pd.DataFrame) -> dict[str, Any]:
    base = _base_rates(train)
    return {
        "base_rates": base,
        "shrinkage_weight": _fit_shrinkage_weight(train, base),
        "market_weights": _fit_market_weights(train, base),
        "behavior_gate": _fit_behavior_gate(train, base),
        "venue_weight": _fit_venue_weight(train, base),
        "venue_base_rates": _venue_base_rates(train, base),
    }


def _variant_probabilities(row: pd.Series, raw: list[float], fitted: dict[str, Any]) -> dict[str, list[float]]:
    base = fitted["base_rates"]
    shrink = _blend(raw, base, fitted["shrinkage_weight"])
    market = _apply_market_prior(row, raw, base, fitted["market_weights"], fitted["shrinkage_weight"])
    behavior = _apply_behavior_gate(row, raw, base, fitted["behavior_gate"])
    venue = _apply_venue_calibration(row, raw, fitted["venue_base_rates"], fitted["venue_weight"])
    full = _apply_venue_calibration(row, _apply_behavior_gate(row, market, base, fitted["behavior_gate"]), fitted["venue_base_rates"], fitted["venue_weight"])
    return {
        "baseline_model": raw,
        "model_plus_shrinkage": shrink,
        "model_plus_market_prior": market,
        "behavior_gated_adjustment": behavior,
        "venue_calibration": venue,
        "full_calibrated_model": full,
    }


def _fit_shrinkage_weight(train: pd.DataFrame, base: list[float]) -> float:
    return _best_weight(train, lambda row, weight: _blend(_row_probs(row, "predicted"), base, weight), [i / 20 for i in range(0, 11)])


def _fit_market_weights(train: pd.DataFrame, base: list[float]) -> dict[str, float]:
    market_rows = train.loc[train[["market_home_prob", "market_draw_prob", "market_away_prob"]].notna().all(axis=1)].copy()
    if len(market_rows) < 3:
        return {"model": 1.0, "market": 0.0, "base": 0.0}
    best = {"model": 1.0, "market": 0.0, "base": 0.0}
    best_score = float("inf")
    for market_weight in [i / 10 for i in range(0, 11)]:
        for base_weight in [i / 10 for i in range(0, 11)]:
            model_weight = 1.0 - market_weight - base_weight
            if model_weight < -1e-9:
                continue
            losses = []
            for _, row in market_rows.iterrows():
                probs = _weighted_probs(
                    _row_probs(row, "predicted"),
                    _row_probs(row, "market"),
                    base,
                    model_weight,
                    market_weight,
                    base_weight,
                )
                losses.append(safe_log_loss(_prob_for_actual(probs, str(row.get("actual_1x2_result", "")))))
            score = float(np.mean(losses)) if losses else float("inf")
            if score < best_score:
                best_score = score
                best = {"model": float(model_weight), "market": float(market_weight), "base": float(base_weight)}
    return best


def _fit_behavior_gate(train: pd.DataFrame, base: list[float]) -> dict[str, float]:
    if train["behavior_disagreement_magnitude"].dropna().empty:
        return {"threshold": float("inf"), "weight": 0.0}
    best = {"threshold": float("inf"), "weight": 0.0}
    best_score = float("inf")
    for threshold in [0.05, 0.10, 0.15, 0.20]:
        for weight in [0.0, 0.10, 0.20, 0.30, 0.40]:
            losses = []
            for _, row in train.iterrows():
                raw = _row_probs(row, "predicted")
                disagreement = coerce_float(row.get("behavior_disagreement_magnitude"), float("nan"))
                probs = _blend(raw, base, weight) if pd.notna(disagreement) and disagreement >= threshold else raw
                losses.append(safe_log_loss(_prob_for_actual(probs, str(row.get("actual_1x2_result", "")))))
            score = float(np.mean(losses)) if losses else float("inf")
            if score < best_score:
                best_score = score
                best = {"threshold": float(threshold), "weight": float(weight)}
    return best


def _fit_venue_weight(train: pd.DataFrame, base: list[float]) -> float:
    return _best_weight(train, lambda row, weight: _apply_venue_calibration(row, _row_probs(row, "predicted"), _venue_base_rates(train, base), weight), [i / 20 for i in range(0, 7)])


def _best_weight(train: pd.DataFrame, fn, weights: list[float]) -> float:
    best_weight = 0.0
    best_score = float("inf")
    for weight in weights:
        losses = []
        for _, row in train.iterrows():
            probs = fn(row, weight)
            losses.append(safe_log_loss(_prob_for_actual(probs, str(row.get("actual_1x2_result", "")))))
        score = float(np.mean(losses)) if losses else float("inf")
        if score < best_score:
            best_score = score
            best_weight = float(weight)
    return best_weight


def _base_rates(df: pd.DataFrame) -> list[float]:
    actual = df["actual_1x2_result"].astype(str) if "actual_1x2_result" in df.columns else pd.Series(dtype=str)
    counts = [float((actual == outcome).sum()) + 1.0 for outcome in OUTCOMES]
    total = sum(counts)
    return [count / total for count in counts]


def _venue_base_rates(df: pd.DataFrame, fallback: list[float]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for col in ["venue_context", "altitude_category"]:
        if col not in df.columns:
            continue
        for value, group in df.groupby(col, dropna=False):
            if len(group) >= 3:
                out[f"{col}:{value}"] = _base_rates(group)
    out["fallback"] = fallback
    return out


def _apply_market_prior(row: pd.Series, raw: list[float], base: list[float], weights: dict[str, float], fallback_shrinkage: float) -> list[float]:
    market = _row_probs(row, "market")
    if not all(pd.notna(value) for value in market):
        return _blend(raw, base, fallback_shrinkage)
    return _weighted_probs(raw, market, base, weights.get("model", 1.0), weights.get("market", 0.0), weights.get("base", 0.0))


def _apply_behavior_gate(row: pd.Series, raw: list[float], base: list[float], gate: dict[str, float]) -> list[float]:
    disagreement = coerce_float(row.get("behavior_disagreement_magnitude"), float("nan"))
    if pd.notna(disagreement) and disagreement >= gate.get("threshold", float("inf")):
        return _blend(raw, base, gate.get("weight", 0.0))
    return _normalise_probs(raw)


def _apply_venue_calibration(row: pd.Series, raw: list[float], venue_rates: dict[str, list[float]], weight: float) -> list[float]:
    keys = [
        f"venue_context:{row.get('venue_context', '')}",
        f"altitude_category:{row.get('altitude_category', '')}",
    ]
    targets = [venue_rates[key] for key in keys if key in venue_rates]
    if not targets:
        return _normalise_probs(raw)
    target = np.mean(np.array(targets, dtype=float), axis=0).tolist()
    return _blend(raw, target, weight)


def _variant_metrics_from_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    rows = []
    for name, group in predictions.groupby("model", dropna=False):
        rows.append(
            {
                "model": name,
                "n": int(len(group)),
                "top_pick_accuracy": float(group["top_pick_correct"].astype(bool).mean()),
                "brier_score": float(pd.to_numeric(group["brier_1x2"], errors="coerce").mean()),
                "multiclass_log_loss": float(pd.to_numeric(group["log_loss_1x2"], errors="coerce").mean()),
                "expected_calibration_error": _prediction_ece(group),
                "mean_actual_result_probability": float(pd.to_numeric(group["probability_assigned_to_actual"], errors="coerce").mean()),
            }
        )
    out = pd.DataFrame(rows)
    baseline = out.loc[out["model"] == "baseline_model"]
    if not baseline.empty:
        brier = float(baseline.iloc[0]["brier_score"])
        log_loss = float(baseline.iloc[0]["multiclass_log_loss"])
        out["brier_delta_vs_baseline"] = out["brier_score"] - brier
        out["log_loss_delta_vs_baseline"] = out["multiclass_log_loss"] - log_loss
    return out.sort_values(["multiclass_log_loss", "brier_score", "model"]).reset_index(drop=True)


def _prediction_ece(predictions: pd.DataFrame) -> float:
    rows = []
    for outcome, column in zip(OUTCOMES, ["home_prob", "draw_prob", "away_prob"]):
        part = pd.DataFrame(
            {
                "prob": pd.to_numeric(predictions[column], errors="coerce"),
                "actual": (predictions["actual_1x2_result"].astype(str) == outcome).astype(float),
            }
        )
        rows.append(part)
    frame = pd.concat(rows, ignore_index=True).dropna()
    if frame.empty:
        return float("nan")
    frame["bucket"] = pd.cut(frame["prob"].clip(0, 1), PROBABILITY_BINS, labels=PROBABILITY_BIN_LABELS, include_lowest=True)
    table = frame.groupby("bucket", observed=False).agg(n=("actual", "size"), avg=("prob", "mean"), actual=("actual", "mean"))
    table = table.loc[table["n"] > 0].copy()
    if table.empty:
        return float("nan")
    return float((table["n"] * (table["avg"] - table["actual"]).abs()).sum() / table["n"].sum())


def brier_1x2(probs: list[float] | np.ndarray, actual: str) -> float:
    p = _normalise_probs(list(probs))
    return float(sum((p[idx] - float(actual == outcome)) ** 2 for idx, outcome in enumerate(OUTCOMES)))


def safe_log_loss(probability: Any) -> float:
    prob = coerce_float(probability, 0.0)
    return float(-math.log(max(min(prob, 0.999), 0.001)))


def favorite_probability_band(value: Any) -> str:
    prob = coerce_float(value, float("nan"))
    if pd.isna(prob):
        return "missing"
    if prob < 0.40:
        return "0.00-0.40"
    if prob < 0.50:
        return "0.40-0.50"
    if prob < 0.60:
        return "0.50-0.60"
    if prob < 0.70:
        return "0.60-0.70"
    return "0.70-1.00"


def gap_band(value: Any) -> str:
    gap = coerce_float(value, float("nan"))
    if pd.isna(gap):
        return "missing"
    if gap < 0.05:
        return "0.00-0.05"
    if gap < 0.10:
        return "0.05-0.10"
    if gap < 0.20:
        return "0.10-0.20"
    return "0.20+"


def behavior_warning(disagreement: Any, threshold: float = 0.10) -> str:
    value = coerce_float(disagreement, float("nan"))
    if pd.isna(value):
        return ""
    if value >= threshold:
        return "Behavior disagreement warning: diagnostic-only layer differs enough to require confidence shrinkage review."
    return ""


def venue_environment_note(venue_context: str, altitude: Any) -> str:
    parts = []
    if str(venue_context) in {"home_host", "away_host", "designated_home"}:
        parts.append(f"venue_context={venue_context}")
    if altitude_category(altitude) != "low_altitude":
        parts.append(f"altitude={altitude_category(altitude)}")
    return "; ".join(parts)


def _market_prior_statement(variants: pd.DataFrame) -> str:
    if variants.empty or "model" not in variants.columns:
        return "Walk-forward market-prior comparison was not available, usually because there are too few completed prediction rows."
    baseline = variants.loc[variants["model"] == "baseline_model"]
    market = variants.loc[variants["model"] == "model_plus_market_prior"]
    if baseline.empty or market.empty:
        return "Walk-forward market-prior comparison was not available."
    brier_delta = coerce_float(market.iloc[0].get("brier_delta_vs_baseline"), float("nan"))
    log_delta = coerce_float(market.iloc[0].get("log_loss_delta_vs_baseline"), float("nan"))
    if pd.isna(brier_delta) or pd.isna(log_delta):
        return "Market-prior improvement could not be determined."
    if brier_delta < 0 and log_delta < 0:
        return f"Market-prior blending improved both Brier ({brier_delta:.4f}) and log loss ({log_delta:.4f}) versus baseline."
    return f"Market-prior blending did not improve both headline metrics: Brier delta {brier_delta:.4f}, log-loss delta {log_delta:.4f}."


def _valid_1x2_rows(df: pd.DataFrame | None) -> pd.DataFrame:
    out = df.copy() if df is not None else pd.DataFrame(columns=EVALUATION_COLUMNS)
    for col in EVALUATION_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out.loc[out["evaluation_scope"].astype(str).eq("90-minute")].copy()
    out = out.loc[out["actual_1x2_result"].astype(str).isin(OUTCOMES)].copy()
    return out.reset_index(drop=True)


def _row_probs(row: pd.Series, prefix: str) -> list[float]:
    if prefix == "predicted":
        cols = ["predicted_home_prob", "predicted_draw_prob", "predicted_away_prob"]
    elif prefix == "market":
        cols = ["market_home_prob", "market_draw_prob", "market_away_prob"]
    elif prefix == "behavior":
        cols = ["behavior_home_prob", "behavior_draw_prob", "behavior_away_prob"]
    else:
        cols = [f"{prefix}_home_prob", f"{prefix}_draw_prob", f"{prefix}_away_prob"]
    return _normalise_probs([coerce_float(row.get(col), float("nan")) for col in cols])


def _normalise_probs(values: list[Any]) -> list[float]:
    nums = [coerce_float(value, float("nan")) for value in values]
    if not all(pd.notna(value) and value >= 0 for value in nums):
        return [1 / 3, 1 / 3, 1 / 3]
    total = float(sum(nums))
    if total <= 0:
        return [1 / 3, 1 / 3, 1 / 3]
    return [float(value / total) for value in nums]


def _blend(source: list[float], target: list[float], target_weight: float) -> list[float]:
    weight = min(max(float(target_weight), 0.0), 1.0)
    return _normalise_probs([(1.0 - weight) * source[idx] + weight * target[idx] for idx in range(3)])


def _weighted_probs(model: list[float], market: list[float], base: list[float], model_weight: float, market_weight: float, base_weight: float) -> list[float]:
    return _normalise_probs(
        [
            model_weight * model[idx] + market_weight * market[idx] + base_weight * base[idx]
            for idx in range(3)
        ]
    )


def _prob_for_actual(probs: list[float] | np.ndarray, actual: str) -> float:
    p = _normalise_probs(list(probs))
    mapping = {"home_win": 0, "draw": 1, "away_win": 2}
    idx = mapping.get(str(actual))
    return float(p[idx]) if idx is not None else float("nan")


def _top_class(probs: list[float] | np.ndarray) -> str:
    p = _normalise_probs(list(probs))
    return OUTCOMES[int(np.argmax(p))]


def _optional_prob(value: Any) -> float | pd.NA:
    number = coerce_float(value, float("nan"))
    if pd.isna(number) or number < 0 or number > 1:
        return pd.NA
    return float(number)


def _max_abs_gap(left: list[Any], right: list[Any]) -> float | pd.NA:
    values = []
    for a, b in zip(left, right):
        if pd.notna(a) and pd.notna(b):
            values.append(abs(float(a) - float(b)))
    return max(values) if values else pd.NA


def _market_probs_from_sources(row: pd.Series, market_odds_df: pd.DataFrame | None) -> tuple[Any, Any, Any]:
    direct = [_optional_prob(row.get(col)) for col in ["market_home_prob", "market_draw_prob", "market_away_prob"]]
    if all(pd.notna(value) for value in direct):
        return tuple(_normalise_probs(direct))
    from_odds = market_implied_1x2_probabilities(market_odds_df, row.get("match_id", ""), row.get("home", ""), row.get("away", ""))
    return tuple(from_odds)


def market_implied_1x2_probabilities(market_odds_df: pd.DataFrame | None, match_id: Any, home: Any, away: Any) -> list[Any]:
    odds = market_odds_df.copy() if market_odds_df is not None else pd.DataFrame()
    if odds.empty or not {"match_id", "market", "selection", "odds"}.issubset(odds.columns):
        return [pd.NA, pd.NA, pd.NA]
    selected = odds.loc[(odds["match_id"].astype(str) == str(match_id)) & (odds["market"].astype(str).str.lower() == "1x2")].copy()
    if selected.empty:
        return [pd.NA, pd.NA, pd.NA]
    lookup = {str(row.get("selection", "")).strip().lower(): coerce_float(row.get("odds"), float("nan")) for _, row in selected.iterrows()}
    decimal_odds = [
        lookup.get(str(home).strip().lower(), float("nan")),
        lookup.get("draw", float("nan")),
        lookup.get(str(away).strip().lower(), float("nan")),
    ]
    if not all(pd.notna(value) and value > 1.0 for value in decimal_odds):
        return [pd.NA, pd.NA, pd.NA]
    implied = [1.0 / value for value in decimal_odds]
    return _normalise_probs(implied)


def _first_numeric(row: pd.Series, columns: list[str], default: Any = pd.NA) -> Any:
    for col in columns:
        if col in row.index:
            value = coerce_float(row.get(col), float("nan"))
            if pd.notna(value):
                return value
    value = coerce_float(default, float("nan"))
    return value if pd.notna(value) else pd.NA


def _first_text(row: pd.Series, columns: list[str], default: str = "") -> str:
    for col in columns:
        if col in row.index:
            value = row.get(col)
            if pd.notna(value) and str(value).strip():
                return str(value).strip()
    if default is None or pd.isna(default):
        return ""
    return str(default).strip()


def _venue_lookup(venues_df: pd.DataFrame | None) -> dict[str, dict[str, Any]]:
    venues = venues_df.copy() if venues_df is not None else pd.DataFrame()
    if venues.empty or "venue" not in venues.columns:
        return {}
    return {str(row.get("venue", "")).strip().lower(): row.to_dict() for _, row in venues.iterrows()}


def _ensure_columns(df: pd.DataFrame | None, columns: list[str]) -> pd.DataFrame:
    out = df.copy() if df is not None else pd.DataFrame(columns=columns)
    for col in columns:
        if col not in out.columns:
            out[col] = pd.NA
    return out.copy()


def _printable(df: Any) -> pd.DataFrame:
    out = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda value: "" if pd.isna(value) else round(float(value), 4))
    return out


def _to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        values = [str(row.get(col, "")).replace("|", "\\|") for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
