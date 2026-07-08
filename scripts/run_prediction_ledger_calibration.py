from __future__ import annotations

import argparse
import math
import re
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.prediction_ledger import select_latest_valid_snapshots  # noqa: E402


OUTCOMES = ["home", "draw", "away"]
OUTCOME_PROB_COLUMNS = {
    "home": "home_win_prob",
    "draw": "draw_prob",
    "away": "away_win_prob",
}
BINARY_MARKETS = [
    ("over_2_5", "over_2_5_prob", "over_2_5_actual"),
    ("over_3_5", "over_3_5_prob", "over_3_5_actual"),
    ("btts_yes", "btts_yes_prob", "btts_actual"),
]
DEDUPE_POLICIES = ["latest_pre_kickoff", "earliest_pre_kickoff", "none"]
UNRESOLVED_TEAM_PATTERNS = [
    r"\bwinner\s",
    r"\bloser\s",
    r"\bwinner\s+group\b",
    r"\b3rd\s+group\b",
    r"\bthird\s+group\b",
    r"\btbd\b",
    r"\bto\s+be\s+determined\b",
]
UNRESOLVED_TEAM_RE = re.compile("|".join(UNRESOLVED_TEAM_PATTERNS), re.IGNORECASE)
RESOLVED_EVENT_COLUMNS = [
    "prediction_id",
    "snapshot_utc",
    "match_id",
    "kickoff_utc",
    "home",
    "away",
    "selection",
    "model_probability",
    "actual",
    "probability_bucket",
    "dedupe_policy",
    "n_snapshots_for_match",
    "snapshot_rank_for_match",
]
DUPE_AUDIT_COLUMNS = [
    "match_id",
    "home",
    "away",
    "n_snapshots",
    "earliest_snapshot_utc",
    "latest_snapshot_utc",
    "min_top_pick_probability",
    "max_top_pick_probability",
    "actual_result",
]
CALIBRATION_COLUMNS = [
    "section",
    "market",
    "selection",
    "probability_bucket",
    "n",
    "avg_predicted_probability",
    "actual_hit_rate",
    "calibration_error",
    "brier_score",
    "log_loss",
    "top_pick_accuracy",
    "average_top_pick_probability",
    "notes",
]
BUCKET_LABELS = [f"{i / 10:.2f}\u2013{(i + 1) / 10:.2f}" for i in range(10)]


def main() -> None:
    args = _parse_args()
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_date = str(args.report_date or date.today().isoformat())
    output_label = _output_label(args.dedupe_policy, report_date)

    predictions = _read_csv(Path(args.predictions))
    raw_results = _read_csv(Path(args.results_ledger))
    results = _deduplicate_results(raw_results)
    joined = _join_predictions_to_results(predictions, results)
    pre_kickoff = _pre_kickoff_rows(joined)
    concrete_pre_kickoff, unresolved_removed = _filter_unresolved_teams(
        pre_kickoff,
        include_unresolved=bool(args.include_unresolved_teams),
    )
    concrete_pre_kickoff = _with_snapshot_metadata(concrete_pre_kickoff)
    duplicate_audit = _duplicate_audit(concrete_pre_kickoff)
    deduped = (
        select_latest_valid_snapshots(
            joined,
            require_pre_kickoff=True,
            exclude_unresolved_teams=not bool(args.include_unresolved_teams),
        )
        if args.dedupe_policy == "latest_pre_kickoff"
        else _deduplicate_predictions(concrete_pre_kickoff, args.dedupe_policy)
    )

    evaluated = _valid_1x2_rows(deduped)
    summary = _summary_counts(
        predictions,
        raw_results,
        results,
        joined,
        pre_kickoff,
        concrete_pre_kickoff,
        deduped,
        evaluated,
        unresolved_removed,
        args.dedupe_policy,
        include_unresolved_teams=bool(args.include_unresolved_teams),
    )
    one_x_two_metrics = _one_x_two_metrics(evaluated)
    summary.update(one_x_two_metrics)

    resolved_events = _resolved_1x2_events(evaluated, args.dedupe_policy)
    calibration = _calibration_rows(summary, evaluated, resolved_events)
    binary_summaries = _binary_market_summaries(deduped)
    for market_summary in binary_summaries:
        calibration = pd.concat([calibration, market_summary["rows"]], ignore_index=True)

    resolved_path = reports_dir / f"prediction_ledger_resolved_events_{output_label}.csv"
    calibration_path = reports_dir / f"prediction_ledger_calibration_{output_label}.csv"
    duplicate_audit_path = reports_dir / f"prediction_ledger_duplicate_audit_{output_label}.csv"
    report_path = reports_dir / f"prediction_ledger_calibration_report_{output_label}.md"

    resolved_events.to_csv(resolved_path, index=False)
    calibration.to_csv(calibration_path, index=False)
    duplicate_audit.to_csv(duplicate_audit_path, index=False)
    report_path.write_text(
        _render_report(
            summary,
            calibration,
            binary_summaries,
            resolved_path,
            calibration_path,
            duplicate_audit_path,
            report_date,
        ),
        encoding="utf-8",
    )

    _print_summary(summary, binary_summaries, resolved_path, calibration_path, duplicate_audit_path, report_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate prediction_ledger.csv pre-kickoff snapshots against results_ledger.csv."
    )
    parser.add_argument("--predictions", default=str(PROJECT_ROOT / "data" / "prediction_ledger.csv"))
    parser.add_argument("--results-ledger", default=str(PROJECT_ROOT / "data" / "results_ledger.csv"))
    parser.add_argument("--reports-dir", default=str(PROJECT_ROOT / "reports"))
    parser.add_argument("--report-date", default=None, help="Date suffix for report files, YYYY-MM-DD.")
    parser.add_argument(
        "--dedupe-policy",
        choices=DEDUPE_POLICIES,
        default="latest_pre_kickoff",
        help="How to select one concrete pre-kickoff snapshot per match for calibration.",
    )
    parser.add_argument(
        "--include-unresolved-teams",
        action="store_true",
        help="Keep placeholder team rows such as Winner Group, 3rd Group, and TBD in calibration.",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _output_label(dedupe_policy: str, report_date: str) -> str:
    policy = re.sub(r"[^a-z0-9_]+", "_", str(dedupe_policy).strip().lower()).strip("_")
    policy = policy or "unknown"
    return f"{policy}_{report_date}"


def _deduplicate_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty or "match_id" not in results.columns:
        return results.copy()
    out = results.copy()
    out["match_id"] = out["match_id"].astype(str).str.strip()
    out = out.loc[out["match_id"] != ""].copy()
    if "last_updated" in out.columns:
        out["_last_updated_sort"] = pd.to_datetime(out["last_updated"], errors="coerce", utc=True)
        out = out.sort_values(["match_id", "_last_updated_sort"], na_position="first")
        out = out.drop(columns=["_last_updated_sort"])
    return out.drop_duplicates(subset=["match_id"], keep="last").reset_index(drop=True)


def _join_predictions_to_results(predictions: pd.DataFrame, results: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or results.empty or "match_id" not in predictions.columns or "match_id" not in results.columns:
        return pd.DataFrame()
    left = predictions.copy()
    right = results.copy()
    left["match_id"] = left["match_id"].astype(str).str.strip()
    right["match_id"] = right["match_id"].astype(str).str.strip()
    return left.merge(right, on="match_id", how="inner", suffixes=("", "_result"))


def _pre_kickoff_rows(joined: pd.DataFrame) -> pd.DataFrame:
    if joined.empty or "prediction_before_kickoff" not in joined.columns:
        return pd.DataFrame(columns=joined.columns)
    mask = joined["prediction_before_kickoff"].map(_truthy)
    return joined.loc[mask].copy()


def _filter_unresolved_teams(rows: pd.DataFrame, *, include_unresolved: bool) -> tuple[pd.DataFrame, int]:
    if include_unresolved or rows.empty:
        return rows.copy(), 0
    if "home" not in rows.columns or "away" not in rows.columns:
        return rows.copy(), 0
    home_unresolved = rows["home"].map(_is_unresolved_team)
    away_unresolved = rows["away"].map(_is_unresolved_team)
    unresolved = home_unresolved | away_unresolved
    return rows.loc[~unresolved].copy(), int(unresolved.sum())


def _is_unresolved_team(value: Any) -> bool:
    if pd.isna(value):
        return False
    text = re.sub(r"\s+", " ", str(value).strip().lower())
    return bool(UNRESOLVED_TEAM_RE.search(f"{text} "))


def _with_snapshot_metadata(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        out = rows.copy()
        out["n_snapshots_for_match"] = pd.Series(dtype="int64")
        out["snapshot_rank_for_match"] = pd.Series(dtype="int64")
        return out
    out = rows.copy()
    if "match_id" not in out.columns:
        out["n_snapshots_for_match"] = 1
        out["snapshot_rank_for_match"] = 1
        return out
    out["_original_order"] = range(len(out))
    out["_snapshot_sort"] = pd.to_datetime(out.get("snapshot_utc", pd.Series(pd.NA, index=out.index)), errors="coerce", utc=True)
    ordered = out.sort_values(["match_id", "_snapshot_sort", "_original_order"], na_position="last").copy()
    ordered["n_snapshots_for_match"] = ordered.groupby("match_id")["match_id"].transform("size").astype(int)
    ordered["snapshot_rank_for_match"] = ordered.groupby("match_id").cumcount().add(1).astype(int)
    out = ordered.sort_values("_original_order").drop(columns=["_snapshot_sort", "_original_order"])
    return out.reset_index(drop=True)


def _deduplicate_predictions(rows: pd.DataFrame, dedupe_policy: str) -> pd.DataFrame:
    if rows.empty or dedupe_policy == "none" or "match_id" not in rows.columns:
        return rows.copy()
    out = rows.copy()
    out["_original_order"] = range(len(out))
    out["_snapshot_sort"] = pd.to_datetime(out.get("snapshot_utc", pd.Series(pd.NA, index=out.index)), errors="coerce", utc=True)
    ordered = out.sort_values(["match_id", "_snapshot_sort", "_original_order"], na_position="last")
    if dedupe_policy == "latest_pre_kickoff":
        selected = ordered.drop_duplicates(subset=["match_id"], keep="last")
    elif dedupe_policy == "earliest_pre_kickoff":
        selected = ordered.drop_duplicates(subset=["match_id"], keep="first")
    else:
        selected = ordered
    return selected.drop(columns=["_snapshot_sort", "_original_order"]).reset_index(drop=True)


def _truthy(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _valid_1x2_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    required = list(OUTCOME_PROB_COLUMNS.values()) + ["actual_result"]
    if any(col not in rows.columns for col in required):
        return pd.DataFrame(columns=rows.columns)
    out = rows.copy()
    out["actual_result_normalized"] = out["actual_result"].map(_normalize_actual_result)
    for col in OUTCOME_PROB_COLUMNS.values():
        out[col] = pd.to_numeric(out[col], errors="coerce")
    valid_probs = pd.Series(True, index=out.index)
    for col in OUTCOME_PROB_COLUMNS.values():
        valid_probs &= out[col].between(0, 1, inclusive="both")
    valid_actual = out["actual_result_normalized"].isin(OUTCOMES)
    return out.loc[valid_probs & valid_actual].copy()


def _normalize_actual_result(value: Any) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    canonical = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    compact = canonical.replace("_", "")
    if canonical in {"home", "home_win", "h"} or compact in {"home", "homewin", "h"}:
        return "home"
    if canonical in {"draw", "d"} or compact in {"draw", "d"}:
        return "draw"
    if canonical in {"away", "away_win", "a"} or compact in {"away", "awaywin", "a"}:
        return "away"
    return None


def _duplicate_audit(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty or "match_id" not in rows.columns:
        return pd.DataFrame(columns=DUPE_AUDIT_COLUMNS)
    out = rows.copy()
    out["_original_order"] = range(len(out))
    out["_top_pick_probability"] = _top_pick_probabilities(out)
    out["_snapshot_sort"] = pd.to_datetime(out.get("snapshot_utc", pd.Series(pd.NA, index=out.index)), errors="coerce", utc=True)
    out = out.sort_values(["match_id", "_snapshot_sort", "_original_order"], na_position="last")
    grouped = (
        out.groupby("match_id", dropna=False)
        .agg(
            home=("home", "first"),
            away=("away", "first"),
            n_snapshots=("match_id", "size"),
            earliest_snapshot_utc=("snapshot_utc", _first_nonempty),
            latest_snapshot_utc=("snapshot_utc", _last_nonempty),
            min_top_pick_probability=("_top_pick_probability", "min"),
            max_top_pick_probability=("_top_pick_probability", "max"),
            actual_result=("actual_result", "first"),
        )
        .reset_index()
    )
    grouped = grouped.loc[grouped["n_snapshots"] > 1].copy()
    if grouped.empty:
        return pd.DataFrame(columns=DUPE_AUDIT_COLUMNS)
    return grouped.loc[:, DUPE_AUDIT_COLUMNS].sort_values(["n_snapshots", "match_id"], ascending=[False, True]).reset_index(drop=True)


def _top_pick_probabilities(rows: pd.DataFrame) -> pd.Series:
    if rows.empty or any(col not in rows.columns for col in OUTCOME_PROB_COLUMNS.values()):
        return pd.Series(pd.NA, index=rows.index, dtype="float64")
    probs = rows[[OUTCOME_PROB_COLUMNS[outcome] for outcome in OUTCOMES]].apply(pd.to_numeric, errors="coerce")
    return probs.max(axis=1, skipna=True)


def _first_nonempty(values: pd.Series) -> str:
    ordered = values.dropna().astype(str)
    ordered = ordered.loc[ordered.str.strip() != ""]
    return str(ordered.iloc[0]) if not ordered.empty else ""


def _last_nonempty(values: pd.Series) -> str:
    ordered = values.dropna().astype(str)
    ordered = ordered.loc[ordered.str.strip() != ""]
    return str(ordered.iloc[-1]) if not ordered.empty else ""


def _summary_counts(
    predictions: pd.DataFrame,
    raw_results: pd.DataFrame,
    results: pd.DataFrame,
    joined: pd.DataFrame,
    pre_kickoff: pd.DataFrame,
    concrete_pre_kickoff: pd.DataFrame,
    deduped: pd.DataFrame,
    evaluated: pd.DataFrame,
    unresolved_removed: int,
    dedupe_policy: str,
    *,
    include_unresolved_teams: bool,
) -> dict[str, Any]:
    duplicate_results = 0
    if not raw_results.empty and "match_id" in raw_results.columns:
        duplicate_results = int(raw_results["match_id"].astype(str).str.strip().duplicated().sum())
    unique_concrete_matches = 0
    if not concrete_pre_kickoff.empty and "match_id" in concrete_pre_kickoff.columns:
        unique_concrete_matches = int(concrete_pre_kickoff["match_id"].astype(str).str.strip().nunique())
    duplicate_snapshot_rows_removed = max(int(len(concrete_pre_kickoff) - len(deduped)), 0)
    return {
        "total_prediction_rows": int(len(predictions)),
        "result_rows": int(len(raw_results)),
        "deduplicated_result_rows": int(len(results)),
        "duplicate_result_match_id_rows": duplicate_results,
        "joined_rows": int(len(joined)),
        "pre_kickoff_joined_rows": int(len(pre_kickoff)),
        "unresolved_placeholder_rows_removed": int(unresolved_removed),
        "concrete_pre_kickoff_snapshot_rows": int(len(concrete_pre_kickoff)),
        "unique_concrete_matched_matches": unique_concrete_matches,
        "duplicate_snapshot_rows_removed": duplicate_snapshot_rows_removed,
        "dedupe_policy": dedupe_policy,
        "unresolved_team_policy": "included" if include_unresolved_teams else "excluded",
        "unresolved_teams_excluded": not include_unresolved_teams,
        "evaluated_rows": int(len(evaluated)),
    }


def _one_x_two_metrics(evaluated: pd.DataFrame) -> dict[str, Any]:
    if evaluated.empty:
        return {
            "brier_score_1x2": math.nan,
            "multiclass_log_loss": math.nan,
            "top_pick_accuracy": math.nan,
            "average_top_pick_probability": math.nan,
            "top_pick_hit_rate": math.nan,
        }
    probs = _normalized_1x2_probabilities(evaluated)
    actual_indices = evaluated["actual_result_normalized"].map({outcome: idx for idx, outcome in enumerate(OUTCOMES)}).to_numpy()
    brier_values = []
    log_losses = []
    top_hits = []
    top_probs = []
    for row_idx, actual_idx in enumerate(actual_indices):
        row_probs = probs[row_idx]
        brier_values.append(float(sum((row_probs[idx] - float(idx == actual_idx)) ** 2 for idx in range(3))))
        log_losses.append(float(-math.log(max(min(row_probs[int(actual_idx)], 0.999), 0.001))))
        top_idx = int(row_probs.argmax())
        top_hits.append(int(top_idx == actual_idx))
        top_probs.append(float(row_probs[top_idx]))
    top_pick_hit_rate = float(pd.Series(top_hits).mean())
    return {
        "brier_score_1x2": float(pd.Series(brier_values).mean()),
        "multiclass_log_loss": float(pd.Series(log_losses).mean()),
        "top_pick_accuracy": top_pick_hit_rate,
        "average_top_pick_probability": float(pd.Series(top_probs).mean()),
        "top_pick_hit_rate": top_pick_hit_rate,
    }


def _normalized_1x2_probabilities(rows: pd.DataFrame) -> Any:
    probs = rows[[OUTCOME_PROB_COLUMNS[outcome] for outcome in OUTCOMES]].astype(float).to_numpy()
    sums = probs.sum(axis=1)
    normalized = probs.copy()
    valid = sums > 0
    normalized[valid] = probs[valid] / sums[valid, None]
    return normalized


def _resolved_1x2_events(evaluated: pd.DataFrame, dedupe_policy: str) -> pd.DataFrame:
    if evaluated.empty:
        return pd.DataFrame(columns=RESOLVED_EVENT_COLUMNS)
    rows: list[dict[str, Any]] = []
    for _, row in evaluated.iterrows():
        actual = row.get("actual_result_normalized")
        for selection in OUTCOMES:
            probability = float(row.get(OUTCOME_PROB_COLUMNS[selection]))
            rows.append(
                {
                    "prediction_id": row.get("prediction_id", ""),
                    "snapshot_utc": row.get("snapshot_utc", ""),
                    "match_id": row.get("match_id", ""),
                    "kickoff_utc": row.get("kickoff_utc", ""),
                    "home": row.get("home", ""),
                    "away": row.get("away", ""),
                    "selection": selection,
                    "model_probability": probability,
                    "actual": int(selection == actual),
                    "probability_bucket": _probability_bucket(probability),
                    "dedupe_policy": dedupe_policy,
                    "n_snapshots_for_match": row.get("n_snapshots_for_match", ""),
                    "snapshot_rank_for_match": row.get("snapshot_rank_for_match", ""),
                }
            )
    return pd.DataFrame(rows, columns=RESOLVED_EVENT_COLUMNS)


def _probability_bucket(probability: Any) -> str:
    try:
        prob = float(probability)
    except (TypeError, ValueError):
        return ""
    if pd.isna(prob) or prob < 0 or prob > 1:
        return ""
    idx = min(int(prob * 10), 9)
    return BUCKET_LABELS[idx]


def _calibration_rows(summary: dict[str, Any], evaluated: pd.DataFrame, resolved_events: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {
            "section": "overall",
            "market": "1x2",
            "selection": "top_pick",
            "probability_bucket": "overall",
            "n": summary["evaluated_rows"],
            "avg_predicted_probability": summary["average_top_pick_probability"],
            "actual_hit_rate": summary["top_pick_hit_rate"],
            "calibration_error": _difference(summary["top_pick_hit_rate"], summary["average_top_pick_probability"]),
            "brier_score": summary["brier_score_1x2"],
            "log_loss": summary["multiclass_log_loss"],
            "top_pick_accuracy": summary["top_pick_accuracy"],
            "average_top_pick_probability": summary["average_top_pick_probability"],
            "notes": "1X2 metrics; top-pick accuracy and top-pick hit rate use the same hit definition.",
        }
    ]
    rows.extend(_bucket_rows(resolved_events, section="outcome_bucket", market="1x2", selection="all_1x2_outcomes"))
    rows.extend(_top_pick_bucket_rows(evaluated))
    return pd.DataFrame(rows, columns=CALIBRATION_COLUMNS)


def _bucket_rows(events: pd.DataFrame, *, section: str, market: str, selection: str) -> list[dict[str, Any]]:
    if events.empty:
        grouped = pd.DataFrame(columns=["probability_bucket", "n", "avg_predicted_probability", "actual_hit_rate"])
    else:
        grouped = (
            events.groupby("probability_bucket", dropna=False)
            .agg(
                n=("actual", "size"),
                avg_predicted_probability=("model_probability", "mean"),
                actual_hit_rate=("actual", "mean"),
            )
            .reset_index()
        )
    lookup = {str(row["probability_bucket"]): row for _, row in grouped.iterrows()}
    rows: list[dict[str, Any]] = []
    for bucket in BUCKET_LABELS:
        row = lookup.get(bucket)
        n = int(row["n"]) if row is not None else 0
        avg_prob = float(row["avg_predicted_probability"]) if row is not None and n else math.nan
        hit_rate = float(row["actual_hit_rate"]) if row is not None and n else math.nan
        rows.append(
            {
                "section": section,
                "market": market,
                "selection": selection,
                "probability_bucket": bucket,
                "n": n,
                "avg_predicted_probability": avg_prob,
                "actual_hit_rate": hit_rate,
                "calibration_error": _difference(hit_rate, avg_prob),
                "brier_score": math.nan,
                "log_loss": math.nan,
                "top_pick_accuracy": math.nan,
                "average_top_pick_probability": math.nan,
                "notes": "",
            }
        )
    return rows


def _top_pick_bucket_rows(evaluated: pd.DataFrame) -> list[dict[str, Any]]:
    if evaluated.empty:
        return _bucket_rows(pd.DataFrame(columns=["probability_bucket", "actual", "model_probability"]), section="top_pick_bucket", market="1x2", selection="top_pick")
    probs = _normalized_1x2_probabilities(evaluated)
    actual_indices = evaluated["actual_result_normalized"].map({outcome: idx for idx, outcome in enumerate(OUTCOMES)}).to_numpy()
    events = []
    for row_idx, actual_idx in enumerate(actual_indices):
        row_probs = probs[row_idx]
        top_idx = int(row_probs.argmax())
        probability = float(row_probs[top_idx])
        events.append(
            {
                "probability_bucket": _probability_bucket(probability),
                "model_probability": probability,
                "actual": int(top_idx == actual_idx),
            }
        )
    return _bucket_rows(pd.DataFrame(events), section="top_pick_bucket", market="1x2", selection="top_pick")


def _binary_market_summaries(pre_kickoff: pd.DataFrame) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for market, prob_col, actual_col in BINARY_MARKETS:
        if pre_kickoff.empty or prob_col not in pre_kickoff.columns or actual_col not in pre_kickoff.columns:
            continue
        scored = pre_kickoff[[prob_col, actual_col]].copy()
        scored["model_probability"] = pd.to_numeric(scored[prob_col], errors="coerce")
        scored["actual"] = scored[actual_col].map(_binary_actual)
        scored = scored.loc[scored["model_probability"].between(0, 1, inclusive="both") & scored["actual"].isin([0, 1])].copy()
        if scored.empty:
            metrics = {
                "n": 0,
                "brier_score": math.nan,
                "log_loss": math.nan,
                "avg_predicted_probability": math.nan,
                "actual_hit_rate": math.nan,
            }
            events = pd.DataFrame(columns=["probability_bucket", "model_probability", "actual"])
        else:
            probabilities = scored["model_probability"].astype(float)
            actual = scored["actual"].astype(int)
            clipped = probabilities.clip(0.001, 0.999)
            metrics = {
                "n": int(len(scored)),
                "brier_score": float(((probabilities - actual) ** 2).mean()),
                "log_loss": float((-(actual * clipped.map(math.log) + (1 - actual) * (1 - clipped).map(math.log))).mean()),
                "avg_predicted_probability": float(probabilities.mean()),
                "actual_hit_rate": float(actual.mean()),
            }
            events = pd.DataFrame(
                {
                    "probability_bucket": probabilities.map(_probability_bucket),
                    "model_probability": probabilities,
                    "actual": actual,
                }
            )
        rows = [
            {
                "section": "overall",
                "market": market,
                "selection": "yes",
                "probability_bucket": "overall",
                "n": metrics["n"],
                "avg_predicted_probability": metrics["avg_predicted_probability"],
                "actual_hit_rate": metrics["actual_hit_rate"],
                "calibration_error": _difference(metrics["actual_hit_rate"], metrics["avg_predicted_probability"]),
                "brier_score": metrics["brier_score"],
                "log_loss": metrics["log_loss"],
                "top_pick_accuracy": math.nan,
                "average_top_pick_probability": math.nan,
                "notes": f"{prob_col} vs {actual_col}.",
            }
        ]
        rows.extend(_bucket_rows(events, section="binary_bucket", market=market, selection="yes"))
        summaries.append({"market": market, "metrics": metrics, "rows": pd.DataFrame(rows, columns=CALIBRATION_COLUMNS)})
    return summaries


def _binary_actual(value: Any) -> int | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return int(value)
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "yes", "y"}:
        return 1
    if text in {"0", "0.0", "false", "no", "n"}:
        return 0
    return None


def _difference(left: Any, right: Any) -> float:
    try:
        left_float = float(left)
        right_float = float(right)
    except (TypeError, ValueError):
        return math.nan
    if pd.isna(left_float) or pd.isna(right_float):
        return math.nan
    return float(left_float - right_float)


def _render_report(
    summary: dict[str, Any],
    calibration: pd.DataFrame,
    binary_summaries: list[dict[str, Any]],
    resolved_path: Path,
    calibration_path: Path,
    duplicate_audit_path: Path,
    report_date: str,
) -> str:
    lines = [
        f"# Prediction Ledger Calibration - {report_date}",
        "",
        "This report evaluates pre-kickoff prediction snapshots against the local results ledger. It is evaluation-only and does not provide staking, sizing, trading, wallet, or execution guidance.",
        "",
        "## Summary",
        "",
        f"- Total prediction rows: {summary['total_prediction_rows']:,}",
        f"- Result rows: {summary['result_rows']:,}",
        f"- Deduplicated result rows used for join: {summary['deduplicated_result_rows']:,}",
        f"- Joined rows: {summary['joined_rows']:,}",
        f"- Pre-kickoff joined snapshot rows: {summary['pre_kickoff_joined_rows']:,}",
        f"- Unresolved placeholder rows removed: {summary['unresolved_placeholder_rows_removed']:,}",
        f"- Concrete pre-kickoff snapshot rows: {summary['concrete_pre_kickoff_snapshot_rows']:,}",
        f"- Unique concrete matched matches: {summary['unique_concrete_matched_matches']:,}",
        f"- Duplicate snapshot rows removed: {summary['duplicate_snapshot_rows_removed']:,}",
        f"- Dedupe policy used: `{summary['dedupe_policy']}`",
        f"- Unresolved teams: `{summary['unresolved_team_policy']}`",
        f"- Evaluated 1X2 rows after dedupe: {summary['evaluated_rows']:,}",
        f"- 1X2 Brier score: {_fmt(summary['brier_score_1x2'])}",
        f"- Multiclass log loss: {_fmt(summary['multiclass_log_loss'])}",
        f"- Top-pick accuracy: {_fmt_pct(summary['top_pick_accuracy'])}",
        f"- Average top-pick probability: {_fmt_pct(summary['average_top_pick_probability'])}",
        f"- Top-pick hit rate: {_fmt_pct(summary['top_pick_hit_rate'])}",
        "",
        "## Favorite Calibration",
        "",
        _markdown_table(
            calibration.loc[calibration["section"].eq("top_pick_bucket")],
            ["probability_bucket", "n", "avg_predicted_probability", "actual_hit_rate", "calibration_error"],
        ),
        "",
        "## 1X2 Outcome Calibration",
        "",
        _markdown_table(
            calibration.loc[calibration["section"].eq("outcome_bucket")],
            ["probability_bucket", "n", "avg_predicted_probability", "actual_hit_rate", "calibration_error"],
        ),
        "",
        "## Binary Markets",
        "",
    ]
    if not binary_summaries:
        lines.append("_No configured binary market probability/result column pairs were available._")
    else:
        for item in binary_summaries:
            metrics = item["metrics"]
            lines.extend(
                [
                    f"### {item['market']}",
                    "",
                    f"- n: {metrics['n']:,}",
                    f"- Brier score: {_fmt(metrics['brier_score'])}",
                    f"- Log loss: {_fmt(metrics['log_loss'])}",
                    f"- Average predicted probability: {_fmt_pct(metrics['avg_predicted_probability'])}",
                    f"- Actual hit rate: {_fmt_pct(metrics['actual_hit_rate'])}",
                    "",
                    _markdown_table(
                        item["rows"].loc[item["rows"]["section"].eq("binary_bucket")],
                        ["probability_bucket", "n", "avg_predicted_probability", "actual_hit_rate", "calibration_error"],
                    ),
                    "",
                ]
            )
    lines.extend(
        [
            "## Outputs",
            "",
            f"- Resolved events: `{_display_path(resolved_path)}`",
            f"- Calibration table: `{_display_path(calibration_path)}`",
            f"- Duplicate audit: `{_display_path(duplicate_audit_path)}`",
            "",
            "## Notes",
            "",
            "- Rows are joined by exact `match_id` only.",
            "- Rows where `prediction_before_kickoff` is not true are excluded from evaluation.",
            f"- Unresolved placeholder team rows were `{summary['unresolved_team_policy']}` for this run.",
            "- Use `--include-unresolved-teams` to keep placeholder teams in calibration.",
            "- The default `latest_pre_kickoff` dedupe policy keeps one concrete pre-kickoff snapshot per `match_id`.",
            "- Result labels are normalized to `home`, `draw`, and `away` before scoring.",
            "- 1X2 Brier score is the average multiclass sum of squared errors across home/draw/away.",
            "- Log loss clips probabilities to `[0.001, 0.999]` to avoid infinite values.",
        ]
    )
    if summary["duplicate_result_match_id_rows"]:
        lines.append(f"- Duplicate result `match_id` rows detected: {summary['duplicate_result_match_id_rows']:,}; the latest row per `match_id` was used for joining.")
    return "\n".join(lines) + "\n"


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "_No rows._"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df.loc[:, columns].iterrows():
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                value = _fmt(value)
            values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _print_summary(
    summary: dict[str, Any],
    binary_summaries: list[dict[str, Any]],
    resolved_path: Path,
    calibration_path: Path,
    duplicate_audit_path: Path,
    report_path: Path,
) -> None:
    print("Prediction ledger calibration")
    print(f"prediction rows: {summary['total_prediction_rows']:,}")
    print(f"result rows: {summary['result_rows']:,}")
    print(f"joined rows: {summary['joined_rows']:,}")
    print(f"pre-kickoff joined snapshot rows: {summary['pre_kickoff_joined_rows']:,}")
    print(f"unresolved placeholder rows removed: {summary['unresolved_placeholder_rows_removed']:,}")
    print(f"concrete pre-kickoff snapshot rows: {summary['concrete_pre_kickoff_snapshot_rows']:,}")
    print(f"unique concrete matched matches: {summary['unique_concrete_matched_matches']:,}")
    print(f"duplicate snapshot rows removed: {summary['duplicate_snapshot_rows_removed']:,}")
    print(f"dedupe policy used: {summary['dedupe_policy']}")
    print(f"unresolved teams: {summary['unresolved_team_policy']}")
    print(f"evaluated 1X2 rows after dedupe: {summary['evaluated_rows']:,}")
    print(f"1X2 Brier score: {_fmt(summary['brier_score_1x2'])}")
    print(f"multiclass log loss: {_fmt(summary['multiclass_log_loss'])}")
    print(f"top-pick accuracy: {_fmt_pct(summary['top_pick_accuracy'])}")
    print(f"average top-pick probability: {_fmt_pct(summary['average_top_pick_probability'])}")
    print(f"top-pick hit rate: {_fmt_pct(summary['top_pick_hit_rate'])}")
    for item in binary_summaries:
        metrics = item["metrics"]
        print(
            f"{item['market']}: n={metrics['n']:,}, "
            f"brier={_fmt(metrics['brier_score'])}, "
            f"log_loss={_fmt(metrics['log_loss'])}, "
            f"hit_rate={_fmt_pct(metrics['actual_hit_rate'])}"
        )
    print(f"resolved events: {_display_path(resolved_path)}")
    print(f"calibration table: {_display_path(calibration_path)}")
    print(f"duplicate audit: {_display_path(duplicate_audit_path)}")
    print(f"report: {_display_path(report_path)}")


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if pd.isna(number):
        return ""
    return f"{number:.4f}"


def _fmt_pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if pd.isna(number):
        return ""
    return f"{number:.1%}"


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
