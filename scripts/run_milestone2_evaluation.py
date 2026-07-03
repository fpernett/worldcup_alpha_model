from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import (  # noqa: E402
    build_formal_evaluation_dataset,
    build_prediction_source_dataset,
    build_result_prediction_join_audit,
    evaluation_coverage_summary,
    render_evaluation_coverage_audit,
    render_result_prediction_join_audit,
    write_milestone2_reports,
)
from src.match_identity import (  # noqa: E402
    build_result_fixture_crosswalk,
    render_schedule_result_bridge_audit,
    result_fixture_crosswalk_summary,
    validate_results_ledger_semantics,
)
from src.odds import load_market_odds  # noqa: E402
from src.prediction_ledger import PREDICTION_LEDGER_COLUMNS, RESULTS_LEDGER_COLUMNS  # noqa: E402
from src.storage import data_path  # noqa: E402


def main() -> None:
    args = _parse_args()
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    previous_usable = int(args.previous_usable_rows) if args.previous_usable_rows is not None else _previous_evaluation_rows(reports_dir / "evaluation_dataset.csv")

    prediction_ledger = _read_csv_with_columns(Path(args.predictions), PREDICTION_LEDGER_COLUMNS)
    raw_results_ledger = _read_csv_with_columns(Path(args.results_ledger), RESULTS_LEDGER_COLUMNS)
    results_ledger = validate_results_ledger_semantics(raw_results_ledger)
    prediction_log = _read_csv_path(Path(args.prediction_log))
    completed_results = _read_csv_path(Path(args.completed_results))
    fixtures = _read_csv_path(Path(args.fixtures))
    result_fixture_crosswalk = build_result_fixture_crosswalk(results_ledger, fixtures)
    crosswalk_path = Path(args.result_fixture_crosswalk)
    crosswalk_path.parent.mkdir(parents=True, exist_ok=True)
    result_fixture_crosswalk.to_csv(crosswalk_path, index=False)
    schedule_audit_path = reports_dir / "schedule_result_bridge_audit.csv"
    schedule_audit_report_path = reports_dir / "schedule_result_bridge_audit.md"
    result_fixture_crosswalk.to_csv(schedule_audit_path, index=False)
    schedule_audit_report_path.write_text(
        render_schedule_result_bridge_audit(
            result_fixture_crosswalk,
            result_rows=len(results_ledger),
            fixture_rows=len(fixtures),
            prediction_rows=len(prediction_ledger) + len(prediction_log),
        ),
        encoding="utf-8",
    )
    market_odds = load_market_odds()
    join_audit = build_result_prediction_join_audit(
        prediction_ledger,
        results_ledger,
        completed_results_df=completed_results,
        fixtures_df=fixtures,
        result_fixture_crosswalk_df=result_fixture_crosswalk,
        prediction_log_df=prediction_log,
        market_odds_df=market_odds,
    )
    evaluation = build_formal_evaluation_dataset(
        prediction_ledger,
        results_ledger,
        completed_results_df=completed_results,
        fixtures_df=fixtures,
        result_fixture_crosswalk_df=result_fixture_crosswalk,
        prediction_log_df=prediction_log,
        market_odds_df=market_odds,
    )
    prediction_sources = build_prediction_source_dataset(
        prediction_ledger,
        prediction_log,
        market_odds,
    )
    paths = write_milestone2_reports(
        evaluation,
        reports_dir=reports_dir,
        prediction_sources_df=prediction_sources,
    )
    join_audit_path = reports_dir / "result_prediction_join_audit.csv"
    join_audit_report_path = reports_dir / "result_prediction_join_audit.md"
    coverage_path = reports_dir / "evaluation_coverage_audit.csv"
    coverage_report_path = reports_dir / "evaluation_coverage_audit.md"
    join_audit.to_csv(join_audit_path, index=False)
    coverage = evaluation_coverage_summary(join_audit, previous_usable_rows=previous_usable)
    coverage.to_csv(coverage_path, index=False)
    join_audit_report_path.write_text(
        render_result_prediction_join_audit(
            join_audit,
            prediction_rows=len(join_audit),
            unique_prediction_match_ids=_nunique(join_audit, "match_id"),
            result_rows=len(results_ledger),
            unique_result_match_ids=_nunique(results_ledger, "match_id"),
            completed_result_rows=len(completed_results),
            previous_usable_rows=previous_usable,
        ),
        encoding="utf-8",
    )
    coverage_report_path.write_text(render_evaluation_coverage_audit(join_audit, previous_usable), encoding="utf-8")

    print(f"prediction ledger rows: {len(prediction_ledger):,}")
    print(f"prediction log rows: {len(prediction_log):,}")
    print(f"results ledger rows: {len(results_ledger):,}")
    print(f"completed-result rows: {len(completed_results):,}")
    crosswalk_summary = result_fixture_crosswalk_summary(
        result_fixture_crosswalk,
        result_rows=len(results_ledger),
        fixture_rows=len(fixtures),
        prediction_rows=len(prediction_ledger) + len(prediction_log),
    )
    for _, row in crosswalk_summary.iterrows():
        print(f"{row['metric']}: {int(row['value']):,}")
    print(f"exact joins possible: {(join_audit['result_join_method'] == 'exact_match_id').sum() if not join_audit.empty else 0:,}")
    print(f"schedule bridge joins possible: {(join_audit['result_join_method'] == 'result_fixture_crosswalk').sum() if not join_audit.empty else 0:,}")
    print(f"bridge joins possible: {(join_audit['result_join_method'] == 'fixture_bridge').sum() if not join_audit.empty else 0:,}")
    print(f"team/date joins possible: {join_audit['result_join_method'].isin(['normalized_team_date', 'fuzzy_team_date', 'symmetric_team_date']).sum() if not join_audit.empty else 0:,}")
    print(f"previous evaluation rows: {previous_usable:,}")
    print(f"evaluation rows: {len(evaluation):,}")
    for key, path in paths.items():
        print(f"{key}: {path.relative_to(PROJECT_ROOT)}")
    print(f"result_prediction_join_audit: {join_audit_path.relative_to(PROJECT_ROOT)}")
    print(f"evaluation_coverage_audit: {coverage_path.relative_to(PROJECT_ROOT)}")
    print(f"result_fixture_crosswalk: {crosswalk_path.relative_to(PROJECT_ROOT)}")
    print(f"schedule_result_bridge_audit: {schedule_audit_path.relative_to(PROJECT_ROOT)}")
    print(f"result_prediction_join_audit_report: {join_audit_report_path.relative_to(PROJECT_ROOT)}")
    print(f"evaluation_coverage_audit_report: {coverage_report_path.relative_to(PROJECT_ROOT)}")
    print(f"schedule_result_bridge_audit_report: {schedule_audit_report_path.relative_to(PROJECT_ROOT)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate predictive evaluation and result/prediction join audit reports.")
    parser.add_argument("--predictions", default=str(data_path("prediction_ledger.csv")))
    parser.add_argument("--results-ledger", default=str(data_path("results_ledger.csv")))
    parser.add_argument("--completed-results", default=str(data_path("completed_results.csv")))
    parser.add_argument("--fixtures", default=str(data_path("fixtures.csv")))
    parser.add_argument("--result-fixture-crosswalk", default=str(data_path("result_fixture_crosswalk.csv")))
    parser.add_argument("--prediction-log", default=str(data_path("prediction_log.csv")))
    parser.add_argument("--reports-dir", default=str(PROJECT_ROOT / "reports"))
    parser.add_argument("--previous-usable-rows", type=int, default=None)
    return parser.parse_args()


def _read_csv_path(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_csv_with_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path)
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df[columns].copy()


def _previous_evaluation_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return len(pd.read_csv(path))
    except Exception:
        return 0


def _nunique(df: pd.DataFrame, col: str) -> int:
    return int(df[col].astype(str).nunique()) if col in df.columns and not df.empty else 0


if __name__ == "__main__":
    main()
