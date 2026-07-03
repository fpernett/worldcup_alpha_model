from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR  # noqa: E402
from src.evaluation import (  # noqa: E402
    build_formal_evaluation_dataset,
    build_prediction_source_dataset,
    build_result_prediction_join_audit,
    calibration_summary,
    evaluation_coverage_summary,
    render_evaluation_coverage_audit,
    render_result_prediction_join_audit,
    write_milestone2_reports,
)
from src.odds import load_market_odds  # noqa: E402
from src.pdf_prediction_import import (  # noqa: E402
    calibration_status_from_sample_size,
    render_prediction_source_coverage,
)


def main() -> None:
    args = _parse_args()
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    latest_only = str(args.latest_prekickoff_only).strip().lower() in {"1", "true", "yes"}

    predictions = _read_csv(Path(args.predictions))
    results = _read_csv(Path(args.results_ledger))
    fixtures = _read_csv(Path(args.fixtures))
    crosswalk = _read_csv(Path(args.result_fixture_crosswalk))
    pdf_audit = _read_csv(Path(args.pdf_audit))
    market_odds = load_market_odds()

    evaluation = build_formal_evaluation_dataset(
        predictions,
        results,
        fixtures_df=fixtures,
        result_fixture_crosswalk_df=crosswalk,
        market_odds_df=market_odds,
        latest_snapshot_only=latest_only,
    )
    all_snapshot_evaluation = build_formal_evaluation_dataset(
        predictions,
        results,
        fixtures_df=fixtures,
        result_fixture_crosswalk_df=crosswalk,
        market_odds_df=market_odds,
        latest_snapshot_only=False,
    )
    prediction_sources = build_prediction_source_dataset(predictions, market_odds_df=market_odds)
    join_audit = build_result_prediction_join_audit(
        predictions,
        results,
        fixtures_df=fixtures,
        result_fixture_crosswalk_df=crosswalk,
        market_odds_df=market_odds,
    )

    paths = write_milestone2_reports(evaluation, reports_dir=reports_dir, prediction_sources_df=prediction_sources)
    all_eval_path = reports_dir / "evaluation_dataset_all_snapshots.csv"
    all_summary_path = reports_dir / "calibration_summary_all_snapshots.csv"
    all_snapshot_evaluation.to_csv(all_eval_path, index=False)
    calibration_summary(all_snapshot_evaluation).to_csv(all_summary_path, index=False)

    coverage_path = reports_dir / "evaluation_coverage_audit.csv"
    coverage_report_path = reports_dir / "evaluation_coverage_audit.md"
    join_audit_path = reports_dir / "result_prediction_join_audit.csv"
    join_audit_report_path = reports_dir / "result_prediction_join_audit.md"
    coverage = evaluation_coverage_summary(join_audit, previous_usable_rows=int(args.previous_usable_rows))
    coverage.to_csv(coverage_path, index=False)
    join_audit.to_csv(join_audit_path, index=False)

    comparison = _read_csv(paths["calibration_model_comparison"])
    calibration_status = calibration_status_from_sample_size(len(evaluation), comparison)
    source_coverage = render_prediction_source_coverage(
        predictions_df=predictions,
        pdf_audit_df=pdf_audit,
        evaluation_df=evaluation,
        calibration_status=calibration_status,
    )
    coverage_report_path.write_text(
        render_evaluation_coverage_audit(join_audit, int(args.previous_usable_rows)) + "\n\n" + source_coverage + "\n",
        encoding="utf-8",
    )
    join_audit_report_path.write_text(
        render_result_prediction_join_audit(
            join_audit,
            prediction_rows=len(join_audit),
            unique_prediction_match_ids=_nunique(join_audit, "match_id"),
            result_rows=len(results),
            unique_result_match_ids=_nunique(results, "match_id"),
            completed_result_rows=0,
            previous_usable_rows=int(args.previous_usable_rows),
        ),
        encoding="utf-8",
    )

    print(f"prediction_rows: {len(predictions):,}")
    print(f"pdf_rows: {(predictions.get('source_type', pd.Series(dtype=str)).astype(str) == 'pdf_report').sum() if not predictions.empty and 'source_type' in predictions.columns else 0:,}")
    print(f"evaluated_rows_latest: {len(evaluation):,}")
    print(f"evaluated_rows_all_snapshots: {len(all_snapshot_evaluation):,}")
    print(f"calibration_status: {calibration_status}")
    print(f"evaluation_dataset: {_display_path(paths['evaluation_dataset'])}")
    print(f"all_snapshot_evaluation: {_display_path(all_eval_path)}")
    print(f"coverage_report: {_display_path(coverage_report_path)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run evaluation on an enriched prediction ledger with PDF backfill rows.")
    parser.add_argument("--predictions", default=str(DATA_DIR / "prediction_ledger_enriched.csv"))
    parser.add_argument("--results-ledger", default=str(DATA_DIR / "results_ledger.csv"))
    parser.add_argument("--fixtures", default=str(DATA_DIR / "fixtures.csv"))
    parser.add_argument("--result-fixture-crosswalk", default=str(DATA_DIR / "result_fixture_crosswalk.csv"))
    parser.add_argument("--pdf-audit", default=str(PROJECT_ROOT / "reports" / "pdf_prediction_import_audit.csv"))
    parser.add_argument("--reports-dir", default=str(PROJECT_ROOT / "reports"))
    parser.add_argument("--latest-prekickoff-only", default="true")
    parser.add_argument("--previous-usable-rows", type=int, default=1)
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _nunique(df: pd.DataFrame, col: str) -> int:
    return int(df[col].astype(str).nunique()) if col in df.columns and not df.empty else 0


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
