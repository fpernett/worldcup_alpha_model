from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.match_identity import (  # noqa: E402
    build_result_fixture_crosswalk,
    render_schedule_result_bridge_audit,
    result_fixture_crosswalk_summary,
)
from src.storage import data_path  # noqa: E402


def main() -> None:
    args = _parse_args()
    results = _read_csv(Path(args.results))
    fixtures = _read_csv(Path(args.fixtures))
    predictions = _read_csv(Path(args.predictions))

    crosswalk = build_result_fixture_crosswalk(results, fixtures)
    output = Path(args.output)
    audit_output = Path(args.audit_output)
    audit_report = Path(args.audit_report)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_report.parent.mkdir(parents=True, exist_ok=True)

    crosswalk.to_csv(output, index=False)
    crosswalk.to_csv(audit_output, index=False)
    audit_report.write_text(
        render_schedule_result_bridge_audit(
            crosswalk,
            result_rows=len(results),
            fixture_rows=len(fixtures),
            prediction_rows=len(predictions),
        ),
        encoding="utf-8",
    )

    summary = result_fixture_crosswalk_summary(
        crosswalk,
        result_rows=len(results),
        fixture_rows=len(fixtures),
        prediction_rows=len(predictions),
    )
    for _, row in summary.iterrows():
        print(f"{row['metric']}: {int(row['value']):,}")
    print(f"crosswalk: {output.relative_to(PROJECT_ROOT)}")
    print(f"audit_csv: {audit_output.relative_to(PROJECT_ROOT)}")
    print(f"audit_report: {audit_report.relative_to(PROJECT_ROOT)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic result-ledger to fixture schedule bridge.")
    parser.add_argument("--results", default=str(data_path("results_ledger.csv")))
    parser.add_argument("--fixtures", default=str(data_path("fixtures.csv")))
    parser.add_argument("--predictions", default=str(data_path("prediction_ledger.csv")))
    parser.add_argument("--output", default=str(data_path("result_fixture_crosswalk.csv")))
    parser.add_argument("--audit-output", default=str(PROJECT_ROOT / "reports" / "schedule_result_bridge_audit.csv"))
    parser.add_argument("--audit-report", default=str(PROJECT_ROOT / "reports" / "schedule_result_bridge_audit.md"))
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


if __name__ == "__main__":
    main()
