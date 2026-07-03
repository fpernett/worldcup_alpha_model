from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR  # noqa: E402
from src.pdf_prediction_import import build_enriched_prediction_ledger  # noqa: E402


def main() -> None:
    args = _parse_args()
    prediction_ledger = _read_csv(Path(args.prediction_ledger))
    backfill = _read_csv(Path(args.backfill))
    enriched, audit = build_enriched_prediction_ledger(prediction_ledger, backfill)

    output = Path(args.output)
    audit_output = Path(args.audit_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output, index=False)
    audit.to_csv(audit_output, index=False)

    source_counts = enriched.get("source_type", pd.Series(dtype=str)).fillna("unknown").value_counts(dropna=False)
    print(f"existing_prediction_rows: {len(prediction_ledger):,}")
    print(f"backfill_rows: {len(backfill):,}")
    print(f"enriched_rows: {len(enriched):,}")
    for source_type, count in source_counts.items():
        print(f"source:{source_type}: {count:,}")
    print(f"output: {_display_path(output)}")
    print(f"audit_output: {_display_path(audit_output)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge app-generated and PDF-backfilled prediction snapshots into an enriched ledger.")
    parser.add_argument("--prediction-ledger", default=str(DATA_DIR / "prediction_ledger.csv"))
    parser.add_argument("--backfill", default=str(DATA_DIR / "backfilled_prediction_ledger.csv"))
    parser.add_argument("--output", default=str(DATA_DIR / "prediction_ledger_enriched.csv"))
    parser.add_argument("--audit-output", default=str(PROJECT_ROOT / "reports" / "backfilled_prediction_merge_audit.csv"))
    return parser.parse_args()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
