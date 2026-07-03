from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR  # noqa: E402
from src.pdf_prediction_import import (  # noqa: E402
    audit_rows_to_prediction_ledger,
    build_pdf_backfill_inventory,
    create_timestamp_override_template,
    load_timestamp_overrides,
    render_pdf_backfill_inventory,
)


def main() -> None:
    args = _parse_args()
    pdf_dir = Path(args.pdf_dir)
    output = Path(args.output)
    audit_output = Path(args.audit_output)
    inventory_output = Path(args.inventory_output)
    inventory_report = Path(args.inventory_report)
    override_template = Path(args.override_template)
    fixtures = _read_csv(Path(args.fixtures))
    existing = _read_csv(Path(args.prediction_ledger))
    overrides = load_timestamp_overrides(args.timestamp_overrides)
    create_timestamp_override_template(override_template)

    inventory = build_pdf_backfill_inventory(
        pdf_dir,
        fixtures_df=fixtures,
        timestamp_overrides=overrides,
        existing_predictions_df=existing,
    )
    backfill = audit_rows_to_prediction_ledger(inventory)

    output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    inventory_output.parent.mkdir(parents=True, exist_ok=True)
    inventory_report.parent.mkdir(parents=True, exist_ok=True)

    backfill.to_csv(output, index=False)
    inventory.to_csv(audit_output, index=False)
    inventory.to_csv(inventory_output, index=False)
    inventory_report.write_text(render_pdf_backfill_inventory(inventory), encoding="utf-8")

    found = len(inventory)
    imported = len(backfill)
    rejected = found - imported
    print(f"pdfs_found: {found:,}")
    print(f"pdfs_imported: {imported:,}")
    print(f"pdfs_rejected_or_review_required: {rejected:,}")
    if not inventory.empty:
        counts = inventory["import_candidate_status"].value_counts(dropna=False)
        for status, count in counts.items():
            print(f"status:{status}: {count:,}")
    print(f"backfill_output: {_display_path(output)}")
    print(f"audit_output: {_display_path(audit_output)}")
    print(f"inventory_output: {_display_path(inventory_output)}")
    print(f"inventory_report: {_display_path(inventory_report)}")
    print(f"override_template: {_display_path(override_template)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import timestamp-validated model prediction PDFs as backfilled prediction snapshots.")
    parser.add_argument("--pdf-dir", default=str(DATA_DIR / "backfill" / "model_pdfs"))
    parser.add_argument("--output", default=str(DATA_DIR / "backfilled_prediction_ledger.csv"))
    parser.add_argument("--audit-output", default=str(PROJECT_ROOT / "reports" / "pdf_prediction_import_audit.csv"))
    parser.add_argument("--inventory-output", default=str(PROJECT_ROOT / "reports" / "pdf_backfill_inventory.csv"))
    parser.add_argument("--inventory-report", default=str(PROJECT_ROOT / "reports" / "pdf_backfill_inventory.md"))
    parser.add_argument("--fixtures", default=str(DATA_DIR / "fixtures.csv"))
    parser.add_argument("--prediction-ledger", default=str(DATA_DIR / "prediction_ledger.csv"))
    parser.add_argument("--timestamp-overrides", default="")
    parser.add_argument("--override-template", default=str(DATA_DIR / "backfill" / "pdf_timestamp_overrides_template.csv"))
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
