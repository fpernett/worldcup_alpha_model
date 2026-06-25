from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DATA_DIR  # noqa: E402
from src.data_sources import FIXTURE_COLUMNS, get_upcoming_fixtures  # noqa: E402
from src.fixture_diagnostics import audit_fixture_availability  # noqa: E402
from src.utils import read_csv_with_columns, today_iso  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit fixture availability for a UTC window.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--hide-past", action="store_true")
    parser.add_argument("--now-utc", default=None)
    args = parser.parse_args()

    if args.start_date and args.end_date:
        fixtures = get_upcoming_fixtures(date.fromisoformat(args.start_date), date.fromisoformat(args.end_date))
    else:
        fixtures = read_csv_with_columns(DATA_DIR / "fixtures.csv", FIXTURE_COLUMNS)
    audit, summary = audit_fixture_availability(
        fixtures,
        start_date=args.start_date,
        end_date=args.end_date,
        hide_past=args.hide_past,
        now_utc=args.now_utc,
    )

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    today = today_iso()
    csv_path = reports_dir / f"fixture_availability_{today}.csv"
    md_path = reports_dir / f"fixture_availability_{today}.md"
    audit.to_csv(csv_path, index=False)
    _write_markdown(md_path, audit, summary)

    print("Fixture availability audit")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"csv_report: {csv_path}")
    print(f"markdown_report: {md_path}")


def _write_markdown(path: Path, audit: pd.DataFrame, summary: dict) -> None:
    lines = ["# Fixture Availability Audit", ""]
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    excluded = audit.loc[~audit.get("visible_in_app", pd.Series(dtype=bool))].copy()
    if not excluded.empty:
        lines.extend(["", "## Excluded Fixtures", ""])
        cols = ["match_id", "date_utc", "time_utc", "home", "away", "excluded_reason"]
        lines.append(_simple_markdown_table(excluded[cols]))
    visible = audit.loc[audit.get("visible_in_app", pd.Series(dtype=bool))].copy()
    if not visible.empty:
        lines.extend(["", "## Visible Fixtures", ""])
        cols = ["match_id", "date_utc", "time_utc", "home", "away", "venue"]
        lines.append(_simple_markdown_table(visible[cols]))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _simple_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_None_"
    out = df.fillna("").astype(str)
    header = "| " + " | ".join(out.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(out.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in out.to_numpy()]
    return "\n".join([header, sep, *rows])


if __name__ == "__main__":
    main()
