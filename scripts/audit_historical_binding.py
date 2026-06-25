from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.historical_binding import audit_historical_binding_for_match, audit_historical_file_status  # noqa: E402
from src.historical_data import load_historical_matches  # noqa: E402
from src.rating_coverage import load_team_ratings_csv  # noqa: E402
from src.team_behavior import load_team_behavior  # noqa: E402
from src.team_names import load_team_name_aliases_df  # noqa: E402
from src.utils import today_iso  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit historical data binding for a selected match.")
    parser.add_argument("--home", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--as-of-date", default=None)
    args = parser.parse_args()

    aliases = load_team_name_aliases_df()
    historical = load_historical_matches()
    behavior = load_team_behavior()
    ratings = load_team_ratings_csv()
    binding = audit_historical_binding_for_match(
        args.home,
        args.away,
        historical,
        team_behavior_df=behavior,
        team_ratings_df=ratings,
        aliases_df=aliases,
        as_of_date=args.as_of_date,
    )
    file_status = audit_historical_file_status(historical)

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    today = today_iso()
    pair = f"{_safe(args.home)}_{_safe(args.away)}"
    csv_path = reports_dir / f"historical_binding_{pair}_{today}.csv"
    md_path = reports_dir / f"historical_binding_{pair}_{today}.md"
    pd.DataFrame([binding]).to_csv(csv_path, index=False)
    _write_markdown(md_path, binding, file_status)

    print("Historical binding audit")
    for key, value in binding.items():
        print(f"{key}: {value}")
    print(f"historical_rows_loaded: {file_status['rows_loaded']}")
    print(f"csv_report: {csv_path}")
    print(f"markdown_report: {md_path}")


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip()).strip("_") or "team"


def _write_markdown(path: Path, binding: dict, file_status: dict) -> None:
    lines = ["# Historical Binding Audit", "", "## Match Binding", ""]
    for key, value in binding.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Historical File Status", ""])
    for key, value in file_status.items():
        lines.append(f"- {key}: {value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
