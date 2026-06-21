from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR  # noqa: E402
from src.elo_calibration import validate_rolling_elo  # noqa: E402
from src.historical_data import load_historical_matches  # noqa: E402
from src.ratings import TEAM_RATING_COLUMNS  # noqa: E402
from src.utils import read_csv_with_columns, today_iso  # noqa: E402


def main() -> None:
    historical = load_historical_matches(use_cache=False)
    manual = read_csv_with_columns(DATA_DIR / "team_ratings.csv", TEAM_RATING_COLUMNS)
    validation = validate_rolling_elo(historical, manual)

    print("Top 30 rolling Elo teams")
    print(_records_to_frame(validation["top_elo_teams"]).to_string(index=False))
    print("\nBottom 30 rolling Elo teams")
    print(_records_to_frame(validation["bottom_elo_teams"]).to_string(index=False))
    print("\nWorld Cup model teams with rolling Elo")
    model_teams = _records_to_frame(validation["model_team_elos"])
    print(model_teams.to_string(index=False))
    print("\nManual-vs-Elo correlation:", validation["manual_vs_elo_correlation"])
    print("\nLarge manual/Elo disagreements")
    print(_records_to_frame(validation["teams_with_large_manual_elo_disagreement"]).to_string(index=False))
    if validation["warnings"]:
        print("\nWarnings")
        for warning in validation["warnings"]:
            print(f"- {warning}")

    report_path = write_elo_calibration_report(validation)
    print(f"\nreport: {report_path.relative_to(PROJECT_ROOT)}")


def write_elo_calibration_report(validation: dict, path: Path | None = None) -> Path:
    report_date = today_iso()
    report_path = path or PROJECT_ROOT / "reports" / f"elo_calibration_audit_{report_date}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    body = [
        f"# Elo Calibration Audit - {report_date}",
        "",
        "## Distribution",
        "",
        _dict_to_markdown(validation.get("elo_distribution_summary", {})),
        "",
        f"Manual-vs-Elo correlation: `{validation.get('manual_vs_elo_correlation')}`",
        "",
        "## Top 30 Rolling Elo Teams",
        "",
        _to_markdown(_records_to_frame(validation["top_elo_teams"])),
        "",
        "## Bottom 30 Rolling Elo Teams",
        "",
        _to_markdown(_records_to_frame(validation["bottom_elo_teams"])),
        "",
        "## Model Teams",
        "",
        _to_markdown(_records_to_frame(validation["model_team_elos"])),
        "",
        "## Large Disagreements",
        "",
        _to_markdown(_records_to_frame(validation["teams_with_large_manual_elo_disagreement"])),
        "",
        "## Warnings",
        "",
        "\n".join(f"- {warning}" for warning in validation.get("warnings", [])) or "_None._",
    ]
    report_path.write_text("\n".join(body), encoding="utf-8")
    return report_path


def _records_to_frame(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(records)


def _dict_to_markdown(values: dict) -> str:
    if not values:
        return "_No values._"
    return "\n".join(f"- {key}: {value}" for key, value in values.items())


def _to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    headers = list(df.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row.get(col, "")).replace("|", "\\|") for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
