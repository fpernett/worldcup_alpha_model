from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.behavior_driver_report import audit_final_model_inputs  # noqa: E402
from src.config import DATA_DIR  # noqa: E402
from src.ratings import TEAM_RATING_COLUMNS, get_team_ratings, _normalise_ratings  # noqa: E402
from src.team_behavior import load_team_behavior  # noqa: E402
from src.utils import read_csv_with_columns, today_iso  # noqa: E402


def main() -> None:
    raw = _normalise_ratings(read_csv_with_columns(DATA_DIR / "team_ratings.csv", TEAM_RATING_COLUMNS), "manual_csv")
    adjusted = get_team_ratings()
    behavior = load_team_behavior()
    audit = audit_final_model_inputs(raw, adjusted, behavior)

    display_columns = [
        "team",
        "manual_attack",
        "behavior_attack_final",
        "adjusted_attack_used_by_model",
        "attack_delta",
        "manual_defense",
        "behavior_defense_final",
        "adjusted_defense_used_by_model",
        "defense_delta",
        "manual_recent_form",
        "adjusted_recent_form_used_by_model",
        "recent_form_delta",
        "warning",
    ]
    display = audit[[col for col in display_columns if col in audit.columns]].copy()
    print(display.to_string(index=False))

    report_path = PROJECT_ROOT / "reports" / f"model_input_audit_{today_iso()}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n\n".join(
            [
                f"# Model Input Audit - {today_iso()}",
                "",
                "This audit shows the actual attack, defense, and recent-form values passed to the model after behavior blending and delta caps.",
                "",
                _to_markdown(display),
                "",
                "Historical behavior is descriptive. Manual ratings remain the stable base.",
            ]
        ),
        encoding="utf-8",
    )
    print(f"\nreport: {report_path.relative_to(PROJECT_ROOT)}")


def _to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        values = [str(row.get(col, "")).replace("|", "\\|") for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
