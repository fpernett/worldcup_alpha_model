from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ratings import get_team_ratings  # noqa: E402
from src.team_behavior import load_team_behavior  # noqa: E402
from src.utils import today_iso  # noqa: E402


AUDIT_COLUMNS = [
    "team",
    "matches_used_recent",
    "mean_opponent_elo_recent",
    "attack_index_raw",
    "attack_index_adjusted_old",
    "attack_index_residual",
    "attack_index_final",
    "weighted_goal_for_residual",
    "defense_index_raw",
    "defense_index_adjusted_old",
    "defense_index_residual",
    "defense_index_final",
    "weighted_goal_against_residual",
    "recent_form_index",
    "warnings",
]


def build_residual_audit_table(teams: list[str] | None = None) -> pd.DataFrame:
    behavior = load_team_behavior()
    if behavior.empty:
        return pd.DataFrame(columns=AUDIT_COLUMNS)

    if teams is None:
        ratings = get_team_ratings(force_refresh=True)
        teams = ratings["team"].dropna().astype(str).tolist() if not ratings.empty and "team" in ratings.columns else []
    wanted = {team.lower() for team in teams}
    selected = behavior.loc[behavior["team"].astype(str).str.lower().isin(wanted)].copy() if wanted else behavior.copy()
    rows = []
    for _, row in selected.sort_values("team").iterrows():
        rows.append(
            {
                "team": str(row.get("team", "")),
                "matches_used_recent": _int_or_blank(row.get("matches_used_recent", row.get("n_matches"))),
                "mean_opponent_elo_recent": _round_or_blank(row.get("mean_opponent_elo_recent")),
                "attack_index_raw": _round_or_blank(row.get("attack_index_raw")),
                "attack_index_adjusted_old": _round_or_blank(row.get("attack_index_adjusted_old", row.get("attack_index_adjusted"))),
                "attack_index_residual": _round_or_blank(row.get("attack_index_residual")),
                "attack_index_final": _round_or_blank(row.get("attack_index_final", row.get("attack_index"))),
                "weighted_goal_for_residual": _signed_round(row.get("weighted_goal_for_residual")),
                "defense_index_raw": _round_or_blank(row.get("defense_index_raw")),
                "defense_index_adjusted_old": _round_or_blank(row.get("defense_index_adjusted_old", row.get("defense_index_adjusted"))),
                "defense_index_residual": _round_or_blank(row.get("defense_index_residual")),
                "defense_index_final": _round_or_blank(row.get("defense_index_final", row.get("defense_index"))),
                "weighted_goal_against_residual": _signed_round(row.get("weighted_goal_against_residual")),
                "recent_form_index": _round_or_blank(row.get("recent_form_index")),
                "warnings": _warnings(row),
            }
        )
    return pd.DataFrame(rows, columns=AUDIT_COLUMNS)


def write_residual_audit_report(audit: pd.DataFrame, path: Path | None = None) -> Path:
    report_date = today_iso()
    report_path = path or PROJECT_ROOT / "reports" / f"performance_residuals_audit_{report_date}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    body = [
        f"# Performance Residuals Audit - {report_date}",
        "",
        "This audit checks whether teams performed better or worse than expected given opponent Elo.",
        "",
        _to_markdown(audit),
        "",
        "Residual behavior is descriptive, not causal proof. Manual ratings remain the stable base.",
    ]
    report_path.write_text("\n".join(body), encoding="utf-8")
    return report_path


def main() -> None:
    audit = build_residual_audit_table()
    print(audit.to_string(index=False))
    report_path = write_residual_audit_report(audit)
    print(f"\nreport: {report_path.relative_to(PROJECT_ROOT)}")


def _warnings(row: pd.Series) -> str:
    warning_parts = [
        str(row.get("residual_warning", "") or ""),
        str(row.get("schedule_strength_warning", "") or ""),
        str(row.get("opponent_quality_warning", "") or ""),
    ]
    return "; ".join([part for part in warning_parts if part.strip()])


def _num(value: object, default: float = float("nan")) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _round_or_blank(value: object) -> str:
    numeric = _num(value)
    if pd.isna(numeric):
        return ""
    return f"{numeric:.3f}"


def _signed_round(value: object) -> str:
    numeric = _num(value)
    if pd.isna(numeric):
        return ""
    return f"{numeric:+.3f}"


def _int_or_blank(value: object) -> str:
    numeric = _num(value)
    if pd.isna(numeric):
        return ""
    return str(int(round(numeric)))


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
