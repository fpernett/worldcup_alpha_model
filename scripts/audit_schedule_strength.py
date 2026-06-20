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
    "strong_opponent_match_count",
    "weak_opponent_match_count",
    "schedule_strength_label",
    "attack_index_raw",
    "attack_index_adjusted",
    "attack_adjustment_delta",
    "defense_index_raw",
    "defense_index_adjusted",
    "defense_adjustment_delta",
    "recent_form_index",
    "warnings",
]


def build_schedule_audit_table(teams: list[str] | None = None) -> pd.DataFrame:
    behavior = load_team_behavior()
    if behavior.empty:
        return pd.DataFrame(columns=AUDIT_COLUMNS)

    if teams is None:
        model_teams = get_team_ratings(force_refresh=True)
        teams = model_teams["team"].dropna().astype(str).tolist() if not model_teams.empty and "team" in model_teams.columns else []
    wanted = {team.lower() for team in teams}
    selected = behavior.loc[behavior["team"].astype(str).str.lower().isin(wanted)].copy() if wanted else behavior.copy()
    rows = []
    for _, row in selected.sort_values("team").iterrows():
        attack_raw = _num(row.get("attack_index_raw", row.get("attack_index")))
        attack_adjusted = _num(row.get("attack_index_adjusted", row.get("attack_index")))
        defense_raw = _num(row.get("defense_index_raw", row.get("defense_index")))
        defense_adjusted = _num(row.get("defense_index_adjusted", row.get("defense_index")))
        rows.append(
            {
                "team": str(row.get("team", "")),
                "matches_used_recent": _int_or_blank(row.get("matches_used_recent", row.get("n_matches"))),
                "mean_opponent_elo_recent": _round_or_blank(row.get("mean_opponent_elo_recent")),
                "strong_opponent_match_count": _int_or_blank(row.get("strong_opponent_match_count")),
                "weak_opponent_match_count": _int_or_blank(row.get("weak_opponent_match_count")),
                "schedule_strength_label": str(row.get("schedule_strength_label", "") or ""),
                "attack_index_raw": _round_or_blank(attack_raw),
                "attack_index_adjusted": _round_or_blank(attack_adjusted),
                "attack_adjustment_delta": _signed_round(attack_adjusted - attack_raw),
                "defense_index_raw": _round_or_blank(defense_raw),
                "defense_index_adjusted": _round_or_blank(defense_adjusted),
                "defense_adjustment_delta": _signed_round(defense_adjusted - defense_raw),
                "recent_form_index": _round_or_blank(row.get("recent_form_index")),
                "warnings": _warnings(row),
            }
        )
    return pd.DataFrame(rows, columns=AUDIT_COLUMNS)


def write_schedule_audit_report(audit: pd.DataFrame, path: Path | None = None) -> Path:
    report_date = today_iso()
    report_path = path or PROJECT_ROOT / "reports" / f"schedule_strength_audit_{report_date}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    body = [
        f"# Schedule Strength Audit - {report_date}",
        "",
        "This audit checks whether recent behavior came against strong or weak opponents.",
        "",
        _to_markdown(audit),
        "",
        "Historical behavior is descriptive, not causal proof. Manual ratings remain the stable base.",
    ]
    report_path.write_text("\n".join(body), encoding="utf-8")
    return report_path


def main() -> None:
    audit = build_schedule_audit_table()
    print(audit.to_string(index=False))
    report_path = write_schedule_audit_report(audit)
    print(f"\nreport: {report_path.relative_to(PROJECT_ROOT)}")


def _warnings(row: pd.Series) -> str:
    warning_parts = [
        str(row.get("schedule_strength_warning", "") or ""),
        str(row.get("opponent_adjustment_warning", "") or ""),
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
