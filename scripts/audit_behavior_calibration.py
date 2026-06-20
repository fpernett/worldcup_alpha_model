from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.behavior_calibration import DEFAULT_BEHAVIOR_CONFIG  # noqa: E402
from src.config import DATA_DIR  # noqa: E402
from src.ratings import TEAM_RATING_COLUMNS, _apply_behavior_blend, _normalise_ratings  # noqa: E402
from src.team_behavior import load_team_behavior  # noqa: E402
from src.utils import read_csv_with_columns, today_iso  # noqa: E402


AUDIT_COLUMNS = [
    "team",
    "matches_available_all_time",
    "matches_used_recent",
    "latest_match_used",
    "weighted_goals_for",
    "weighted_goals_against",
    "attack_index",
    "defense_index",
    "recent_form_index",
    "overall_data_quality",
    "manual_attack",
    "adjusted_attack",
    "attack_delta",
    "manual_defense",
    "adjusted_defense",
    "defense_delta",
    "manual_recent_form",
    "adjusted_recent_form",
    "recent_form_delta",
    "warnings",
]


def build_audit_table(teams: list[str] | None = None) -> pd.DataFrame:
    manual = _normalise_ratings(read_csv_with_columns(DATA_DIR / "team_ratings.csv", TEAM_RATING_COLUMNS), "manual_csv")
    adjusted = _apply_behavior_blend(manual.copy(), manual_base=manual)
    behavior = load_team_behavior()

    if teams:
        wanted = {team.lower() for team in teams}
        adjusted = adjusted.loc[adjusted["team"].astype(str).str.lower().isin(wanted)].copy()

    rows: list[dict[str, object]] = []
    for _, adjusted_row in adjusted.iterrows():
        team = str(adjusted_row.get("team", ""))
        behavior_row = _matching_row(behavior, team)

        manual_attack = _num(adjusted_row.get("manual_attack"), _num(adjusted_row.get("attack")))
        adjusted_attack = _num(adjusted_row.get("attack"), manual_attack)
        manual_defense = _num(adjusted_row.get("manual_defense"), _num(adjusted_row.get("defense")))
        adjusted_defense = _num(adjusted_row.get("defense"), manual_defense)
        manual_form = _num(adjusted_row.get("manual_recent_form"), _num(adjusted_row.get("recent_form")))
        adjusted_form = _num(adjusted_row.get("recent_form"), manual_form)

        rows.append(
            {
                "team": team,
                "matches_available_all_time": _int_or_blank(behavior_row.get("matches_available_all_time")),
                "matches_used_recent": _int_or_blank(behavior_row.get("matches_used_recent", behavior_row.get("n_matches"))),
                "latest_match_used": str(behavior_row.get("latest_match_used", "") or ""),
                "weighted_goals_for": _round_or_blank(behavior_row.get("weighted_goals_for")),
                "weighted_goals_against": _round_or_blank(behavior_row.get("weighted_goals_against")),
                "attack_index": _round_or_blank(behavior_row.get("attack_index")),
                "defense_index": _round_or_blank(behavior_row.get("defense_index")),
                "recent_form_index": _round_or_blank(behavior_row.get("recent_form_index")),
                "overall_data_quality": str(behavior_row.get("overall_data_quality", "none") or "none"),
                "manual_attack": _round_or_blank(manual_attack),
                "adjusted_attack": _round_or_blank(adjusted_attack),
                "attack_delta": _signed_round(adjusted_attack - manual_attack),
                "manual_defense": _round_or_blank(manual_defense),
                "adjusted_defense": _round_or_blank(adjusted_defense),
                "defense_delta": _signed_round(adjusted_defense - manual_defense),
                "manual_recent_form": _round_or_blank(manual_form),
                "adjusted_recent_form": _round_or_blank(adjusted_form),
                "recent_form_delta": _signed_round(adjusted_form - manual_form),
                "warnings": _warnings(behavior_row, adjusted_row),
            }
        )

    return pd.DataFrame(rows, columns=AUDIT_COLUMNS)


def write_audit_report(audit: pd.DataFrame, path: Path | None = None) -> Path:
    report_date = today_iso()
    report_path = path or PROJECT_ROOT / "reports" / f"behavior_calibration_audit_{report_date}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    body = [
        f"# Behavior Calibration Audit - {report_date}",
        "",
        "## Configuration",
        "",
        f"- lookback_years: {DEFAULT_BEHAVIOR_CONFIG.lookback_years}",
        f"- max_matches: {DEFAULT_BEHAVIOR_CONFIG.max_matches}",
        f"- min_matches: {DEFAULT_BEHAVIOR_CONFIG.min_matches}",
        f"- half_life_days: {DEFAULT_BEHAVIOR_CONFIG.half_life_days}",
        f"- friendly_weight: {DEFAULT_BEHAVIOR_CONFIG.friendly_weight}",
        f"- nations_league_weight: {DEFAULT_BEHAVIOR_CONFIG.nations_league_weight}",
        f"- qualifier_weight: {DEFAULT_BEHAVIOR_CONFIG.qualifier_weight}",
        f"- continental_weight: {DEFAULT_BEHAVIOR_CONFIG.continental_weight}",
        f"- world_cup_weight: {DEFAULT_BEHAVIOR_CONFIG.world_cup_weight}",
        f"- opponent_adjustment_strength: {DEFAULT_BEHAVIOR_CONFIG.opponent_adjustment_strength}",
        f"- goal_contribution_cap: {DEFAULT_BEHAVIOR_CONFIG.goal_contribution_cap}",
        f"- max_behavior_blend: {DEFAULT_BEHAVIOR_CONFIG.max_behavior_blend}",
        f"- max_form_blend: {DEFAULT_BEHAVIOR_CONFIG.max_form_blend}",
        "",
        "## Model Teams",
        "",
        _to_markdown(audit),
        "",
        "Historical behavior is descriptive, not causal proof. Manual ratings remain the stable base.",
    ]
    report_path.write_text("\n".join(body), encoding="utf-8")
    return report_path


def main() -> None:
    audit = build_audit_table()
    print(audit.to_string(index=False))
    report_path = write_audit_report(audit)
    print(f"\nreport: {report_path.relative_to(PROJECT_ROOT)}")


def _matching_row(df: pd.DataFrame, team: str) -> pd.Series:
    if df is None or df.empty or "team" not in df.columns:
        return pd.Series(dtype="object")
    row = df.loc[df["team"].astype(str).str.lower() == str(team).lower()]
    if row.empty:
        return pd.Series(dtype="object")
    return row.iloc[0]


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


def _warnings(behavior_row: pd.Series, adjusted_row: pd.Series) -> str:
    behavior_warning = str(behavior_row.get("behavior_warning", "") or "")
    warnings = [behavior_warning] if behavior_warning.strip() else [
        str(behavior_row.get("sample_size_warning", "") or ""),
        str(behavior_row.get("staleness_warning", "") or ""),
        str(behavior_row.get("opponent_quality_warning", "") or ""),
    ]
    if str(behavior_row.get("team", "") or "") == "":
        warnings.append("No behavior row.")
    if _truthy(adjusted_row.get("behavior_blend_cap_hit", False)):
        warnings.append("Behavior delta cap hit.")
    return "; ".join([warning for warning in warnings if warning.strip()])


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


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
