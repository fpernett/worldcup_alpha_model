from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.external_benchmark_calibration import (  # noqa: E402
    CALIBRATED_PROPOSAL_COLUMNS,
    build_external_calibration_proposals,
    external_benchmark_calibration_summary,
)
from src.external_priors import load_external_priors  # noqa: E402
from src.rating_coverage import load_team_ratings_csv  # noqa: E402
from src.team_behavior import load_team_behavior  # noqa: E402
from src.utils import today_iso  # noqa: E402


def main() -> None:
    ratings = load_team_ratings_csv()
    behavior = load_team_behavior()
    priors = load_external_priors()
    proposals = build_external_calibration_proposals(ratings, behavior, priors)
    summary = external_benchmark_calibration_summary(proposals, ratings)

    missing = _warning_rows(proposals, "missing_external_benchmark")
    large = _warning_rows(proposals, "internal_.*external|behavior_above_external|large_delta_capped")
    generated = _warning_rows(proposals, "generated_rating_calibrated")
    preserved = _warning_rows(proposals, "manual_existing_calibration_suggested|manual_reviewed_preserved")
    upward = _largest_changes(proposals, ascending=False)
    downward = _largest_changes(proposals, ascending=True)

    print(f"teams with external benchmark: {summary['teams_with_external_benchmark']}")
    print(f"teams missing external benchmark: {_team_list(missing)}")
    print(f"generated rows calibrated: {len(generated)}")
    print(f"manual rows preserved or suggested: {len(preserved)}")
    print(f"large disagreements: {_team_list(large)}")
    print(f"largest upward rating changes: {_team_list(upward)}")
    print(f"largest downward rating changes: {_team_list(downward)}")
    print(f"teams still needing external benchmark: {_team_list(missing)}")
    print("team_ratings.csv modified by this audit: no")
    print(f"current external_benchmark_calibrated rows: {summary['current_external_benchmark_calibrated_rows']}")

    reports = PROJECT_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    today = today_iso()
    csv_path = reports / f"external_benchmark_calibration_{today}.csv"
    report_path = reports / f"external_benchmark_calibration_audit_{today}.md"
    proposals.to_csv(csv_path, index=False)
    report_path.write_text(
        "\n\n".join(
            [
                f"# External Benchmark Calibration Audit - {today}",
                "",
                "This audit compares internal team ratings with independent FIFA/Elo-style benchmark strength inputs. It is a quality-control workflow only.",
                "",
                "## Summary",
                "",
                _to_markdown(pd.DataFrame([summary])),
                "",
                "## Missing External Benchmarks",
                "",
                _to_markdown(_printable(missing)),
                "",
                "## Large Disagreements",
                "",
                _to_markdown(_printable(large)),
                "",
                "## Largest Upward Changes",
                "",
                _to_markdown(_printable(upward)),
                "",
                "## Largest Downward Changes",
                "",
                _to_markdown(_printable(downward)),
                "",
                "The audit did not modify `data/team_ratings.csv`.",
            ]
        ),
        encoding="utf-8",
    )
    print(f"\ncomparison CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"report: {report_path.relative_to(PROJECT_ROOT)}")


def _warning_rows(df: pd.DataFrame, pattern: str) -> pd.DataFrame:
    if df is None or df.empty or "warning" not in df.columns:
        return pd.DataFrame(columns=CALIBRATED_PROPOSAL_COLUMNS)
    return df.loc[df["warning"].astype(str).str.contains(pattern, case=False, na=False, regex=True)].copy()


def _largest_changes(proposals: pd.DataFrame, ascending: bool, n: int = 10) -> pd.DataFrame:
    if proposals.empty:
        return pd.DataFrame(columns=CALIBRATED_PROPOSAL_COLUMNS)
    out = proposals.copy()
    for col in ["attack_delta", "defense_delta", "recent_form_delta"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["max_metric_delta"] = out[["attack_delta", "defense_delta", "recent_form_delta"]].max(axis=1)
    out["min_metric_delta"] = out[["attack_delta", "defense_delta", "recent_form_delta"]].min(axis=1)
    sort_col = "max_metric_delta" if not ascending else "min_metric_delta"
    cols = [
        "team",
        "external_overall_strength",
        "current_attack",
        "current_defense",
        "current_recent_form",
        "calibrated_attack",
        "calibrated_defense",
        "calibrated_recent_form",
        "attack_delta",
        "defense_delta",
        "recent_form_delta",
        "warning",
    ]
    return out.sort_values(sort_col, ascending=ascending).head(n)[[col for col in cols if col in out.columns]]


def _team_list(df: pd.DataFrame) -> str:
    if df is None or df.empty or "team" not in df.columns:
        return "None"
    return ", ".join(df["team"].dropna().astype(str).tolist()) or "None"


def _printable(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda value: "" if pd.isna(value) else round(float(value), 4))
    return out


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
