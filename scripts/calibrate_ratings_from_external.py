from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR  # noqa: E402
from src.external_benchmark_calibration import (  # noqa: E402
    CALIBRATED_PROPOSAL_COLUMNS,
    apply_external_calibration_proposals,
    build_external_calibration_proposals,
    external_benchmark_calibration_summary,
)
from src.external_priors import load_external_priors  # noqa: E402
from src.rating_coverage import load_team_ratings_csv  # noqa: E402
from src.team_behavior import load_team_behavior  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Create conservative external-benchmark calibrated rating proposals.")
    parser.add_argument("--output", default=str(DATA_DIR / "team_ratings_external_calibrated_proposed.csv"))
    parser.add_argument("--write", action="store_true", help="Update allowed rows in data/team_ratings.csv.")
    parser.add_argument(
        "--include-manual-existing",
        action="store_true",
        help="Also update manual_existing rows. By default these are proposal-only.",
    )
    parser.add_argument(
        "--include-reviewed",
        action="store_true",
        help="Allow manual_reviewed rows to be updated. By default they are preserved.",
    )
    args = parser.parse_args()

    ratings_path = DATA_DIR / "team_ratings.csv"
    output_path = _resolve_path(args.output)
    ratings = load_team_ratings_csv(ratings_path)
    behavior = load_team_behavior()
    priors = load_external_priors()

    proposals = build_external_calibration_proposals(ratings, behavior, priors, include_reviewed=args.include_reviewed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    proposals.to_csv(output_path, index=False)

    updated_count = 0
    if args.write:
        updated = apply_external_calibration_proposals(
            ratings,
            proposals,
            include_manual_existing=args.include_manual_existing,
            include_reviewed=args.include_reviewed,
        )
        updated_count = _changed_rows(ratings, updated)
        updated.to_csv(ratings_path, index=False)

    summary = external_benchmark_calibration_summary(proposals, ratings)
    print("External benchmark calibration")
    print(f"teams with external benchmark: {summary['teams_with_external_benchmark']}")
    print(f"teams missing external benchmark: {summary['teams_missing_external_benchmark']}")
    print(f"generated rows calibrated in proposal: {summary['generated_rows_calibrated']}")
    print(f"large disagreement warnings: {summary['large_disagreement_warnings']}")
    print(f"proposal output: {output_path.relative_to(PROJECT_ROOT)}")
    if args.write:
        print(f"updated rating rows: {updated_count}")
        print(f"updated: {ratings_path.relative_to(PROJECT_ROOT)}")
    else:
        print("data/team_ratings.csv was not modified. Pass --write to update allowed rows.")

    print("\nLargest upward changes")
    print(_printable(_largest_changes(proposals, ascending=False)).to_string(index=False) if not proposals.empty else "None")
    print("\nLargest downward changes")
    print(_printable(_largest_changes(proposals, ascending=True)).to_string(index=False) if not proposals.empty else "None")


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
        "calibrated_attack",
        "attack_delta",
        "current_defense",
        "calibrated_defense",
        "defense_delta",
        "current_recent_form",
        "calibrated_recent_form",
        "recent_form_delta",
        "warning",
    ]
    return out.sort_values(sort_col, ascending=ascending).head(n)[[col for col in cols if col in out.columns]]


def _changed_rows(before: pd.DataFrame, after: pd.DataFrame) -> int:
    if before.empty or after.empty or "team" not in before.columns or "team" not in after.columns:
        return 0
    changed = 0
    before_by_team = before.set_index(before["team"].astype(str).str.lower(), drop=False)
    for _, row in after.iterrows():
        key = _text_value(row.get("team", "")).lower()
        if not key or key not in before_by_team.index:
            continue
        old = before_by_team.loc[key]
        if isinstance(old, pd.DataFrame):
            old = old.iloc[0]
        if any(_text_value(row.get(col, "")) != _text_value(old.get(col, "")) for col in ["attack", "defense", "recent_form", "data_quality"]):
            changed += 1
    return changed


def _printable(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda value: "" if pd.isna(value) else round(float(value), 4))
    return out


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def _text_value(value) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


if __name__ == "__main__":
    main()
