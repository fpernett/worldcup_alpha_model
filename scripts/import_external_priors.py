from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR  # noqa: E402
from src.external_prior_import import (  # noqa: E402
    build_external_prior_updates,
    calculate_generated_rating_disagreements,
    import_diagnostics_summary,
    load_external_strength_input,
    merge_external_prior_updates,
)
from src.external_priors import load_external_priors  # noqa: E402
from src.rating_coverage import load_team_ratings_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Import manually entered external rank/Elo data into proposed external priors.")
    parser.add_argument("--input", default=str(DATA_DIR / "raw" / "external_team_strength.csv"))
    parser.add_argument("--output", default=str(DATA_DIR / "team_rating_external_priors.csv"))
    parser.add_argument("--write", action="store_true", help="Update matching teams in data/team_rating_external_priors.csv.")
    args = parser.parse_args()

    input_path = _resolve_path(args.input)
    output_path = _resolve_path(args.output)
    proposed_path = output_path.with_name(f"{output_path.stem}_proposed{output_path.suffix}")

    input_df = load_external_strength_input(input_path)
    existing = load_external_priors(output_path)
    proposed, diagnostics = build_external_prior_updates(input_df, existing)
    ratings = load_team_ratings_csv()
    disagreements = calculate_generated_rating_disagreements(proposed, ratings)

    proposed_path.parent.mkdir(parents=True, exist_ok=True)
    proposed.to_csv(proposed_path, index=False)

    updated_count = 0
    if args.write:
        updated = merge_external_prior_updates(existing, proposed, diagnostics)
        updated_count = _changed_rows(existing, updated)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        updated.to_csv(output_path, index=False)

    summary = import_diagnostics_summary(diagnostics, updated_count=updated_count)
    print(f"input teams read: {summary['input_teams_read']}")
    print(f"teams matched to external prior table: {summary['teams_matched_to_external_prior_table']}")
    print(f"teams not found: {_team_list(diagnostics.loc[~diagnostics['matched_external_prior'].astype(bool)] if not diagnostics.empty else diagnostics)}")
    print(f"teams updated: {summary['teams_updated']}")
    print(f"teams skipped: {summary['teams_skipped']}")
    print(f"missing rank/Elo data: {_team_list(_warning_rows(diagnostics, 'missing rank/Elo data'))}")
    print(f"large disagreement with generated ratings: {_team_list(_warning_rows(disagreements, 'large disagreement'))}")
    print(f"\nproposed output: {proposed_path.relative_to(PROJECT_ROOT)}")
    if args.write:
        print(f"updated external priors: {output_path.relative_to(PROJECT_ROOT)}")
    else:
        print(f"{output_path.relative_to(PROJECT_ROOT)} was not modified. Pass --write to update matching teams.")

    print("\nNext commands:")
    print(".venv/bin/python scripts/audit_external_priors.py")
    print(
        ".venv/bin/python scripts/build_rating_review_sheet.py "
        '--teams Argentina Brazil France Germany Spain Netherlands Portugal England Uruguay Morocco "United States" Norway '
        "--output data/manual_rating_review_sheet.csv"
    )


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
        if any(_text_value(row.get(col, "")) != _text_value(old.get(col, "")) for col in after.columns if col in old.index):
            changed += 1
    return changed


def _warning_rows(df: pd.DataFrame, pattern: str) -> pd.DataFrame:
    if df is None or df.empty or "warning" not in df.columns:
        return pd.DataFrame()
    return df.loc[df["warning"].astype(str).str.contains(pattern, case=False, na=False)]


def _team_list(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "None"
    col = "team" if "team" in df.columns else "canonical_team"
    if col not in df.columns:
        return "None"
    return ", ".join(df[col].dropna().astype(str).tolist()) or "None"


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
