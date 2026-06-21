from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR  # noqa: E402
from src.rating_coverage import load_team_ratings_csv  # noqa: E402
from src.rating_review import (  # noqa: E402
    apply_review_proposals,
    audit_generated_ratings,
    build_review_proposals,
)
from src.team_behavior import load_team_behavior  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Create conservative manual-review proposals for generated team ratings.")
    parser.add_argument("--teams", nargs="*", default=None)
    parser.add_argument("--output", default=str(DATA_DIR / "team_ratings_review_proposals.csv"))
    parser.add_argument("--write", action="store_true", help="Update generated_from_behavior rows in data/team_ratings.csv.")
    args = parser.parse_args()

    ratings_path = DATA_DIR / "team_ratings.csv"
    ratings = load_team_ratings_csv(ratings_path)
    behavior = load_team_behavior()
    audit = audit_generated_ratings(ratings, behavior, pd.DataFrame())
    proposals = build_review_proposals(args.teams, ratings, behavior, audit)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    proposals.to_csv(output_path, index=False)

    print("Review proposals")
    print(_printable(proposals).to_string(index=False) if not proposals.empty else "No rating review proposals generated.")
    print(f"\nproposal output: {output_path.relative_to(PROJECT_ROOT)}")

    if args.write:
        updated = apply_review_proposals(ratings, proposals)
        changed = _changed_generated_rows(ratings, updated)
        updated.to_csv(ratings_path, index=False)
        print(f"updated generated rows: {changed}")
        print(f"updated: {ratings_path.relative_to(PROJECT_ROOT)}")
    else:
        print("data/team_ratings.csv was not modified. Pass --write to update generated_from_behavior rows.")


def _changed_generated_rows(before: pd.DataFrame, after: pd.DataFrame) -> int:
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
        old_quality = _text_value(old.get("data_quality", ""))
        if old_quality != "generated_from_behavior":
            continue
        if _text_value(row.get("data_quality", "")) != old_quality:
            changed += 1
    return changed


def _printable(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda value: "" if pd.isna(value) else round(float(value), 4))
    return out


def _text_value(value) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


if __name__ == "__main__":
    main()
