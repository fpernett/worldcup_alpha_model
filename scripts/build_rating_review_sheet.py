from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR  # noqa: E402
from src.external_priors import (  # noqa: E402
    DEFAULT_REVIEW_SHEET_PATH,
    apply_approved_review_sheet,
    build_manual_rating_review_sheet,
    load_external_priors,
    load_manual_rating_review_sheet,
    load_review_proposals,
)
from src.historical_data import load_historical_matches  # noqa: E402
from src.rating_coverage import load_team_ratings_csv  # noqa: E402
from src.rating_review import audit_generated_ratings  # noqa: E402
from src.team_behavior import load_team_behavior  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or apply the manual rating review worksheet.")
    parser.add_argument("--teams", nargs="*", default=None)
    parser.add_argument("--output", default=str(DEFAULT_REVIEW_SHEET_PATH))
    parser.add_argument("--write", action="store_true", help="Apply rows marked review_status=approved to data/team_ratings.csv.")
    args = parser.parse_args()

    ratings_path = DATA_DIR / "team_ratings.csv"
    ratings = load_team_ratings_csv(ratings_path)
    behavior = load_team_behavior()
    historical = load_historical_matches(use_cache=False)
    proposals = load_review_proposals(DATA_DIR / "team_ratings_review_proposals.csv")
    priors = load_external_priors()
    audit = audit_generated_ratings(ratings, behavior, historical)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.write:
        if output_path.exists():
            sheet = load_manual_rating_review_sheet(output_path)
        else:
            sheet = build_manual_rating_review_sheet(ratings, audit, proposals, priors, args.teams)
            sheet.to_csv(output_path, index=False)
        updated = apply_approved_review_sheet(ratings, sheet)
        changed = _changed_rows(ratings, updated)
        updated.to_csv(ratings_path, index=False)
        print(f"approved rows applied: {changed}")
        print(f"updated: {ratings_path.relative_to(PROJECT_ROOT)}")
        print(f"review worksheet: {output_path.relative_to(PROJECT_ROOT)}")
        return

    sheet = build_manual_rating_review_sheet(ratings, audit, proposals, priors, args.teams)
    sheet.to_csv(output_path, index=False)
    print(f"review worksheet rows: {len(sheet)}")
    print(f"review worksheet: {output_path.relative_to(PROJECT_ROOT)}")
    print("data/team_ratings.csv was not modified. Edit review_status to approved and pass --write to apply approved rows.")


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
        if (
            _float_value(row.get("attack", 0)) != _float_value(old.get("attack", 0))
            or _float_value(row.get("defense", 0)) != _float_value(old.get("defense", 0))
            or _float_value(row.get("recent_form", 0)) != _float_value(old.get("recent_form", 0))
            or _text_value(row.get("data_quality", "")) != _text_value(old.get("data_quality", ""))
        ):
            changed += 1
    return changed


def _text_value(value) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _float_value(value) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


if __name__ == "__main__":
    main()
