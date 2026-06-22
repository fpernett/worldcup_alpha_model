from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_sources import get_upcoming_fixtures  # noqa: E402
from src.model import ModelConfig  # noqa: E402
from src.odds import load_market_odds  # noqa: E402
from src.prediction_ledger import PREDICTION_LEDGER_PATH, snapshot_predictions_for_fixtures  # noqa: E402
from src.ratings import get_team_ratings  # noqa: E402
from src.weather import load_venues  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Append pre-match model predictions to data/prediction_ledger.csv.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--competition", default=None)
    parser.add_argument("--model-version", default="baseline_external_calibrated_v1")
    parser.add_argument("--parameter-set-id", default="baseline_current")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    fixtures = get_upcoming_fixtures(start_date=args.start_date, end_date=args.end_date)
    if args.competition and not fixtures.empty and "competition" in fixtures.columns:
        needle = str(args.competition).strip().lower()
        fixtures = fixtures.loc[fixtures["competition"].astype(str).str.lower().str.contains(needle, na=False)].copy()

    ratings = get_team_ratings(model_mode="baseline_manual")
    snapshots = snapshot_predictions_for_fixtures(
        fixtures,
        team_ratings_df=ratings,
        venues_df=load_venues(),
        market_odds_df=load_market_odds(),
        cfg=ModelConfig(),
        model_version=args.model_version,
        parameter_set_id=args.parameter_set_id,
        primary_model_mode="baseline_manual",
        notes=args.notes,
        path=PREDICTION_LEDGER_PATH,
        append=True,
    )

    print(f"fixtures considered: {len(fixtures):,}")
    print(f"predictions snapshotted: {len(snapshots):,}")
    print(f"ledger: {PREDICTION_LEDGER_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
