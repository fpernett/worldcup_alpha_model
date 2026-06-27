from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.polymarket_event_registry import (  # noqa: E402
    DEFAULT_POLYMARKET_EVENT_REGISTRY_PATH,
    POLYMARKET_EVENT_REGISTRY_COLUMNS,
    load_polymarket_event_registry,
)
from src.polymarket_url_resolver import parse_polymarket_url_or_slug, score_polymarket_event_slug_for_fixture  # noqa: E402
from src.utils import utc_now_iso  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Append or update one Polymarket sports event registry row.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--home", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--competition", default="World Cup")
    parser.add_argument("--event-date", required=True)
    args = parser.parse_args()

    parsed = parse_polymarket_url_or_slug(args.url)
    slug = str(parsed.get("slug", "") or "").strip()
    score = score_polymarket_event_slug_for_fixture(slug, args.home, args.away, args.event_date)
    if not slug or score.get("confidence") not in {"high", "medium"}:
        raise SystemExit(f"Slug did not validate against fixture: {score.get('warning', parsed.get('warning', 'invalid slug'))}")

    registry = load_polymarket_event_registry(str(DEFAULT_POLYMARKET_EVENT_REGISTRY_PATH))
    row = {
        "competition": args.competition,
        "event_date": args.event_date,
        "event_slug": slug,
        "event_url": args.url,
        "home": args.home,
        "away": args.away,
        "home_code": parsed.get("team_code_1", ""),
        "away_code": parsed.get("team_code_2", ""),
        "event_title": f"{args.home} - {args.away}",
        "source": "manual",
        "last_checked_utc": utc_now_iso(),
        "confidence": score.get("confidence", "high"),
        "notes": "Added from known Polymarket sports event URL.",
    }
    if registry.empty:
        registry = pd.DataFrame(columns=POLYMARKET_EVENT_REGISTRY_COLUMNS)
    registry = registry.loc[registry["event_slug"].astype(str) != slug].copy()
    registry = pd.concat([registry, pd.DataFrame([row])], ignore_index=True)
    registry = registry[POLYMARKET_EVENT_REGISTRY_COLUMNS].sort_values(["event_date", "event_slug"]).reset_index(drop=True)
    DEFAULT_POLYMARKET_EVENT_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    registry.to_csv(DEFAULT_POLYMARKET_EVENT_REGISTRY_PATH, index=False)
    print(f"updated registry: {DEFAULT_POLYMARKET_EVENT_REGISTRY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"event_slug: {slug}")
    print(f"confidence: {score.get('confidence', '')}")


if __name__ == "__main__":
    main()
