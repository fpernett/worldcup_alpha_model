from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.polymarket_match_search import (  # noqa: E402
    build_polymarket_match_queries,
    candidate_summary,
    find_best_polymarket_match_candidates,
    write_match_search_audit_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit match-aware Polymarket market discovery for one fixture.")
    parser.add_argument("--home", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--competition", default="World Cup")
    parser.add_argument("--fixture-date", default="")
    parser.add_argument("--max-candidates", type=int, default=25)
    args = parser.parse_args()

    queries = build_polymarket_match_queries(args.home, args.away, args.competition)
    candidates = find_best_polymarket_match_candidates(
        args.home,
        args.away,
        fixture_date=args.fixture_date or None,
        competition=args.competition,
        max_candidates=args.max_candidates,
        include_rejected=True,
    )
    summary = candidate_summary(candidates)
    csv_path, report_path = write_match_search_audit_report(
        candidates,
        queries,
        args.home,
        args.away,
        args.competition,
        report_dir=PROJECT_ROOT / "reports",
    )

    accepted = candidates.loc[~candidates["rejected"].astype(bool)].copy() if not candidates.empty else pd.DataFrame()
    rejected = candidates.loc[candidates["rejected"].astype(bool)].copy() if not candidates.empty else pd.DataFrame()
    best = accepted.head(1)

    print("Polymarket match search audit")
    print("queries used:")
    for query in queries:
        print(f"- {query}")
    print(f"markets returned: {summary['markets_returned']}")
    print(f"accepted candidates: {summary['accepted_candidates']}")
    print(f"rejected candidates: {summary['rejected_candidates']}")
    if not best.empty:
        print(f"best candidate: {best.iloc[0].get('question', '')}")
        print(f"confidence: {best.iloc[0].get('confidence', '')}")
        print(f"reason: {best.iloc[0].get('score_breakdown', '')}")
    else:
        print("best candidate: None")
        print("confidence: n/a")
        print("reason: no accepted candidates")
    if not rejected.empty:
        print("top rejected candidates:")
        for _, row in rejected.head(5).iterrows():
            print(f"- {row.get('question', '')} | {row.get('reject_reason', '')}")
    print(f"report CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"report: {report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
