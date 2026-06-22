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
)
from src.polymarket_sports_discovery import polymarket_discovery_cache_status  # noqa: E402
from src.utils import today_iso  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit broad Gamma sports event discovery for one Polymarket matchup.")
    parser.add_argument("--home", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--competition", default="World Cup")
    parser.add_argument("--fixture-date", default="")
    parser.add_argument(
        "--retrieval-mode",
        default="auto",
        choices=["query_only", "sports_events", "all_events", "cache", "auto"],
    )
    parser.add_argument("--include-closed", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=50)
    args = parser.parse_args()

    queries = build_polymarket_match_queries(args.home, args.away, args.competition)
    candidates = find_best_polymarket_match_candidates(
        args.home,
        args.away,
        fixture_date=args.fixture_date or None,
        competition=args.competition,
        max_candidates=args.max_candidates,
        include_rejected=True,
        retrieval_mode=args.retrieval_mode,
        min_confidence="low",
        closed=None if args.include_closed else False,
    )
    summary = candidate_summary(candidates)
    cache_status = polymarket_discovery_cache_status()
    csv_path, report_path = _write_report(args, queries, candidates, summary, cache_status)
    accepted = candidates.loc[~candidates["rejected"].astype(bool)].copy() if not candidates.empty else pd.DataFrame()
    rejected = candidates.loc[candidates["rejected"].astype(bool)].copy() if not candidates.empty else pd.DataFrame()
    best = accepted.head(1)

    print("Polymarket sports discovery audit")
    print(f"retrieval layers tried: {', '.join(candidates.attrs.get('retrieval_layers_tried', [])) or 'None'}")
    print("queries used:")
    for query in queries:
        print(f"- {query}")
    print(f"events fetched: {candidates.attrs.get('events_fetched', 0)}")
    print(f"markets flattened: {candidates.attrs.get('markets_flattened', 0)}")
    print(f"markets scored: {candidates.attrs.get('markets_scored', 0)}")
    print(f"accepted candidates: {len(accepted)}")
    print(f"rejected candidates: {len(rejected)}")
    if not best.empty:
        print(f"best candidate: {best.iloc[0].get('question', '')}")
        print(f"confidence: {best.iloc[0].get('confidence', '')}")
        print(f"reason: {best.iloc[0].get('score_breakdown', '')}")
    else:
        print("best candidate: None")
        print("confidence: n/a")
        print(f"reason no match: {candidates.attrs.get('no_match_reason', _no_match_reason())}")
        print("Next steps:")
        for step in _next_steps(args, candidates):
            print(f"- {step}")
    print(f"cache paths written: {cache_status.get('events_cache_path')}, {cache_status.get('markets_cache_path')}")
    print(f"report CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"report: {report_path.relative_to(PROJECT_ROOT)}")


def _write_report(
    args: argparse.Namespace,
    queries: list[str],
    candidates: pd.DataFrame,
    summary: dict,
    cache_status: dict,
) -> tuple[Path, Path]:
    reports = PROJECT_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    today = today_iso()
    csv_path = reports / f"polymarket_sports_discovery_{today}.csv"
    report_path = reports / f"polymarket_sports_discovery_{today}.md"
    candidates.to_csv(csv_path, index=False)
    accepted = candidates.loc[~candidates["rejected"].astype(bool)].copy() if not candidates.empty else pd.DataFrame()
    rejected = candidates.loc[candidates["rejected"].astype(bool)].copy() if not candidates.empty else pd.DataFrame()
    diagnostics = pd.DataFrame(
        [
            {
                "retrieval_layers_tried": ", ".join(candidates.attrs.get("retrieval_layers_tried", [])),
                "events_fetched": candidates.attrs.get("events_fetched", 0),
                "markets_flattened": candidates.attrs.get("markets_flattened", 0),
                "markets_scored": candidates.attrs.get("markets_scored", 0),
                "accepted_candidates": len(accepted),
                "rejected_candidates": len(rejected),
                "no_match_reason": candidates.attrs.get("no_match_reason", ""),
                **cache_status,
            }
        ]
    )
    report_path.write_text(
        "\n".join(
            [
                f"# Polymarket Sports Discovery Audit - {today}",
                "",
                f"Match: **{args.home} vs {args.away}**",
                f"Competition: `{args.competition}`",
                f"Fixture date: `{args.fixture_date}`",
                f"Retrieval mode: `{args.retrieval_mode}`",
                "",
                "This report audits read-only Gamma event discovery. It does not place trades, size positions, or change the football model.",
                "",
                "## Summary",
                "",
                _to_markdown(pd.DataFrame([summary])),
                "",
                "## Diagnostics",
                "",
                _to_markdown(diagnostics),
                "",
                "## Queries Used",
                "",
                "\n".join(f"- `{query}`" for query in queries) or "_No queries._",
                "",
                "## Accepted Candidates",
                "",
                _to_markdown(accepted),
                "",
                "## Rejected Candidates",
                "",
                _to_markdown(rejected),
            ]
        ),
        encoding="utf-8",
    )
    return csv_path, report_path


def _to_markdown(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_No rows._"
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        values = [str(row.get(col, "")).replace("|", "\\|") for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _no_match_reason() -> str:
    return (
        "No Polymarket match market was discovered. Possible reasons: market does not exist yet; market is nested under an event "
        "not returned by active filters; team names differ from model aliases; market is closed/resolved; API cache is stale."
    )


def _next_steps(args: argparse.Namespace, candidates: pd.DataFrame) -> list[str]:
    layers = set(candidates.attrs.get("retrieval_layers_tried", []))
    steps: list[str] = []
    if args.retrieval_mode != "all_events" and "all_events" not in layers:
        steps.append("run with --retrieval-mode all_events")
    if not args.include_closed:
        steps.append("run with --include-closed")
    steps.append("inspect Polymarket manually and add the market to data/market_mappings.csv")
    return steps


if __name__ == "__main__":
    main()
