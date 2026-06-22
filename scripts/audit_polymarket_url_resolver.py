from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.polymarket_url_resolver import (  # noqa: E402
    parse_polymarket_url_or_slug,
    resolve_polymarket_event_markets_from_url_or_slug,
    score_polymarket_event_slug_for_fixture,
)
from src.utils import today_iso  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit exact Polymarket sports event URL/slug resolution.")
    parser.add_argument("--url", required=True, help="Polymarket sports event URL or event slug.")
    parser.add_argument("--home", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--fixture-date", default="")
    parser.add_argument("--competition", default="World Cup")
    args = parser.parse_args()

    parsed = parse_polymarket_url_or_slug(args.url)
    markets, diagnostics = resolve_polymarket_event_markets_from_url_or_slug(args.url)
    slug_score = score_polymarket_event_slug_for_fixture(
        parsed.get("slug", ""),
        args.home,
        args.away,
        fixture_date=args.fixture_date or None,
    )
    csv_path, report_path = _write_report(args, parsed, markets, diagnostics, slug_score)
    market_types = sorted(markets["market_type"].dropna().astype(str).unique().tolist()) if not markets.empty else []

    print("Polymarket URL resolver audit")
    print(f"parsed slug: {parsed.get('slug', '')}")
    print(f"parsed team codes: {parsed.get('team_code_1', '')}, {parsed.get('team_code_2', '')}")
    print(f"canonical teams from slug: {slug_score.get('slug_team_1', '')}, {slug_score.get('slug_team_2', '')}")
    print(f"fixture teams: {args.home}, {args.away}")
    print(f"date match: {slug_score.get('matched_date', False)}")
    print(f"event title: {diagnostics.get('event_title', '')}")
    print(f"markets extracted: {len(markets)}")
    print(f"market types found: {', '.join(market_types) or 'None'}")
    print(f"best mapping confidence: {slug_score.get('confidence', '')}")
    print(f"candidate moneyline markets: {_count_type(markets, 'moneyline')}")
    print(f"candidate spread markets: {_count_type(markets, 'spread')}")
    print(f"candidate totals markets: {_count_type(markets, 'total')}")
    warnings = [parsed.get("warning", ""), diagnostics.get("warning", ""), slug_score.get("warning", "")]
    print(f"warnings: {'; '.join(part for part in warnings if part) or 'None'}")
    print(f"report CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"report: {report_path.relative_to(PROJECT_ROOT)}")


def _write_report(
    args: argparse.Namespace,
    parsed: dict,
    markets: pd.DataFrame,
    diagnostics: dict,
    slug_score: dict,
) -> tuple[Path, Path]:
    reports = PROJECT_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    today = today_iso()
    csv_path = reports / f"polymarket_url_resolver_{today}.csv"
    report_path = reports / f"polymarket_url_resolver_{today}.md"
    markets.to_csv(csv_path, index=False)
    summary = pd.DataFrame(
        [
            {
                "slug": parsed.get("slug", ""),
                "team_code_1": parsed.get("team_code_1", ""),
                "team_code_2": parsed.get("team_code_2", ""),
                "slug_team_1": slug_score.get("slug_team_1", ""),
                "slug_team_2": slug_score.get("slug_team_2", ""),
                "fixture_home": args.home,
                "fixture_away": args.away,
                "matched_home": slug_score.get("matched_home", False),
                "matched_away": slug_score.get("matched_away", False),
                "matched_date": slug_score.get("matched_date", False),
                "confidence": slug_score.get("confidence", ""),
                "markets_extracted": len(markets),
                "market_types_found": ", ".join(sorted(markets["market_type"].dropna().astype(str).unique().tolist())) if not markets.empty else "",
                "extraction_method": diagnostics.get("fetch", {}).get("method", ""),
                "warning": "; ".join(
                    part
                    for part in [parsed.get("warning", ""), diagnostics.get("warning", ""), slug_score.get("warning", "")]
                    if part
                ),
            }
        ]
    )
    report_path.write_text(
        "\n".join(
            [
                f"# Polymarket URL Resolver Audit - {today}",
                "",
                f"Input: `{args.url}`",
                f"Fixture: **{args.home} vs {args.away}**",
                f"Fixture date: `{args.fixture_date}`",
                f"Competition: `{args.competition}`",
                "",
                "This report audits read-only URL/slug market discovery. It does not place trades, size positions, or change the football model.",
                "",
                "## Summary",
                "",
                _to_markdown(summary),
                "",
                "## Parsed URL",
                "",
                _to_markdown(pd.DataFrame([parsed])),
                "",
                "## Slug Fixture Score",
                "",
                _to_markdown(pd.DataFrame([slug_score])),
                "",
                "## Extracted Markets",
                "",
                _to_markdown(markets),
            ]
        ),
        encoding="utf-8",
    )
    return csv_path, report_path


def _count_type(markets: pd.DataFrame, market_type: str) -> int:
    if markets is None or markets.empty or "market_type" not in markets.columns:
        return 0
    return int((markets["market_type"].astype(str) == market_type).sum())


def _to_markdown(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_No rows._"
    display = df.head(50).copy()
    headers = list(display.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in display.iterrows():
        values = [str(row.get(col, "")).replace("|", "\\|").replace("\n", " ") for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
