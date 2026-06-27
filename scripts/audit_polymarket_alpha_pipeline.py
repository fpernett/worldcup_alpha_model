from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_polymarket_slug_and_join import _model_markets_for_fixture  # noqa: E402
from src.polymarket_slug_join import (  # noqa: E402
    build_polymarket_alpha_for_fixture,
    build_polymarket_sports_slug_candidates,
    polymarket_alpha_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the registry-first Polymarket alpha pipeline for one fixture.")
    parser.add_argument("--home", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--fixture-date", required=True)
    parser.add_argument("--competition", default="World Cup")
    parser.add_argument("--url", default="")
    args = parser.parse_args()

    model_markets = _model_markets_for_fixture(args.home, args.away, args.fixture_date, args.competition)
    joined, diagnostics = build_polymarket_alpha_for_fixture(
        args.home,
        args.away,
        args.fixture_date,
        args.competition,
        model_markets,
        user_supplied_slug_or_url=args.url or None,
    )
    event_markets = joined.attrs.get("polymarket_event_markets", pd.DataFrame())
    alpha = polymarket_alpha_rows(joined)
    candidates = build_polymarket_sports_slug_candidates(args.home, args.away, args.fixture_date, args.competition)
    csv_path, md_path = _write_reports(args, diagnostics, candidates, event_markets, joined, alpha)

    print(f"fixture: {args.home} vs {args.away} {args.fixture_date}")
    print(f"user_url: {args.url}")
    print(f"registry_match_found: {diagnostics.get('registry_match_found', False)}")
    print("slug_candidates:")
    for candidate in candidates:
        print(f"- {candidate}")
    print(f"resolved_slug: {diagnostics.get('resolved_slug', '')}")
    print(f"resolution_status: {diagnostics.get('resolution_status', diagnostics.get('slug_resolution_status', ''))}")
    print(f"markets_loaded_count: {diagnostics.get('event_markets_loaded_count', 0)}")
    print(f"market_types_found: {', '.join(diagnostics.get('event_market_types_found', [])) or 'None'}")
    print(f"moneyline_rows: {_market_type_count(event_markets, 'moneyline')}")
    print(f"totals_rows: {_market_type_count(event_markets, 'total')}")
    print(f"spread_rows: {_market_type_count(event_markets, 'spread')}")
    print(f"btts_rows: {_market_type_count(event_markets, 'btts')}")
    print(f"model_markets_count: {diagnostics.get('model_markets_count', 0)}")
    print(f"joined_markets_count: {diagnostics.get('joined_markets_count', 0)}")
    print(f"top_alpha_rows_count: {diagnostics.get('top_alpha_rows_count', 0)}")
    print(f"polymarket_alpha_rows_count: {len(alpha)}")
    print(f"reason_no_rows: {diagnostics.get('reason_no_alpha_rows', '')}")
    print(f"warnings: {'; '.join(diagnostics.get('warnings', []))}")
    print(f"report CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"report: {md_path.relative_to(PROJECT_ROOT)}")


def _write_reports(
    args: argparse.Namespace,
    diagnostics: dict,
    candidates: list[str],
    event_markets: pd.DataFrame,
    joined: pd.DataFrame,
    alpha: pd.DataFrame,
) -> tuple[Path, Path]:
    reports = PROJECT_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    stem = f"polymarket_alpha_pipeline_{_report_part(args.home)}_{_report_part(args.away)}_{args.fixture_date}"
    csv_path = reports / f"{stem}.csv"
    md_path = reports / f"{stem}.md"
    joined.to_csv(csv_path, index=False)

    summary = pd.DataFrame(
        [
            {
                "fixture": f"{args.home} vs {args.away}",
                "user_url": args.url,
                "registry_match_found": diagnostics.get("registry_match_found", False),
                "resolved_slug": diagnostics.get("resolved_slug", ""),
                "resolution_status": diagnostics.get("resolution_status", diagnostics.get("slug_resolution_status", "")),
                "markets_loaded_count": diagnostics.get("event_markets_loaded_count", 0),
                "market_types_found": ", ".join(diagnostics.get("event_market_types_found", [])),
                "moneyline_rows": _market_type_count(event_markets, "moneyline"),
                "totals_rows": _market_type_count(event_markets, "total"),
                "spread_rows": _market_type_count(event_markets, "spread"),
                "btts_rows": _market_type_count(event_markets, "btts"),
                "model_markets_count": diagnostics.get("model_markets_count", 0),
                "joined_markets_count": diagnostics.get("joined_markets_count", 0),
                "top_alpha_rows_count": diagnostics.get("top_alpha_rows_count", 0),
                "polymarket_alpha_rows_count": len(alpha),
                "reason_no_rows": diagnostics.get("reason_no_alpha_rows", ""),
                "warnings": "; ".join(diagnostics.get("warnings", [])),
            }
        ]
    )
    md_path.write_text(
        "\n".join(
            [
                f"# Polymarket Alpha Pipeline Audit - {args.home} vs {args.away} {args.fixture_date}",
                "",
                "Read-only audit. No staking, wallet, private-key, order-placement, or trade execution logic is used.",
                "",
                "## Summary",
                "",
                _to_markdown(summary),
                "",
                "## Slug Candidates",
                "",
                _to_markdown(pd.DataFrame({"slug_candidate": candidates})),
                "",
                "## Diagnostics",
                "",
                _to_markdown(pd.DataFrame([diagnostics])),
                "",
                "## Event Markets",
                "",
                _to_markdown(event_markets.head(50)),
                "",
                "## Joined Markets",
                "",
                _to_markdown(joined.head(50)),
                "",
                "## Alpha Rows",
                "",
                _to_markdown(alpha.head(50)),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return csv_path, md_path


def _market_type_count(df: pd.DataFrame, market_type: str) -> int:
    if df is None or df.empty or "market_type" not in df.columns:
        return 0
    return int((df["market_type"].astype(str) == market_type).sum())


def _report_part(value: str) -> str:
    clean = "".join(ch if ch.isalnum() else "_" for ch in str(value).strip())
    return "_".join(part for part in clean.split("_") if part)


def _to_markdown(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_No rows._"
    display = df.copy()
    display.attrs = {}
    display = display.fillna("").astype(str)
    headers = list(display.columns)
    rows = display.values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", "<br>") for value in row) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
