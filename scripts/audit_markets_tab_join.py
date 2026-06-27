from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_polymarket_slug_and_join import _model_markets_for_fixture  # noqa: E402
from src.polymarket_slug_join import (  # noqa: E402
    build_markets_tab_joined_dataframe,
    build_polymarket_alpha_for_fixture,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Markets-tab Polymarket price join wiring for one fixture.")
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
    markets_tab_df, markets_tab_diagnostics = build_markets_tab_joined_dataframe(model_markets, joined, diagnostics)
    csv_path, md_path = _write_reports(args, markets_tab_df, markets_tab_diagnostics, diagnostics)

    print("Markets tab join audit")
    print(f"model_market_rows: {len(model_markets)}")
    print(f"resolved_slug: {markets_tab_diagnostics.get('resolved_slug', '')}")
    print(f"event_markets_loaded: {markets_tab_diagnostics.get('event_markets_loaded', 0)}")
    print(f"joined_market_rows: {markets_tab_diagnostics.get('joined_markets', 0)}")
    print(f"rows_with_market_odds: {markets_tab_diagnostics.get('rows_with_market_odds', 0)}")
    print(f"rows_with_alpha_ev: {markets_tab_diagnostics.get('rows_with_alpha_ev', 0)}")
    print(f"odds_sources: {', '.join(_odds_sources(markets_tab_df)) or 'None'}")
    print(f"empty_column_warning: {markets_tab_diagnostics.get('reason_if_zero', '')}")
    print(f"report CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"report: {md_path.relative_to(PROJECT_ROOT)}")


def _write_reports(
    args: argparse.Namespace,
    markets_tab_df: pd.DataFrame,
    markets_tab_diagnostics: dict,
    alpha_diagnostics: dict,
) -> tuple[Path, Path]:
    reports = PROJECT_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    stem = f"markets_tab_join_{_report_part(args.home)}_{_report_part(args.away)}_{args.fixture_date}"
    csv_path = reports / f"{stem}.csv"
    md_path = reports / f"{stem}.md"
    markets_tab_df.to_csv(csv_path, index=False)

    summary = pd.DataFrame(
        [
            {
                "fixture": f"{args.home} vs {args.away}",
                "user_url": args.url,
                "model_market_rows": len(markets_tab_df),
                "resolved_slug": markets_tab_diagnostics.get("resolved_slug", ""),
                "event_markets_loaded": markets_tab_diagnostics.get("event_markets_loaded", 0),
                "joined_market_rows": markets_tab_diagnostics.get("joined_markets", 0),
                "rows_with_market_odds": markets_tab_diagnostics.get("rows_with_market_odds", 0),
                "rows_with_alpha_ev": markets_tab_diagnostics.get("rows_with_alpha_ev", 0),
                "odds_sources": ", ".join(_odds_sources(markets_tab_df)),
                "empty_column_warning": markets_tab_diagnostics.get("reason_if_zero", ""),
                "alpha_reason_no_rows": alpha_diagnostics.get("reason_no_alpha_rows", ""),
                "warnings": "; ".join(alpha_diagnostics.get("warnings", [])),
            }
        ]
    )
    md_path.write_text(
        "\n".join(
            [
                f"# Markets Tab Join Audit - {args.home} vs {args.away} {args.fixture_date}",
                "",
                "Read-only audit. No staking, wallet, private-key, order-placement, or trade execution logic is used.",
                "",
                "## Summary",
                "",
                _to_markdown(summary),
                "",
                "## Markets Tab Dataframe",
                "",
                _to_markdown(markets_tab_df.head(100)),
                "",
                "## Pipeline Diagnostics",
                "",
                _to_markdown(pd.DataFrame([alpha_diagnostics])),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return csv_path, md_path


def _odds_sources(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty or "odds_source" not in df.columns:
        return []
    return sorted(value for value in df["odds_source"].dropna().astype(str).unique().tolist() if value)


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
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in display.values.tolist():
        lines.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", "<br>") for value in row) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
