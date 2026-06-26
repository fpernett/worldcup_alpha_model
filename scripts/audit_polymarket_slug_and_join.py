from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import ModelConfig, market_probability_map, run_match_model  # noqa: E402
from src.polymarket_slug_join import (  # noqa: E402
    build_polymarket_sports_slug_candidates,
    join_polymarket_prices_to_model_markets,
    load_polymarket_event_markets_by_slug,
    polymarket_alpha_rows,
    resolve_polymarket_slug_for_fixture,
)
from src.ratings import get_team_ratings  # noqa: E402
from src.storage import load_csv  # noqa: E402
from src.utils import today_iso  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Polymarket sports slug resolution and market price joining.")
    parser.add_argument("--home", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--fixture-date", required=True)
    parser.add_argument("--competition", default="World Cup")
    parser.add_argument("--slug", default="")
    args = parser.parse_args()

    candidates = build_polymarket_sports_slug_candidates(args.home, args.away, args.fixture_date, args.competition)
    resolution = resolve_polymarket_slug_for_fixture(
        args.home,
        args.away,
        args.fixture_date,
        args.competition,
        user_supplied_slug_or_url=args.slug or None,
    )
    resolved_slug = str(resolution.get("resolved_slug", "") or args.slug or "")
    event_markets = load_polymarket_event_markets_by_slug(resolved_slug) if resolved_slug else pd.DataFrame()
    model_markets = _model_markets_for_fixture(args.home, args.away, args.fixture_date, args.competition)
    joined = join_polymarket_prices_to_model_markets(model_markets, event_markets, args.home, args.away)
    alpha = polymarket_alpha_rows(joined)
    csv_path, md_path = _write_reports(args, candidates, resolution, event_markets, model_markets, joined, alpha)

    joined_with_price = joined.loc[joined["market_price_cents"].notna()] if not joined.empty else pd.DataFrame()
    print("Polymarket slug and join audit")
    print("slug candidates:")
    for candidate in candidates:
        print(f"- {candidate}")
    print(f"resolved slug: {resolution.get('resolved_slug', '')}")
    print(f"resolution confidence: {resolution.get('confidence', '')}")
    print(f"markets loaded: {len(event_markets)}")
    print(f"market types found: {', '.join(_market_types(event_markets)) or 'None'}")
    print(f"model markets generated: {len(model_markets)}")
    print(f"joined markets count: {len(joined_with_price)}")
    print(f"moneyline joined: {_joined_count(joined, '1X2')}")
    print(f"totals joined: {_joined_count(joined, 'Total')}")
    print(f"spreads joined: {_joined_count(joined, 'Handicap')}")
    print(f"BTTS joined: {_joined_count(joined, 'BTTS')}")
    print(f"Polymarket Alpha rows: {len(alpha)}")
    if alpha.empty:
        print(f"reason if no rows: {_reason_no_rows(resolution, event_markets, joined)}")
    print(f"report CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"report: {md_path.relative_to(PROJECT_ROOT)}")


def _model_markets_for_fixture(home: str, away: str, fixture_date: str, competition: str) -> pd.DataFrame:
    fixtures = load_csv(
        "fixtures.csv",
        ["match_id", "date_utc", "time_utc", "competition", "group", "home", "away", "venue", "city", "country"],
    )
    fixture = _find_fixture(fixtures, home, away, fixture_date)
    teams = get_team_ratings()
    venues = load_csv("venues.csv", ["venue"])
    result = run_match_model(fixture, teams, venues, pd.DataFrame(), ModelConfig())
    rows = []
    for (market, selection), probability in market_probability_map(result["home"], result["away"], result["probs"]).items():
        rows.append(
            {
                "market": market,
                "selection": selection,
                "model_prob": probability,
                "fair_odds": 1.0 / probability if probability > 0 else float("inf"),
            }
        )
    return pd.DataFrame(rows)


def _find_fixture(fixtures: pd.DataFrame, home: str, away: str, fixture_date: str) -> pd.Series:
    if not fixtures.empty:
        home_l = home.lower()
        away_l = away.lower()
        dates = pd.to_datetime(fixtures["date_utc"], errors="coerce").dt.date.astype(str)
        mask = (
            (fixtures["home"].astype(str).str.lower() == home_l)
            & (fixtures["away"].astype(str).str.lower() == away_l)
            & (dates == fixture_date)
        )
        if mask.any():
            return fixtures.loc[mask].iloc[0]
        reverse = (
            (fixtures["home"].astype(str).str.lower() == away_l)
            & (fixtures["away"].astype(str).str.lower() == home_l)
            & (dates == fixture_date)
        )
        if reverse.any():
            return fixtures.loc[reverse].iloc[0]
    return pd.Series(
        {
            "match_id": f"audit_{fixture_date}_{home}_{away}".lower().replace(" ", "_"),
            "date_utc": fixture_date,
            "time_utc": "00:00",
            "competition": "FIFA World Cup",
            "group": "",
            "home": home,
            "away": away,
            "venue": "",
            "city": "",
            "country": "",
        }
    )


def _write_reports(
    args: argparse.Namespace,
    candidates: list[str],
    resolution: dict,
    event_markets: pd.DataFrame,
    model_markets: pd.DataFrame,
    joined: pd.DataFrame,
    alpha: pd.DataFrame,
) -> tuple[Path, Path]:
    reports = PROJECT_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    today = today_iso()
    csv_path = reports / f"polymarket_slug_join_{today}.csv"
    md_path = reports / f"polymarket_slug_join_{today}.md"
    joined.to_csv(csv_path, index=False)

    summary = pd.DataFrame(
        [
            {
                "fixture": f"{args.home} vs {args.away}",
                "fixture_date": args.fixture_date,
                "input_slug": args.slug,
                "resolved_slug": resolution.get("resolved_slug", ""),
                "resolution_status": resolution.get("resolution_status", ""),
                "confidence": resolution.get("confidence", ""),
                "markets_loaded": len(event_markets),
                "market_types_found": ", ".join(_market_types(event_markets)),
                "model_markets_generated": len(model_markets),
                "joined_markets_count": int(joined["market_price_cents"].notna().sum()) if not joined.empty else 0,
                "moneyline_joined": _joined_count(joined, "1X2"),
                "totals_joined": _joined_count(joined, "Total"),
                "spreads_joined": _joined_count(joined, "Handicap"),
                "btts_joined": _joined_count(joined, "BTTS"),
                "polymarket_alpha_rows": len(alpha),
                "reason_no_rows": _reason_no_rows(resolution, event_markets, joined) if alpha.empty else "",
            }
        ]
    )
    md_path.write_text(
        "\n".join(
            [
                f"# Polymarket Slug Join Audit - {today}",
                "",
                "This read-only audit resolves a sports event slug and joins discovered prices to model market rows. It does not trade, size, or change the football prediction formula.",
                "",
                "## Summary",
                "",
                _to_markdown(summary),
                "",
                "## Slug Candidates",
                "",
                _to_markdown(pd.DataFrame({"slug_candidate": candidates})),
                "",
                "## Resolution",
                "",
                _to_markdown(pd.DataFrame([resolution])),
                "",
                "## Joined Markets",
                "",
                _to_markdown(joined.head(50)),
                "",
                "## Polymarket Alpha Rows",
                "",
                _to_markdown(alpha.head(50)),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return csv_path, md_path


def _market_types(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty or "market_type" not in df.columns:
        return []
    return sorted(df["market_type"].dropna().astype(str).unique().tolist())


def _joined_count(joined: pd.DataFrame, market: str) -> int:
    if joined is None or joined.empty:
        return 0
    rows = joined.loc[(joined["market"].astype(str) == market) & joined["market_price_cents"].notna()]
    return int(len(rows))


def _reason_no_rows(resolution: dict, event_markets: pd.DataFrame, joined: pd.DataFrame) -> str:
    if resolution.get("resolution_status") != "resolved":
        return "No Polymarket event resolved for this fixture."
    if event_markets is None or event_markets.empty:
        return "Polymarket event resolved, but no nested market prices were loaded."
    if joined is None or joined.empty or not joined["market_price_cents"].notna().any():
        return "Polymarket event resolved, but no matching market prices were found."
    return ""


def _to_markdown(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_No rows._"
    display = df.fillna("").astype(str)
    headers = list(display.columns)
    rows = display.values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_escape_md_cell(value) for value in row) + " |")
    return "\n".join(lines)


def _escape_md_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


if __name__ == "__main__":
    main()
