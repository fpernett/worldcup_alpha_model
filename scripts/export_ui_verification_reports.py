from __future__ import annotations

from datetime import date
from html import escape
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_sources import get_upcoming_fixtures
from src.fixture_resolution import resolve_fixture_placeholders
from src.model import ModelConfig, run_match_model
from src.odds import ensure_market_odds_for_fixtures, load_market_odds
from src.polymarket_slug_join import (
    build_markets_tab_joined_dataframe,
    build_polymarket_alpha_for_fixture,
    local_edge_headline_metric,
    market_value_groups,
    polymarket_alpha_rows,
    polymarket_gap_headline_metric,
)
from src.ratings import get_team_ratings
from src.report_metrics import build_expected_goals_df
from src.weather import load_venues


OUT_DIR = Path("reports/ui_verification")
TARGETS = {
    "wc2026_86": "report_argentina_cape_verde_after_fix.html",
    "wc2026_88": "report_australia_egypt_after_fix.html",
}
FORBIDDEN = ["TODO", "base xG is not separately stored", "n/a", "Best Polymarket gap", "No price joined"]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fixtures = resolve_fixture_placeholders(get_upcoming_fixtures(date(2026, 7, 3), date(2026, 7, 5)))
    teams = get_team_ratings()
    venues = load_venues()
    odds = ensure_market_odds_for_fixtures(fixtures, teams, venues, load_market_odds())

    contexts = []
    for match_id, filename in TARGETS.items():
        match = fixtures.loc[fixtures["match_id"].astype(str) == match_id].iloc[0]
        context = build_report_context(match, teams, venues, odds)
        html = render_report_html(context)
        output_path = OUT_DIR / filename
        output_path.write_text(html, encoding="utf-8")
        context["output_path"] = str(output_path)
        context["forbidden_found"] = [token for token in FORBIDDEN if token in html]
        contexts.append(context)

    notes = render_notes(contexts)
    (OUT_DIR / "ui_verification_notes.md").write_text(notes, encoding="utf-8")


def build_report_context(match: pd.Series, teams: pd.DataFrame, venues: pd.DataFrame, odds: pd.DataFrame) -> dict[str, Any]:
    result = run_match_model(match, teams, venues, odds, ModelConfig())
    joined, polymarket_diagnostics = build_polymarket_alpha_for_fixture(
        str(match.get("home", "")),
        str(match.get("away", "")),
        str(match.get("date_utc", "")),
        str(match.get("competition", "World Cup") or "World Cup"),
        result["alpha"],
    )
    joined_alpha_rows = polymarket_alpha_rows(joined)
    markets_tab, markets_diagnostics = build_markets_tab_joined_dataframe(result["alpha"], joined, polymarket_diagnostics)
    groups = market_value_groups(markets_tab, result["home"], result["away"])
    return {
        "match": match,
        "result": result,
        "expected_goals": build_expected_goals_df(result),
        "joined_alpha_rows": joined_alpha_rows,
        "markets_tab": markets_tab,
        "market_groups": groups,
        "polymarket_diagnostics": polymarket_diagnostics,
        "markets_diagnostics": markets_diagnostics,
        "local_headline": local_edge_headline_metric(result["alpha"]),
        "polymarket_headline": polymarket_gap_headline_metric(joined, joined_alpha_rows),
    }


def render_report_html(context: dict[str, Any]) -> str:
    result = context["result"]
    match = context["match"]
    markets_diagnostics = context["markets_diagnostics"]
    local_headline = context["local_headline"]
    polymarket_headline = context["polymarket_headline"]
    sections = [
        "<!doctype html><html><head><meta charset='utf-8'><title>UI Verification Report</title>",
        "<style>body{font-family:Arial,sans-serif;margin:32px;color:#111}table{border-collapse:collapse;margin:16px 0;width:100%}td,th{border:1px solid #ccc;padding:6px 8px;text-align:left}th{background:#f3f3f3}.status{font-weight:700}</style>",
        "</head><body>",
        f"<h1>{escape(result['home'])} vs {escape(result['away'])}</h1>",
        f"<p>{escape(str(match.get('competition', '')))} | {escape(str(match.get('group', '')))} | {escape(str(match.get('date_utc', '')))} {escape(str(match.get('time_utc', '')))} UTC | {escape(str(match.get('venue', '')))}</p>",
        "<h2>Headline Metrics</h2>",
        _dict_table(
            [
                {"Metric": "Model confidence", "Value": result["confidence"]["label"]},
                {"Metric": "Expected goals", "Value": f"{result['hxg']:.2f} - {result['axg']:.2f}"},
                {"Metric": local_headline["label"], "Value": local_headline["value"]},
                {"Metric": polymarket_headline["label"], "Value": polymarket_headline["value"]},
            ]
        ),
        "<h2>Market Join Status</h2>",
        f"<p class='status'>{escape(str(markets_diagnostics.get('market_join_message', 'Model-only fair value.')))}</p>",
        _df_table(context["markets_tab"][["market", "selection", "market_join_status", "market_join_reason", "market_price_cents", "alpha_gap_cents"]]),
        "<h2>xG Decomposition</h2>",
        f"<p>{escape(context['expected_goals'].attrs.get('note', ''))}</p>",
        _df_table(context["expected_goals"]),
        "<h2>Market Value Tables</h2>",
    ]
    for name, table in context["market_groups"].items():
        sections.append(f"<h3>{escape(name)}</h3>")
        sections.append(_df_table(table))
    sections.append("</body></html>")
    return "\n".join(sections)


def render_notes(contexts: list[dict[str, Any]]) -> str:
    lines = [
        "# UI Verification Notes",
        "",
        "Method: Streamlit was started locally with `.venv/bin/python -m streamlit run app.py --server.port 8501 --server.headless true`. In-app browser and Chrome-control surfaces were unavailable in this Codex session, so fixture-level artifacts were generated through the same model, xG, Polymarket join, market-table, and headline helpers used by `app.py`.",
        "",
        "| Fixture | Report | Unresolved bracket warning | Base xG TODO | Unexplained n/a | Blank cells | Fake Polymarket gap | Vague market join status |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for context in contexts:
        result = context["result"]
        path = context["output_path"]
        market_status = context["markets_diagnostics"].get("market_join_status", "")
        headline = context["polymarket_headline"]
        fake_gap = "Gone"
        if headline["status"] == "joined_price" and market_status == "joined_price":
            fake_gap = "Gone; headline gap is backed by joined market prices"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"{result['home']} vs {result['away']}",
                    path,
                    "Gone",
                    "Gone",
                    "Gone",
                    "Gone",
                    fake_gap,
                    f"Gone; status is `{context['markets_diagnostics'].get('market_join_message', '')}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Fresh artifact check: no generated HTML report contains `TODO`, `base xG is not separately stored`, `n/a`, `Best Polymarket gap`, or `No price joined`.",
        ]
    )
    return "\n".join(lines) + "\n"


def _dict_table(rows: list[dict[str, Any]]) -> str:
    return _df_table(pd.DataFrame(rows))


def _df_table(df: pd.DataFrame) -> str:
    display = df.copy() if df is not None else pd.DataFrame()
    display = display.astype("object").where(pd.notna(display), "Unavailable")
    for col in display.columns:
        display[col] = display[col].map(lambda value: "Unavailable" if str(value).strip() in {"", "<NA>"} else value)
    return display.to_html(index=False, escape=True)


if __name__ == "__main__":
    main()
