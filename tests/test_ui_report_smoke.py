from __future__ import annotations

from datetime import date

import pandas as pd

from src.data_sources import get_upcoming_fixtures
from src.fixture_resolution import resolve_fixture_placeholders
from src.model import ModelConfig, run_match_model
from src.odds import ensure_market_odds_for_fixtures, load_market_odds
from src.polymarket_slug_join import (
    build_markets_tab_joined_dataframe,
    local_edge_headline_metric,
    market_value_groups,
    polymarket_gap_headline_metric,
)
from src.ratings import get_team_ratings
from src.report_metrics import build_expected_goals_df
from src.weather import load_venues


FORBIDDEN_SUBSCRIBER_STRINGS = [
    "TODO",
    "base xG is not separately stored",
    "n/a",
    "Best Polymarket gap",
    "No price joined",
    "pending prior results or manual mapping",
]


def test_july_3_subscriber_reports_are_clean_for_model_only_polymarket_path() -> None:
    fixtures = resolve_fixture_placeholders(get_upcoming_fixtures(date(2026, 7, 3), date(2026, 7, 5)))
    teams = get_team_ratings()
    venues = load_venues()
    odds = ensure_market_odds_for_fixtures(fixtures, teams, venues, load_market_odds())

    for match_id in ["wc2026_86", "wc2026_88"]:
        match = fixtures.loc[fixtures["match_id"].astype(str) == match_id].iloc[0]
        result = run_match_model(match, teams, venues, odds, ModelConfig())
        expected_goals = build_expected_goals_df(result)
        markets_tab, diagnostics = build_markets_tab_joined_dataframe(
            result["alpha"],
            pd.DataFrame(),
            {"slug_resolution_status": "unresolved", "event_markets_loaded_count": 0, "joined_markets_count": 0},
        )
        market_groups = market_value_groups(markets_tab, result["home"], result["away"])
        local_headline = local_edge_headline_metric(result["alpha"])
        polymarket_headline = polymarket_gap_headline_metric(pd.DataFrame(), pd.DataFrame())

        report_text = _subscriber_text(result, expected_goals, market_groups, local_headline, polymarket_headline, diagnostics)

        for forbidden in FORBIDDEN_SUBSCRIBER_STRINGS:
            assert forbidden not in report_text
        assert "No Polymarket event resolved." in report_text
        assert "Polymarket price status: No joined market price" in report_text
        assert polymarket_headline["label"] != "Best Polymarket gap"
        assert diagnostics["market_join_status"] == "no_event_resolved"
        assert result["home"] not in {"Winner Match 86", "Winner Group K"}
        _assert_no_empty_display_cells(expected_goals)
        for group in market_groups.values():
            _assert_no_empty_display_cells(group)


def _subscriber_text(
    result: dict,
    expected_goals: pd.DataFrame,
    market_groups: dict[str, pd.DataFrame],
    local_headline: dict[str, str],
    polymarket_headline: dict[str, str],
    diagnostics: dict,
) -> str:
    parts = [
        f"{result['home']} vs {result['away']}",
        f"Expected goals: {result['hxg']:.2f} - {result['axg']:.2f}",
        f"{local_headline['label']}: {local_headline['value']}",
        f"{polymarket_headline['label']}: {polymarket_headline['value']}",
        str(diagnostics.get("market_join_message", "")),
        expected_goals.attrs.get("note", ""),
        expected_goals.to_string(index=False),
    ]
    for name, group in market_groups.items():
        parts.append(name)
        parts.append(group.to_string(index=False))
    return "\n".join(parts)


def _assert_no_empty_display_cells(df: pd.DataFrame) -> None:
    display = df.astype("object").where(pd.notna(df), "Unavailable")
    for value in display.to_numpy().ravel().tolist():
        assert str(value).strip() != ""
