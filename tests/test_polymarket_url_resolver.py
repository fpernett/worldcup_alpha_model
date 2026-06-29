from __future__ import annotations

import sys

import pandas as pd

from src.polymarket_match_search import build_polymarket_match_queries
from src.polymarket_url_resolver import (
    classify_polymarket_market_type,
    parse_polymarket_url_or_slug,
    resolve_polymarket_event_markets_from_url_or_slug,
    score_polymarket_event_slug_for_fixture,
)
from src.team_names import normalize_polymarket_team_code


def _synthetic_event() -> dict:
    return {
        "id": "EV-COL-CDR",
        "slug": "fifwc-col-cdr-2026-06-23",
        "title": "Colombia vs. DR Congo",
        "sport": "Soccer",
        "startDate": "2026-04-06T22:30:18Z",
        "endDate": "2026-06-24T02:00:00Z",
        "active": True,
        "closed": False,
        "markets": [
            {
                "id": "ML-COL",
                "slug": "fifwc-col-cdr-2026-06-23-col",
                "question": "Will Colombia win on 2026-06-23?",
                "groupItemTitle": "Colombia",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.65", "0.35"]',
                "liquidityNum": "1000",
                "volumeNum": "500",
            },
            {
                "id": "SPREAD",
                "slug": "fifwc-col-cdr-2026-06-23-spread-colombia--1.5",
                "question": "Will Colombia cover the -1.5 spread?",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.4", "0.6"]',
            },
            {
                "id": "TOTAL",
                "slug": "fifwc-col-cdr-2026-06-23-total-2.5",
                "question": "Will total goals be over 2.5?",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.5", "0.5"]',
            },
        ],
    }


def test_parse_polymarket_sports_url() -> None:
    parsed = parse_polymarket_url_or_slug("https://polymarket.com/sports/world-cup/fifwc-col-cdr-2026-06-23")

    assert parsed["slug"] == "fifwc-col-cdr-2026-06-23"
    assert parsed["team_code_1"] == "COL"
    assert parsed["team_code_2"] == "CDR"
    assert parsed["date_hint"] == "2026-06-23"
    assert parsed["competition_hint"] == "world-cup"
    assert parsed["is_sports_url"] is True


def test_cdr_alias_maps_to_dr_congo() -> None:
    assert normalize_polymarket_team_code("CDR") == "DR Congo"
    assert normalize_polymarket_team_code("COD") == "DR Congo"


def test_slug_fixture_scoring_is_high_for_colombia_dr_congo() -> None:
    score = score_polymarket_event_slug_for_fixture(
        "fifwc-col-cdr-2026-06-23",
        "Colombia",
        "DR Congo",
        fixture_date="2026-06-23",
    )

    assert score["confidence"] == "high"
    assert score["matched_home"] is True
    assert score["matched_away"] is True
    assert score["matched_date"] is True
    assert score["rejected"] is False


def test_slug_fixture_scoring_accepts_prior_local_date_for_late_utc_fixture() -> None:
    score = score_polymarket_event_slug_for_fixture(
        "fifwc-nld-mar-2026-06-29",
        "Netherlands",
        "Morocco",
        fixture_date="2026-06-30",
    )

    assert score["confidence"] == "high"
    assert score["matched_home"] is True
    assert score["matched_away"] is True
    assert score["matched_date"] is True
    assert score["rejected"] is False


def test_slug_fixture_scoring_accepts_mexico_code_and_prior_local_date() -> None:
    score = score_polymarket_event_slug_for_fixture(
        "fifwc-mex-ecu-2026-06-30",
        "Mexico",
        "Ecuador",
        fixture_date="2026-07-01",
    )

    assert score["confidence"] == "high"
    assert score["matched_home"] is True
    assert score["matched_away"] is True
    assert score["matched_date"] is True
    assert score["rejected"] is False


def test_query_generation_includes_polymarket_slug_codes() -> None:
    queries = build_polymarket_match_queries("Colombia", "DR Congo", "World Cup")

    assert "COL CDR" in queries
    assert "COL COD" in queries
    assert "fifwc-col-cdr" in queries
    assert "fifwc-col-cod" in queries


def test_market_extraction_from_synthetic_event(monkeypatch) -> None:
    import src.polymarket_url_resolver as resolver

    monkeypatch.setattr(
        resolver,
        "fetch_polymarket_event_by_slug_with_diagnostics",
        lambda *args, **kwargs: (_synthetic_event(), {"method": "synthetic", "warnings": [], "methods_tried": ["synthetic"]}),
    )

    markets, diagnostics = resolve_polymarket_event_markets_from_url_or_slug("fifwc-col-cdr-2026-06-23")

    assert len(markets) == 3
    assert set(markets["market_type"]) == {"moneyline", "spread", "total"}
    assert diagnostics["markets_extracted"] == 3


def test_html_fetch_without_exact_event_payload_does_not_resolve(monkeypatch) -> None:
    import src.polymarket_url_resolver as resolver

    monkeypatch.setattr(resolver, "_fetch_event_by_slug_param", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(resolver, "_fetch_event_by_slug_path", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(resolver, "_fetch_event_by_exact_search", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(resolver, "_fetch_event_from_cache", lambda *_args, **_kwargs: None)

    class FakeResponse:
        text = "<html><head><title>World Cup Odds &amp; Prediction Markets 2026 | Polymarket</title></head></html>"

        def raise_for_status(self) -> None:
            return None

    monkeypatch.setattr(resolver.requests, "get", lambda *_args, **_kwargs: FakeResponse())

    event, diagnostics = resolver.fetch_polymarket_event_by_slug_with_diagnostics("fifwc-ned-mar-2026-06-30")

    assert event is None
    assert "HTML page loaded, but no exact event payload matched the slug." in diagnostics["warnings"]


def test_wrong_fixture_is_rejected_or_low_confidence() -> None:
    score = score_polymarket_event_slug_for_fixture(
        "fifwc-col-cdr-2026-06-23",
        "Colombia",
        "Brazil",
        fixture_date="2026-06-23",
    )

    assert score["confidence"] == "reject"
    assert score["rejected"] is True
    assert "away mismatch" in score["warning"]


def test_market_type_classifier() -> None:
    assert classify_polymarket_market_type({"question": "Both Teams to Score?"}) == "btts"
    assert classify_polymarket_market_type({"question": "Exact Score 1-0"}) == "exact_score"
    assert classify_polymarket_market_type({"question": "Will total goals be under 2.5?"}) == "total"


def test_url_resolver_audit_script_runs(monkeypatch, tmp_path, capsys) -> None:
    import scripts.audit_polymarket_url_resolver as script

    markets = pd.DataFrame(
        [
            {
                "event_id": "EV",
                "event_slug": "fifwc-col-cdr-2026-06-23",
                "event_title": "Colombia vs. DR Congo",
                "event_category": "Soccer",
                "event_start_date": "2026-04-06",
                "event_end_date": "2026-06-24",
                "market_id": "ML-COL",
                "market_slug": "fifwc-col-cdr-2026-06-23-col",
                "question": "Will Colombia win on 2026-06-23?",
                "market_title": "",
                "market_type": "moneyline",
                "outcomes": '["Yes", "No"]',
                "yes_price": 0.65,
                "no_price": 0.35,
                "liquidity": 1000,
                "volume": 500,
                "source_url": "https://polymarket.com/sports/world-cup/fifwc-col-cdr-2026-06-23",
                "source": "API",
                "last_updated": "2026-06-22T00:00:00Z",
                "extraction_method": "synthetic",
            }
        ]
    )
    diagnostics = {"event_title": "Colombia vs. DR Congo", "warning": "", "fetch": {"method": "synthetic"}}
    monkeypatch.setattr(script, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(script, "resolve_polymarket_event_markets_from_url_or_slug", lambda *_args, **_kwargs: (markets, diagnostics))
    monkeypatch.setattr(script, "today_iso", lambda: "2026-06-22")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_polymarket_url_resolver.py",
            "--url",
            "https://polymarket.com/sports/world-cup/fifwc-col-cdr-2026-06-23",
            "--home",
            "Colombia",
            "--away",
            "DR Congo",
            "--fixture-date",
            "2026-06-23",
        ],
    )

    script.main()

    output = capsys.readouterr().out
    assert "parsed slug: fifwc-col-cdr-2026-06-23" in output
    assert "candidate moneyline markets: 1" in output
    assert (tmp_path / "reports" / "polymarket_url_resolver_2026-06-22.csv").exists()
