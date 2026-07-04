from __future__ import annotations

import sys

import pandas as pd

from src.polymarket_match_search import find_best_polymarket_match_candidates, score_polymarket_markets_for_match
from src.polymarket_sports_discovery import (
    GAMMA_EVENT_MARKET_COLUMNS,
    discover_world_cup_identifiers,
    fetch_all_gamma_events,
    flatten_gamma_events_to_markets,
    flattened_markets_to_polymarket_rows,
)


def _match_event() -> dict:
    return {
        "id": "EV1",
        "slug": "colombia-dr-congo",
        "title": "Colombia vs DR Congo",
        "sport": "soccer",
        "active": True,
        "closed": False,
        "endDate": "2026-06-23T20:00:00Z",
        "markets": [
            {
                "id": "M1",
                "question": "Will Colombia beat DR Congo?",
                "slug": "will-colombia-beat-dr-congo",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.55", "0.45"]',
                "liquidityNum": "1000",
                "volumeNum": "5000",
                "active": True,
                "closed": False,
            }
        ],
    }


def test_event_with_nested_market_flattens_and_scores_high() -> None:
    flat = flatten_gamma_events_to_markets([_match_event()])
    rows = flattened_markets_to_polymarket_rows(flat)
    scored = score_polymarket_markets_for_match(rows, "Colombia", "DR Congo", fixture_date="2026-06-23", competition="World Cup")

    assert len(flat) == 1
    assert list(flat.columns) == GAMMA_EVENT_MARKET_COLUMNS
    assert scored.iloc[0]["confidence"] == "high"
    assert bool(scored.iloc[0]["rejected"]) is False


def test_query_empty_but_cache_contains_match(monkeypatch, tmp_path) -> None:
    from src import polymarket_sports_discovery as discovery

    cache_path = tmp_path / "polymarket_markets_cache.csv"
    monkeypatch.setattr(discovery, "MARKETS_CACHE_PATH", cache_path)
    flatten_gamma_events_to_markets([_match_event()]).to_csv(cache_path, index=False)

    candidates = find_best_polymarket_match_candidates(
        "Colombia",
        "DR Congo",
        fixture_date="2026-06-23",
        competition="World Cup",
        retrieval_mode="cache",
        min_confidence="medium",
    )

    assert len(candidates) == 1
    assert candidates.iloc[0]["market_id"] == "M1"
    assert candidates.iloc[0]["retrieval_layer"] == "cache"


def test_political_event_in_broad_all_events_is_rejected(monkeypatch) -> None:
    from src import polymarket_sports_discovery as discovery

    political_event = {
        "id": "EVPOL",
        "slug": "colombia-election",
        "title": "Colombia election",
        "category": "Politics",
        "markets": [
            {
                "id": "POL",
                "question": "Will the next president of Colombia be from the left?",
                "slug": "colombia-president-left",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.5", "0.5"]',
            }
        ],
    }

    def fake_discover(*args, **kwargs):
        markets = flatten_gamma_events_to_markets([political_event])
        return markets, {"events_fetched": 1, "markets_flattened": 1, "cache_paths": {}}

    monkeypatch.setattr(discovery, "discover_polymarket_sports_event_markets", fake_discover)

    candidates = find_best_polymarket_match_candidates(
        "Colombia",
        "DR Congo",
        retrieval_mode="all_events",
        include_rejected=True,
        min_confidence="low",
    )

    assert len(candidates) == 1
    assert bool(candidates.iloc[0]["rejected"]) is True
    assert "political/non-sports" in candidates.iloc[0]["reject_reason"]


def test_world_cup_tag_discovery() -> None:
    identifiers = discover_world_cup_identifiers(
        tags=[{"id": "123", "label": "FIFA World Cup"}],
        series=[{"id": "456", "title": "Politics"}],
        sports=[
            {
                "id": "174",
                "sport": "fifwc",
                "series": "11433",
                "resolution": "https://www.fifa.com/fifaplus/en/tournaments/mens/worldcup",
            }
        ],
    )

    assert identifiers["tag_ids"] == ["123"]
    assert identifiers["series_ids"] == ["11433"]
    assert identifiers["sports_ids"] == []


def test_fetch_all_gamma_events_paginates(monkeypatch) -> None:
    calls = []

    class FakeResponse:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return self.payload

    def fake_get(url, params=None, timeout=0):
        calls.append(params or {})
        offset = (params or {}).get("offset", 0)
        if offset == 0:
            return FakeResponse({"events": [{"id": "E1"}], "has_more": True})
        return FakeResponse({"events": [{"id": "E2"}], "has_more": False})

    monkeypatch.setattr("src.polymarket_sports_discovery.requests.get", fake_get)

    events = fetch_all_gamma_events(max_pages=5, limit=1)

    assert [event["id"] for event in events] == ["E1", "E2"]
    assert len(calls) == 2


def test_no_candidate_has_clear_diagnostic(monkeypatch) -> None:
    from src import polymarket_sports_discovery as discovery
    from src import polymarket as polymarket_source

    def fake_discover(*args, **kwargs):
        return pd.DataFrame(columns=GAMMA_EVENT_MARKET_COLUMNS), {"events_fetched": 0, "markets_flattened": 0, "cache_paths": {}}

    monkeypatch.setattr(discovery, "discover_polymarket_sports_event_markets", fake_discover)
    monkeypatch.setattr(discovery, "load_polymarket_markets_cache", lambda: pd.DataFrame(columns=GAMMA_EVENT_MARKET_COLUMNS))
    monkeypatch.setattr(polymarket_source, "get_polymarket_markets", lambda *args, **kwargs: pd.DataFrame())

    candidates = find_best_polymarket_match_candidates("Colombia", "DR Congo", retrieval_mode="auto")

    assert candidates.empty
    assert "No Polymarket match market was discovered" in candidates.attrs["no_match_reason"]


def test_sports_discovery_audit_script_runs(monkeypatch, tmp_path, capsys) -> None:
    import scripts.audit_polymarket_sports_discovery as script

    candidates = pd.DataFrame(
        [
            {
                "market_id": "M1",
                "question": "Will Colombia beat DR Congo?",
                "slug": "will-colombia-beat-dr-congo",
                "event_title": "Colombia vs DR Congo",
                "category": "Sports",
                "score": 100,
                "confidence": "high",
                "matched_home": True,
                "matched_away": True,
                "matched_competition": True,
                "matched_sports_terms": True,
                "rejected": False,
                "reject_reason": "",
                "score_breakdown": "both teams matched",
                "retrieval_layer": "sports_events",
                "search_query": "",
            }
        ]
    )
    candidates.attrs["retrieval_layers_tried"] = ["sports_events"]
    candidates.attrs["events_fetched"] = 1
    candidates.attrs["markets_flattened"] = 1
    candidates.attrs["markets_scored"] = 1
    candidates.attrs["cache_paths"] = {}

    monkeypatch.setattr(script, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(script, "find_best_polymarket_match_candidates", lambda *args, **kwargs: candidates)
    monkeypatch.setattr(
        script,
        "polymarket_discovery_cache_status",
        lambda: {
            "events_cache_path": "data/polymarket_events_cache.csv",
            "events_cache_exists": True,
            "events_cache_rows": 1,
            "events_cache_last_updated": "2026-06-22T00:00:00Z",
            "markets_cache_path": "data/polymarket_markets_cache.csv",
            "markets_cache_exists": True,
            "markets_cache_rows": 1,
            "markets_cache_last_updated": "2026-06-22T00:00:00Z",
        },
    )
    monkeypatch.setattr(script, "today_iso", lambda: "2026-06-22")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_polymarket_sports_discovery.py",
            "--home",
            "Colombia",
            "--away",
            "DR Congo",
            "--competition",
            "World Cup",
        ],
    )

    script.main()

    output = capsys.readouterr().out
    assert "accepted candidates: 1" in output
    assert (tmp_path / "reports" / "polymarket_sports_discovery_2026-06-22.csv").exists()
