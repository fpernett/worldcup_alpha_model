from __future__ import annotations

import pandas as pd
import requests

from src.alpha import calculate_polymarket_alpha, dashboard_market_selection
from src.backtesting import load_prediction_log, save_prediction_snapshot
from src.market_mapping import explain_unmapped_polymarket_markets, map_match_to_polymarket_markets
from src.polymarket import (
    POLYMARKET_COLUMNS,
    _match_event_slug_candidates,
    get_match_polymarket_markets,
    get_polymarket_markets,
    normalize_price,
)
from src.sensitivity import run_sensitivity_analysis
from src.model import ModelConfig


def sample_model_result() -> dict:
    return {
        "match_id": "M1",
        "home": "England",
        "away": "Croatia",
        "hxg": 1.4,
        "axg": 1.0,
        "probs": {
            "home_win": 0.491,
            "draw": 0.287,
            "away_win": 0.222,
            "over_2_5": 0.46,
            "under_2_5": 0.54,
            "btts_yes": 0.50,
            "btts_no": 0.50,
        },
    }


def sample_mapping(polymarket_side: str = "YES") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "M1",
                "home": "England",
                "away": "Croatia",
                "market_id": "PM1",
                "question": "Will England beat Croatia?",
                "market_type": "match_winner_home",
                "model_side": "home_win",
                "polymarket_side": polymarket_side,
                "mapping_confidence": "high",
                "mapping_reason": "test",
                "manual_confirmed": True,
                "yes_price": 0.42,
                "no_price": 0.58,
                "liquidity": 500,
                "volume": 1000,
                "closed": 0,
            }
        ]
    )


def sample_match() -> pd.Series:
    return pd.Series(
        {
            "match_id": "M1",
            "date_utc": "2026-06-17",
            "time_utc": "20:00",
            "home": "England",
            "away": "Croatia",
            "venue": "Dallas Stadium",
        }
    )


def sample_teams() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"team": "England", "elo": 1980, "attack": 0.82, "defense": 0.80, "recent_form": 0.78, "training_temp_c": 15, "training_humidity_pct": 70},
            {"team": "Croatia", "elo": 1855, "attack": 0.70, "defense": 0.74, "recent_form": 0.67, "training_temp_c": 18, "training_humidity_pct": 65},
        ]
    )


def sample_venues() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"venue": "Dallas Stadium", "city": "Arlington", "country": "USA", "latitude": 32.7473, "longitude": -97.0945, "altitude_m": 190, "temp_c": 22, "humidity_pct": 45, "wind_kmh": 0, "precipitation_mm": 0, "roof_expected_closed": 1, "neutral_site": 1}
        ]
    )


def test_polymarket_price_normalization() -> None:
    assert normalize_price(42) == 0.42
    assert normalize_price(0.42) == 0.42
    assert pd.isna(normalize_price(142))


def test_yes_no_mapping_probability() -> None:
    yes_alpha = calculate_polymarket_alpha(sample_model_result(), sample_mapping("YES"))
    no_alpha = calculate_polymarket_alpha(sample_model_result(), sample_mapping("NO"))
    assert abs(float(yes_alpha.iloc[0]["model_probability"]) - 0.491) < 1e-9
    assert abs(float(no_alpha.iloc[0]["model_probability"]) - 0.509) < 1e-9


def test_alpha_gap_calculation() -> None:
    alpha = calculate_polymarket_alpha(sample_model_result(), sample_mapping("YES"))
    assert round(float(alpha.iloc[0]["fair_price_cents"]), 1) == 49.1
    assert round(float(alpha.iloc[0]["polymarket_price_cents"]), 1) == 42.0
    assert round(float(alpha.iloc[0]["alpha_gap_cents"]), 1) == 7.1


def test_alpha_signal_gating_requires_calibration_support_for_strong_edge() -> None:
    result = sample_model_result()
    result["confidence"] = {"label": "High"}
    alpha = calculate_polymarket_alpha(result, sample_mapping("YES"))

    assert alpha.iloc[0]["signal_strength"] == "Watchlist"
    assert "calibrated probability unavailable" in alpha.iloc[0]["signal_policy_reasons"]


def test_alpha_signal_gating_allows_strong_edge_only_when_supported() -> None:
    result = sample_model_result()
    result["probs"]["home_win"] = 0.54
    result["confidence"] = {"label": "High"}
    result["calibrated_probability_available"] = True
    result["historical_support"] = True
    alpha = calculate_polymarket_alpha(result, sample_mapping("YES"))

    assert alpha.iloc[0]["signal_strength"] == "Strong edge"


def test_polymarket_alpha_rows_include_dashboard_market_selection() -> None:
    alpha = calculate_polymarket_alpha(sample_model_result(), sample_mapping("YES"))

    assert alpha.iloc[0]["market"] == "1X2"
    assert alpha.iloc[0]["selection"] == "England"


def test_dashboard_market_selection_maps_totals_and_btts() -> None:
    assert dashboard_market_selection(pd.Series({"market_type": "over_2_5", "model_side": "over_2_5"})) == ("Total", "Over 2.5")
    assert dashboard_market_selection(pd.Series({"market_type": "btts_no", "model_side": "btts_no"})) == ("BTTS", "No")


def test_final_winner_market_maps_to_extra_time_and_penalty_probability() -> None:
    final_match = pd.Series(
        {
            "match_id": "wc2026_104",
            "date_utc": "2026-07-19",
            "time_utc": "19:00",
            "competition": "FIFA World Cup",
            "group": "Final",
            "home": "Argentina",
            "away": "Spain",
            "venue": "MetLife Stadium",
        }
    )
    markets = pd.DataFrame(
        [
            {
                "market_id": "PM-FINAL-ARG",
                "question": "Will Argentina beat Spain in the World Cup Final?",
                "slug": "argentina-spain-final-argentina",
                "event_title": "World Cup Final: Argentina vs Spain",
                "category": "Sports",
                "start_date": "2026-07-19",
                "end_date": "2026-07-20",
                "active": True,
                "closed": False,
                "outcomes": '["Yes", "No"]',
                "yes_price": 0.48,
                "no_price": 0.52,
                "liquidity": 1000,
                "volume": 5000,
                "source": "test",
                "last_updated": "2026-07-16T00:00:00Z",
            }
        ],
        columns=POLYMARKET_COLUMNS,
    )

    mapped = map_match_to_polymarket_markets(final_match, markets)

    assert len(mapped) == 1
    assert mapped.iloc[0]["market_type"] == "decisive_winner_home"
    assert mapped.iloc[0]["model_side"] == "home_advance"

    result = sample_model_result()
    result.update({"match_id": "wc2026_104", "home": "Argentina", "away": "Spain"})
    result["probs"]["home_advance"] = 0.57
    result["probs"]["away_advance"] = 0.43
    alpha = calculate_polymarket_alpha(result, mapped)

    assert float(alpha.iloc[0]["model_probability"]) == 0.57
    assert alpha.iloc[0]["market"] == "Winner (incl. ET/pens)"
    assert alpha.iloc[0]["selection"] == "Argentina"


def test_same_winner_wording_remains_regulation_time_outside_decisive_fixtures() -> None:
    markets = pd.DataFrame(
        [
            {
                "market_id": "PM1",
                "question": "Will England beat Croatia?",
                "slug": "england-croatia-england",
                "event_title": "England vs Croatia World Cup match",
                "category": "Sports",
                "yes_price": 0.42,
                "no_price": 0.58,
                "liquidity": 500,
                "volume": 1000,
                "closed": False,
            }
        ],
        columns=POLYMARKET_COLUMNS,
    )
    group_match = sample_match().copy()
    group_match["competition"] = "FIFA World Cup"
    group_match["group"] = "Group A"

    mapped = map_match_to_polymarket_markets(group_match, markets)

    assert mapped.iloc[0]["model_side"] == "home_win"
    assert mapped.iloc[0]["market_type"] == "match_winner_home"


def test_sensitivity_output_shape() -> None:
    out = run_sensitivity_analysis(sample_match(), sample_teams(), sample_venues(), pd.DataFrame(), {"Default": ModelConfig()})
    assert len(out) == 1
    for col in ["scenario", "home_win", "draw", "away_win", "over_2_5", "under_2_5", "btts_yes", "btts_no", "home_xg", "away_xg"]:
        assert col in out.columns


def test_prediction_log_append(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("src.storage.DATA_DIR", tmp_path)
    alpha = calculate_polymarket_alpha(sample_model_result(), sample_mapping("YES"))
    save_prediction_snapshot(sample_match(), sample_model_result(), alpha)
    log = load_prediction_log()
    assert len(log) == 1
    assert log.iloc[0]["market_id"] == "PM1"


def test_gamma_api_default_used_when_env_empty(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("POLYMARKET_API_URL", raising=False)
    monkeypatch.delenv("POLYMARKET_GAMMA_API_URL", raising=False)
    monkeypatch.delenv("POLYMARKET_CLOB_API_URL", raising=False)
    monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr("src.polymarket.DATA_DIR", tmp_path)

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> list[dict]:
            return [
                {
                    "id": "PM-MEX",
                    "question": "Will Mexico win the 2026 FIFA World Cup?",
                    "slug": "will-mexico-win-the-2026-fifa-world-cup",
                    "events": [{"title": "World Cup Winner", "category": "Sports"}],
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.035", "0.965"]',
                    "liquidityNum": "1000",
                    "volumeNum": "5000",
                    "active": True,
                    "closed": False,
                    "updatedAt": "2026-06-18T12:00:00Z",
                }
            ]

    calls = []

    def fake_get(url, params=None, timeout=0):
        calls.append((url, params, timeout))
        return FakeResponse()

    monkeypatch.setattr("src.polymarket.requests.get", fake_get)

    markets = get_polymarket_markets(query="World Cup", force_refresh=True)

    assert calls
    assert calls[0][0] == "https://gamma-api.polymarket.com/markets"
    assert markets.attrs.get("source_label") == "API"
    assert markets.iloc[0]["market_id"] == "PM-MEX"
    assert float(markets.iloc[0]["yes_price"]) == 0.035
    assert markets.iloc[0]["event_title"] == "World Cup Winner"


def test_multi_word_polymarket_query_matches_tokens(monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> list[dict]:
            return [
                {
                    "id": "PM-ENG-CRO",
                    "question": "Will England beat Croatia?",
                    "slug": "will-england-beat-croatia",
                    "events": [{"title": "England vs Croatia", "category": "Sports"}],
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.42", "0.58"]',
                    "liquidityNum": "1000",
                    "volumeNum": "5000",
                    "active": True,
                    "closed": False,
                }
            ]

    def fake_get(url, params=None, timeout=0):
        return FakeResponse()

    monkeypatch.setattr("src.polymarket.requests.get", fake_get)

    markets = get_polymarket_markets(query="England Croatia", force_refresh=True, write_cache=False)

    assert len(markets) == 1
    assert markets.iloc[0]["market_id"] == "PM-ENG-CRO"


def test_selected_match_search_finds_match_market_and_mapper_rejects_outrights(monkeypatch) -> None:
    match_market = {
        "id": "PM-ENG-CRO",
        "question": "Will England beat Croatia?",
        "slug": "will-england-beat-croatia",
        "events": [{"title": "England vs Croatia", "category": "Sports"}],
        "outcomes": '["Yes", "No"]',
        "outcomePrices": '["0.42", "0.58"]',
        "liquidityNum": "1000",
        "volumeNum": "5000",
        "active": True,
        "closed": False,
    }
    outright_market = {
        "id": "PM-ENG-OUTRIGHT",
        "question": "Will England win the 2026 FIFA World Cup?",
        "slug": "will-england-win-the-2026-fifa-world-cup",
        "events": [{"title": "World Cup Winner", "category": "Sports"}],
        "outcomes": '["Yes", "No"]',
        "outcomePrices": '["0.12", "0.88"]',
        "liquidityNum": "1000",
        "volumeNum": "5000",
        "active": True,
        "closed": False,
    }

    class FakeResponse:
        def __init__(self, records: list[dict]) -> None:
            self.records = records

        def raise_for_status(self) -> None:
            return None

        def json(self) -> list[dict]:
            return self.records

    def fake_get(url, params=None, timeout=0):
        search = (params or {}).get("search")
        if search in {"England Croatia", "England vs Croatia", "World Cup England Croatia"}:
            return FakeResponse([match_market, outright_market])
        if search == "Will England beat Croatia":
            return FakeResponse([match_market])
        return FakeResponse([])

    monkeypatch.setattr("src.polymarket.requests.get", fake_get)

    markets = get_match_polymarket_markets("England", "Croatia")
    mapped = map_match_to_polymarket_markets(sample_match(), markets)

    assert set(markets["market_id"]) == {"PM-ENG-CRO"}
    assert len(mapped) == 1
    assert mapped.iloc[0]["market_id"] == "PM-ENG-CRO"
    assert mapped.iloc[0]["model_side"] == "home_win"


def test_selected_match_search_loads_gamma_event_markets(monkeypatch) -> None:
    event_payload = [
        {
            "id": "351741",
            "slug": "fifwc-can-qat-2026-06-18",
            "title": "Canada vs. Qatar",
            "sport": "soccer",
            "endDate": "2026-06-18T22:00:00Z",
            "markets": [
                {
                    "id": "1897112",
                    "question": "Will Canada win on 2026-06-18?",
                    "slug": "fifwc-can-qat-2026-06-18-can",
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.765", "0.235"]',
                    "liquidityNum": "3021430.7208",
                    "volumeNum": "4282896.4475",
                    "active": True,
                    "closed": False,
                },
                {
                    "id": "1897113",
                    "question": "Will Canada vs. Qatar end in a draw?",
                    "slug": "fifwc-can-qat-2026-06-18-draw",
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.155", "0.845"]',
                    "liquidityNum": "3304100.4944",
                    "volumeNum": "791226.6480",
                    "active": True,
                    "closed": False,
                },
                {
                    "id": "1897114",
                    "question": "Will Qatar win on 2026-06-18?",
                    "slug": "fifwc-can-qat-2026-06-18-qat",
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.075", "0.925"]',
                    "liquidityNum": "3238977.6558",
                    "volumeNum": "1240425.3224",
                    "active": True,
                    "closed": False,
                },
            ],
        }
    ]

    class FakeResponse:
        def __init__(self, records):
            self.records = records

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return self.records

    calls = []

    def fake_get(url, params=None, timeout=0):
        calls.append((url, params or {}))
        if url == "https://gamma-api.polymarket.com/events":
            return FakeResponse(event_payload if (params or {}).get("slug") == "fifwc-can-qat-2026-06-18" else [])
        return FakeResponse([])

    monkeypatch.setattr("src.polymarket.requests.get", fake_get)

    match = pd.Series(
        {
            "match_id": "CAN-QAT",
            "date_utc": "2026-06-18",
            "time_utc": "22:00",
            "home": "Canada",
            "away": "Qatar",
        }
    )
    markets = get_match_polymarket_markets("Canada", "Qatar", "2026-06-18")
    mapped = map_match_to_polymarket_markets(match, markets)

    assert set(markets["market_id"]) == {"1897112", "1897113", "1897114"}
    assert set(mapped["model_side"]) == {"home_win", "draw", "away_win"}
    assert mapped.loc[mapped["model_side"] == "home_win", "yes_price"].iloc[0] == 0.765
    assert not any(url.endswith("/markets") for url, _params in calls)


def test_mexico_korea_event_slug_uses_polymarket_team_code_and_local_date(monkeypatch) -> None:
    candidates = _match_event_slug_candidates("Mexico", "South Korea", "2026-06-19")
    assert "fifwc-mex-kr-2026-06-18" in candidates

    event_payload = [
        {
            "id": "351742",
            "slug": "fifwc-mex-kr-2026-06-18",
            "title": "Mexico vs. Korea Republic",
            "sport": "soccer",
            "endDate": "2026-06-19T01:00:00Z",
            "markets": [
                {
                    "id": "1897115",
                    "question": "Will Mexico win on 2026-06-18?",
                    "slug": "fifwc-mex-kr-2026-06-18-mex",
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.475", "0.525"]',
                    "liquidityNum": "3000000",
                    "volumeNum": "4000000",
                    "active": True,
                    "closed": False,
                },
                {
                    "id": "1897116",
                    "question": "Will Mexico vs. Korea Republic end in a draw?",
                    "slug": "fifwc-mex-kr-2026-06-18-draw",
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.295", "0.705"]',
                    "liquidityNum": "3000000",
                    "volumeNum": "4000000",
                    "active": True,
                    "closed": False,
                },
                {
                    "id": "1897117",
                    "question": "Will Korea Republic win on 2026-06-18?",
                    "slug": "fifwc-mex-kr-2026-06-18-kr",
                    "outcomes": '["Yes", "No"]',
                    "outcomePrices": '["0.235", "0.765"]',
                    "liquidityNum": "3000000",
                    "volumeNum": "4000000",
                    "active": True,
                    "closed": False,
                },
            ],
        }
    ]

    class FakeResponse:
        def __init__(self, records):
            self.records = records

        def raise_for_status(self) -> None:
            return None

        def json(self):
            return self.records

    def fake_get(url, params=None, timeout=0):
        if url == "https://gamma-api.polymarket.com/events":
            return FakeResponse(event_payload if (params or {}).get("slug") == "fifwc-mex-kr-2026-06-18" else [])
        return FakeResponse([])

    monkeypatch.setattr("src.polymarket.requests.get", fake_get)

    match = pd.Series(
        {
            "match_id": "MEX-KOR",
            "date_utc": "2026-06-19",
            "time_utc": "01:00",
            "home": "Mexico",
            "away": "South Korea",
        }
    )
    markets = get_match_polymarket_markets("Mexico", "South Korea", "2026-06-19")
    mapped = map_match_to_polymarket_markets(match, markets)

    assert set(markets["market_id"]) == {"1897115", "1897116", "1897117"}
    assert set(mapped["model_side"]) == {"home_win", "draw", "away_win"}
    assert mapped.loc[mapped["model_side"] == "away_win", "yes_price"].iloc[0] == 0.235


def test_mapper_recognizes_usa_alias_for_united_states() -> None:
    match = pd.Series(
        {
            "match_id": "USA-AUS",
            "date_utc": "2026-06-19",
            "time_utc": "19:00",
            "home": "United States",
            "away": "Australia",
        }
    )
    markets = pd.DataFrame(
        [
            {
                "market_id": "PM-USA-AUS",
                "question": "Will USA beat Australia?",
                "slug": "fifwc-usa-aus-2026-06-19-usa",
                "event_title": "USA vs Australia",
                "category": "Sports",
                "yes_price": 0.55,
                "no_price": 0.45,
                "liquidity": 1000,
                "volume": 5000,
                "closed": 0,
            }
        ]
    )

    mapped = map_match_to_polymarket_markets(match, markets)

    assert len(mapped) == 1
    assert mapped.iloc[0]["model_side"] == "home_win"
    assert mapped.iloc[0]["mapping_confidence"] == "high"


def test_missing_polymarket_api_falls_back_to_csv(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("POLYMARKET_API_URL", raising=False)
    monkeypatch.delenv("POLYMARKET_GAMMA_API_URL", raising=False)
    monkeypatch.delenv("POLYMARKET_CLOB_API_URL", raising=False)
    monkeypatch.setattr("src.storage.DATA_DIR", tmp_path)
    monkeypatch.setattr("src.polymarket.DATA_DIR", tmp_path)
    monkeypatch.setattr("src.cache.CACHE_DIR", tmp_path / "cache")
    pd.DataFrame(columns=POLYMARKET_COLUMNS).to_csv(tmp_path / "polymarket_markets.csv", index=False)

    def fake_get(*args, **kwargs):
        raise requests.RequestException("offline")

    monkeypatch.setattr("src.polymarket.requests.get", fake_get)

    markets = get_polymarket_markets()
    assert list(markets.columns) == POLYMARKET_COLUMNS
    assert markets.attrs.get("source_label") == "local CSV"


def test_world_cup_outright_is_not_mapped_to_selected_match() -> None:
    match = pd.Series(
        {
            "match_id": "MEX-KOR",
            "date_utc": "2026-06-19",
            "time_utc": "01:00",
            "home": "Mexico",
            "away": "South Korea",
        }
    )
    markets = pd.DataFrame(
        [
            {
                "market_id": "PM-MEX",
                "question": "Will Mexico win the 2026 FIFA World Cup?",
                "slug": "will-mexico-win-the-2026-fifa-world-cup",
                "event_title": "World Cup Winner",
                "category": "Sports",
                "start_date": "2025-07-02T22:28:24Z",
                "end_date": "2026-07-20T00:00:00Z",
                "active": 1,
                "closed": 0,
                "outcomes": '["Yes", "No"]',
                "yes_price": 0.035,
                "no_price": 0.965,
                "liquidity": 1000,
                "volume": 5000,
                "source": "API",
                "last_updated": "2026-06-18T12:00:00Z",
            }
        ]
    )

    mapped = map_match_to_polymarket_markets(match, markets)

    assert mapped.empty


def test_unmapped_diagnostics_explain_tournament_outrights() -> None:
    match = pd.Series(
        {
            "match_id": "CAN-QAT",
            "date_utc": "2026-06-18",
            "time_utc": "22:00",
            "home": "Canada",
            "away": "Qatar",
        }
    )
    markets = pd.DataFrame(
        [
            {
                "market_id": "PM-CAN",
                "question": "Will Canada win the 2026 FIFA World Cup?",
                "slug": "will-canada-win-the-2026-fifa-world-cup",
                "event_title": "World Cup Winner",
                "category": "Sports",
                "yes_price": 0.0025,
                "no_price": 0.9975,
                "liquidity": 1000,
                "volume": 5000,
            },
            {
                "market_id": "PM-QAT",
                "question": "Will Qatar win the 2026 FIFA World Cup?",
                "slug": "will-qatar-win-the-2026-fifa-world-cup",
                "event_title": "World Cup Winner",
                "category": "Sports",
                "yes_price": 0.0005,
                "no_price": 0.9995,
                "liquidity": 1000,
                "volume": 5000,
            },
        ]
    )

    diagnostics = explain_unmapped_polymarket_markets(match, markets)

    assert len(diagnostics) == 2
    assert diagnostics["rejection_reason"].str.contains("Tournament outright").all()
