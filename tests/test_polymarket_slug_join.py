from __future__ import annotations

import pandas as pd

from src.polymarket_slug_join import (
    build_polymarket_sports_slug_candidates,
    get_polymarket_team_code_variants,
    join_polymarket_prices_to_model_markets,
    load_polymarket_event_markets_by_slug,
    polymarket_alpha_rows,
    resolve_polymarket_slug_for_fixture,
)


def _model_markets() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"market": "1X2", "selection": "Home", "model_prob": 0.42, "fair_odds": 2.38},
            {"market": "1X2", "selection": "Draw", "model_prob": 0.28, "fair_odds": 3.57},
            {"market": "1X2", "selection": "Away", "model_prob": 0.30, "fair_odds": 3.33},
            {"market": "Total", "selection": "Over 2.5", "model_prob": 0.55, "fair_odds": 1.82},
            {"market": "Total", "selection": "Under 2.5", "model_prob": 0.45, "fair_odds": 2.22},
            {"market": "BTTS", "selection": "Yes", "model_prob": 0.51, "fair_odds": 1.96},
            {"market": "BTTS", "selection": "No", "model_prob": 0.49, "fair_odds": 2.04},
        ]
    )


def _event_markets() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_slug": "fifwc-ury-esp-2026-06-26",
                "event_title": "Uruguay vs. Spain",
                "market_id": "ML-URY",
                "market_slug": "fifwc-ury-esp-2026-06-26-ury",
                "question": "Will Uruguay win on 2026-06-26?",
                "market_title": "",
                "market_type": "moneyline",
                "outcomes": '["Yes","No"]',
                "outcome_name": "Uruguay",
                "price_cents": 35.0,
                "odds_decimal": 2.857,
                "liquidity": 1000,
                "volume": 2000,
                "source": "test",
                "last_updated": "2026-06-26T00:00:00Z",
            },
            {
                "event_slug": "fifwc-ury-esp-2026-06-26",
                "event_title": "Uruguay vs. Spain",
                "market_id": "ML-DRAW",
                "market_slug": "fifwc-ury-esp-2026-06-26-draw",
                "question": "Will Uruguay vs. Spain end in a draw?",
                "market_title": "",
                "market_type": "moneyline",
                "outcomes": '["Yes","No"]',
                "outcome_name": "Draw",
                "price_cents": 27.0,
                "odds_decimal": 3.704,
                "liquidity": 1000,
                "volume": 2000,
                "source": "test",
                "last_updated": "2026-06-26T00:00:00Z",
            },
            {
                "event_slug": "fifwc-ury-esp-2026-06-26",
                "event_title": "Uruguay vs. Spain",
                "market_id": "ML-ESP",
                "market_slug": "fifwc-ury-esp-2026-06-26-esp",
                "question": "Will Spain win on 2026-06-26?",
                "market_title": "",
                "market_type": "moneyline",
                "outcomes": '["Yes","No"]',
                "outcome_name": "Spain",
                "price_cents": 38.0,
                "odds_decimal": 2.632,
                "liquidity": 1000,
                "volume": 2000,
                "source": "test",
                "last_updated": "2026-06-26T00:00:00Z",
            },
            {
                "event_slug": "fifwc-ury-esp-2026-06-26",
                "event_title": "Uruguay vs. Spain",
                "market_id": "TOT-OVER",
                "market_slug": "fifwc-ury-esp-2026-06-26-over-2-5",
                "question": "Will total goals be over 2.5?",
                "market_title": "",
                "market_type": "total",
                "outcomes": '["Yes","No"]',
                "outcome_name": "Over 2.5",
                "price_cents": 48.0,
                "odds_decimal": 2.083,
                "liquidity": 1000,
                "volume": 2000,
                "source": "test",
                "last_updated": "2026-06-26T00:00:00Z",
            },
            {
                "event_slug": "fifwc-ury-esp-2026-06-26",
                "event_title": "Uruguay vs. Spain",
                "market_id": "TOT-UNDER",
                "market_slug": "fifwc-ury-esp-2026-06-26-under-2-5",
                "question": "Will total goals be under 2.5?",
                "market_title": "",
                "market_type": "total",
                "outcomes": '["Yes","No"]',
                "outcome_name": "Under 2.5",
                "price_cents": 52.0,
                "odds_decimal": 1.923,
                "liquidity": 1000,
                "volume": 2000,
                "source": "test",
                "last_updated": "2026-06-26T00:00:00Z",
            },
        ]
    )


def test_uruguay_spain_slug_generation_includes_both_orders() -> None:
    candidates = build_polymarket_sports_slug_candidates("Uruguay", "Spain", "2026-06-26")

    assert "fifwc-ury-esp-2026-06-26" in candidates
    assert "fifwc-esp-ury-2026-06-26" in candidates


def test_cape_verde_saudi_slug_generation_prefers_cvi() -> None:
    candidates = build_polymarket_sports_slug_candidates("Cape Verde", "Saudi Arabia", "2026-06-26")

    assert "fifwc-cvi-ksa-2026-06-26" in candidates


def test_dr_congo_slug_generation_includes_cdr_and_cod() -> None:
    codes = get_polymarket_team_code_variants("DR Congo")
    candidates = build_polymarket_sports_slug_candidates("Colombia", "DR Congo", "2026-06-23")

    assert "CDR" in codes
    assert "COD" in codes
    assert "fifwc-col-cdr-2026-06-23" in candidates
    assert "fifwc-col-cod-2026-06-23" in candidates


def test_user_supplied_slug_is_parsed_and_validated() -> None:
    resolved = resolve_polymarket_slug_for_fixture(
        "Uruguay",
        "Spain",
        "2026-06-26",
        user_supplied_slug_or_url="https://polymarket.com/sports/world-cup/fifwc-ury-esp-2026-06-26",
    )

    assert resolved["resolved_slug"] == "fifwc-ury-esp-2026-06-26"
    assert resolved["resolution_status"] == "resolved"
    assert resolved["confidence"] == "high"


def test_reversed_slug_order_still_validates() -> None:
    resolved = resolve_polymarket_slug_for_fixture(
        "Uruguay",
        "Spain",
        "2026-06-26",
        user_supplied_slug_or_url="fifwc-esp-ury-2026-06-26",
    )

    assert resolved["resolution_status"] == "resolved"
    assert resolved["matched_home"] is True
    assert resolved["matched_away"] is True
    assert resolved["team_order"] == "away-home"


def test_moneyline_markets_join_home_draw_away() -> None:
    joined = join_polymarket_prices_to_model_markets(_model_markets().head(3), _event_markets(), "Uruguay", "Spain")

    assert set(joined["selection"]) == {"Home", "Draw", "Away"}
    assert joined["market_price_cents"].notna().all()
    assert set(joined["polymarket_market_id"]) == {"ML-URY", "ML-DRAW", "ML-ESP"}


def test_totals_markets_join_over_under_2_5() -> None:
    model = _model_markets().loc[_model_markets()["market"] == "Total"]
    joined = join_polymarket_prices_to_model_markets(model, _event_markets(), "Uruguay", "Spain")

    assert set(joined["selection"]) == {"Over 2.5", "Under 2.5"}
    assert joined["market_price_cents"].notna().all()


def test_missing_price_gives_no_signal_with_diagnostic_reason() -> None:
    model = pd.DataFrame([{"market": "BTTS", "selection": "Yes", "model_prob": 0.51, "fair_odds": 1.96}])
    joined = join_polymarket_prices_to_model_markets(model, _event_markets(), "Uruguay", "Spain")

    assert pd.isna(joined.iloc[0]["market_price_cents"])
    assert joined.iloc[0]["signal"] == "No signal"
    assert "No btts market found" in joined.iloc[0]["mapping_reason"]


def test_available_price_populates_value_columns() -> None:
    joined = join_polymarket_prices_to_model_markets(_model_markets().head(1), _event_markets(), "Uruguay", "Spain")
    row = joined.iloc[0]

    assert row["market_price_cents"] == 35.0
    assert round(float(row["market_odds_decimal"]), 2) == 2.86
    assert row["alpha_gap_cents"] == 7.0
    assert row["score"] == 7.0
    assert row["signal"] == "Positive model gap"


def test_polymarket_alpha_table_not_empty_when_markets_match() -> None:
    joined = join_polymarket_prices_to_model_markets(_model_markets(), _event_markets(), "Uruguay", "Spain")
    alpha_rows = polymarket_alpha_rows(joined)

    assert not alpha_rows.empty
    assert "market_price_cents" in alpha_rows.columns


def test_event_market_loader_extracts_outcome_rows(monkeypatch) -> None:
    event = {
        "id": "EV-URY-ESP",
        "slug": "fifwc-ury-esp-2026-06-26",
        "title": "Uruguay vs. Spain",
        "markets": [
            {
                "id": "ML-URY",
                "slug": "fifwc-ury-esp-2026-06-26-ury",
                "question": "Will Uruguay win on 2026-06-26?",
                "groupItemTitle": "Uruguay",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.35", "0.65"]',
            },
            {
                "id": "TOT-OVER",
                "slug": "fifwc-ury-esp-2026-06-26-over-2-5",
                "question": "Will total goals be over 2.5?",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.48", "0.52"]',
            },
        ],
    }

    monkeypatch.setattr(
        "src.polymarket_slug_join.fetch_polymarket_event_by_slug_with_diagnostics",
        lambda *_args, **_kwargs: (event, {"method": "synthetic", "warnings": []}),
    )

    markets = load_polymarket_event_markets_by_slug("fifwc-ury-esp-2026-06-26")

    assert set(markets["market_type"]) == {"moneyline", "total"}
    assert "Uruguay" in set(markets["outcome_name"])
    assert "Over 2.5" in set(markets["outcome_name"])
