from __future__ import annotations

import pandas as pd
from pathlib import Path

from src.polymarket_slug_join import (
    build_markets_tab_joined_dataframe,
    build_polymarket_alpha_for_fixture,
    build_polymarket_sports_slug_candidates,
    extract_markets_from_polymarket_event_html,
    get_polymarket_team_code_variants,
    joined_market_groups,
    join_polymarket_prices_to_model_markets,
    load_polymarket_event_markets_by_slug,
    market_value_groups,
    markets_tab_export_dataframe,
    polymarket_alpha_rows,
    polymarket_gap_headline_metric,
    resolve_polymarket_slug_for_fixture,
    top_alpha_empty_state_message,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
    assert "fifwc-cdr-col-2026-06-23" in candidates
    assert "fifwc-col-cod-2026-06-23" in candidates
    assert "fifwc-cod-col-2026-06-23" in candidates


def test_portugal_code_variants_and_colombia_portugal_candidates() -> None:
    codes = get_polymarket_team_code_variants("Portugal")
    candidates = build_polymarket_sports_slug_candidates("Colombia", "Portugal", "2026-06-27")

    assert "PRT" in codes
    assert "POR" in codes
    assert "fifwc-col-prt-2026-06-27" in candidates
    assert "fifwc-prt-col-2026-06-27" in candidates
    assert "fifwc-col-por-2026-06-27" in candidates
    assert "fifwc-por-col-2026-06-27" in candidates


def test_portugal_croatia_slug_generation_includes_polymarket_hrv_code() -> None:
    croatia_codes = get_polymarket_team_code_variants("Croatia")
    candidates = build_polymarket_sports_slug_candidates("Portugal", "Croatia", "2026-07-02")

    assert "CRO" in croatia_codes
    assert "HRV" in croatia_codes
    assert "fifwc-prt-hrv-2026-07-02" in candidates
    assert "fifwc-hrv-prt-2026-07-02" in candidates


def test_netherlands_morocco_slug_generation_includes_fifa_codes_and_prior_local_date() -> None:
    netherlands_codes = get_polymarket_team_code_variants("Netherlands")
    morocco_codes = get_polymarket_team_code_variants("Morocco")
    candidates = build_polymarket_sports_slug_candidates("Netherlands", "Morocco", "2026-06-30")

    assert "NLD" in netherlands_codes
    assert "NED" in netherlands_codes
    assert "MAR" in morocco_codes
    assert "fifwc-nld-mar-2026-06-30" in candidates
    assert "fifwc-mar-nld-2026-06-30" in candidates
    assert "fifwc-nld-mar-2026-06-29" in candidates
    assert "fifwc-mar-nld-2026-06-29" in candidates
    assert "fifwc-ned-mar-2026-06-30" in candidates
    assert "fifwc-mar-ned-2026-06-30" in candidates
    assert "fifwc-ned-mar-2026-06-29" in candidates
    assert "fifwc-mar-ned-2026-06-29" in candidates


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


def test_portugal_croatia_user_supplied_hrv_slug_validates() -> None:
    resolved = resolve_polymarket_slug_for_fixture(
        "Portugal",
        "Croatia",
        "2026-07-02",
        user_supplied_slug_or_url="fifwc-prt-hrv-2026-07-02",
    )

    assert resolved["resolution_status"] == "resolved"
    assert resolved["resolved_slug"] == "fifwc-prt-hrv-2026-07-02"
    assert resolved["matched_home"] is True
    assert resolved["matched_away"] is True
    assert resolved["matched_date"] is True
    assert resolved["team_order"] == "home-away"


def test_moneyline_markets_join_home_draw_away() -> None:
    joined = join_polymarket_prices_to_model_markets(_model_markets().head(3), _event_markets(), "Uruguay", "Spain")

    assert set(joined["selection"]) == {"Home", "Draw", "Away"}
    assert joined["market_price_cents"].notna().all()
    assert set(joined["polymarket_market_id"]) == {"ML-URY", "ML-DRAW", "ML-ESP"}


def test_moneyline_join_normalizes_cabo_verde_to_cape_verde() -> None:
    model = pd.DataFrame(
        [
            {"market": "1X2", "selection": "Away", "model_prob": 0.12, "fair_odds": 8.33},
        ]
    )
    event_markets = pd.DataFrame(
        [
            {
                "event_slug": "fifwc-arg-cvi-2026-07-03",
                "event_title": "Argentina vs. Cabo Verde",
                "market_id": "ML-CVI",
                "market_slug": "fifwc-arg-cvi-2026-07-03-cvi",
                "question": "Will Cabo Verde win on 2026-07-03?",
                "market_title": "",
                "market_type": "moneyline",
                "outcomes": '["Yes","No"]',
                "outcome_name": "Cabo Verde",
                "price_cents": 9.0,
                "odds_decimal": 11.11,
                "liquidity": 1000,
                "volume": 2000,
                "source": "test",
                "last_updated": "2026-07-03T00:00:00Z",
            },
        ]
    )

    joined = join_polymarket_prices_to_model_markets(model, event_markets, "Argentina", "Cape Verde")

    assert joined.iloc[0]["polymarket_market_id"] == "ML-CVI"
    assert joined.iloc[0]["market_price_cents"] == 9.0
    assert joined.iloc[0]["mapping_confidence"] == "high"


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


def test_polymarket_gap_headline_requires_fresh_joined_price() -> None:
    joined = join_polymarket_prices_to_model_markets(_model_markets().head(1), _event_markets(), "Uruguay", "Spain")

    headline = polymarket_gap_headline_metric(joined, now_utc="2026-06-26T01:00:00Z")
    stale = polymarket_gap_headline_metric(joined, now_utc="2026-07-03T01:00:00Z")
    no_join = polymarket_gap_headline_metric(pd.DataFrame(), pd.DataFrame())

    assert headline == {"label": "Polymarket gap", "value": "7.0c", "status": "joined_price"}
    assert stale == {"label": "Polymarket price status", "value": "No joined market price", "status": "model_only"}
    assert no_join == {"label": "Polymarket price status", "value": "No joined market price", "status": "model_only"}


def test_markets_tab_model_only_keeps_fair_odds_without_prices() -> None:
    table, diagnostics = build_markets_tab_joined_dataframe(
        _model_markets(),
        pd.DataFrame(),
        {"slug_resolution_status": "unresolved", "event_markets_loaded_count": 0, "joined_markets_count": 0},
    )

    assert table["fair_odds"].notna().all()
    assert table["market_odds"].isna().all()
    assert table["signal"].eq("Model-only fair value").all()
    assert table["market_join_status"].eq("no_event_resolved").all()
    assert diagnostics["market_join_status"] == "no_event_resolved"
    assert diagnostics["reason_if_zero"] == "No Polymarket event resolved."


def test_market_value_groups_populate_model_rows_without_polymarket_event() -> None:
    table, _diagnostics = build_markets_tab_joined_dataframe(
        _model_markets(),
        pd.DataFrame(),
        {"slug_resolution_status": "unresolved", "event_markets_loaded_count": 0, "joined_markets_count": 0},
    )

    groups = market_value_groups(table, "Uruguay", "Spain")

    assert all(not groups[name].empty for name in ["High Scoring", "Low Scoring", "Result Home", "Result Away", "Other Markets"])
    assert groups["High Scoring"]["Fair odds / fair price"].astype(str).str.strip().any()
    assert groups["Result Home"]["Market"].str.contains("1X2: Home", regex=False).any()
    assert groups["Result Away"]["Market"].str.contains("1X2: Away", regex=False).any()
    assert groups["Other Markets"]["Market"].str.contains("1X2: Draw", regex=False).any()
    assert groups["Result Home"]["Odds / Price"].eq("No market price").all()
    assert groups["Result Home"]["Signal"].eq("Model-only fair value").all()


def test_market_value_groups_render_safe_missing_values_without_polymarket_event() -> None:
    table, _diagnostics = build_markets_tab_joined_dataframe(
        _model_markets().head(1),
        pd.DataFrame(),
        {"slug_resolution_status": "unresolved", "event_markets_loaded_count": 0, "joined_markets_count": 0},
    )
    table.loc[:, "signal"] = pd.NA
    table.loc[:, "mapping_confidence"] = pd.NA
    table["score"] = table["score"].astype("object")
    table.loc[:, "score"] = pd.NA

    groups = market_value_groups(table, "Uruguay", "Spain")
    row = groups["Result Home"].iloc[0]

    assert row["Signal"] == "Model-only fair value"
    assert row["Mapping confidence"] == "Not mapped"
    assert row["Score"] == 0.0
    assert "<NA>" not in row.astype(str).to_string()
    assert "" not in row.astype(str).tolist()


def test_market_value_groups_bucket_actual_team_names_by_fixture_side() -> None:
    model = pd.DataFrame(
        [
            {"market": "1X2", "selection": "Uruguay", "model_prob": 0.42, "fair_odds": 2.38},
            {"market": "1X2", "selection": "Draw", "model_prob": 0.28, "fair_odds": 3.57},
            {"market": "1X2", "selection": "Spain", "model_prob": 0.30, "fair_odds": 3.33},
            {"market": "Double Chance", "selection": "Uruguay/Draw", "model_prob": 0.70, "fair_odds": 1.43},
            {"market": "Double Chance", "selection": "Draw/Spain", "model_prob": 0.58, "fair_odds": 1.72},
        ]
    )
    table, _diagnostics = build_markets_tab_joined_dataframe(
        model,
        pd.DataFrame(),
        {"slug_resolution_status": "unresolved", "event_markets_loaded_count": 0, "joined_markets_count": 0},
    )

    groups = market_value_groups(table, "Uruguay", "Spain")

    assert set(groups["Result Home"]["Market"]) == {"1X2: Uruguay", "Double Chance: Uruguay/Draw"}
    assert set(groups["Result Away"]["Market"]) == {"1X2: Spain", "Double Chance: Draw/Spain"}
    assert set(groups["Other Markets"]["Market"]) == {"1X2: Draw"}


def test_markets_tab_joined_moneyline_populates_market_odds_alpha_ev_source_and_timestamp() -> None:
    joined = join_polymarket_prices_to_model_markets(_model_markets().head(3), _event_markets(), "Uruguay", "Spain")
    table, diagnostics = build_markets_tab_joined_dataframe(
        _model_markets().head(3),
        joined,
        {"slug_resolution_status": "resolved", "resolved_slug": "fifwc-ury-esp-2026-06-26", "event_markets_loaded_count": 5, "joined_markets_count": 3},
    )

    home = table.loc[(table["market"] == "1X2") & (table["selection"] == "Home")].iloc[0]
    assert round(float(home["market_odds"]), 2) == 2.86
    assert round(float(home["alpha_ev"]), 4) == round(0.42 * (100.0 / 35.0) - 1.0, 4)
    assert home["odds_source"] == "polymarket"
    assert home["odds_last_updated"] == "2026-06-26T00:00:00Z"
    assert diagnostics["rows_with_market_odds"] == 3
    assert diagnostics["rows_with_alpha_ev"] == 3


def test_markets_tab_uses_mapped_polymarket_alpha_when_event_join_is_empty() -> None:
    model = pd.DataFrame([{"market": "1X2", "selection": "Uruguay", "model_prob": 0.42, "fair_odds": 2.38}])
    mapped_alpha = pd.DataFrame(
        [
            {
                "market_id": "PM-URY",
                "question": "Will Uruguay beat Spain?",
                "market": "1X2",
                "selection": "Uruguay",
                "model_probability": 0.42,
                "primary_model_probability": 0.42,
                "fair_price_cents": 42.0,
                "polymarket_price_cents": 35.0,
                "alpha_gap_cents": 7.0,
                "alpha_ev": 0.20,
                "signal_strength": "Moderate",
                "mapping_confidence": "high",
            }
        ]
    )

    table, diagnostics = build_markets_tab_joined_dataframe(
        model,
        pd.DataFrame(),
        {"slug_resolution_status": "unresolved", "event_markets_loaded_count": 0, "joined_markets_count": 0},
        mapped_polymarket_alpha_df=mapped_alpha,
    )
    row = table.iloc[0]

    assert round(float(row["market_odds"]), 2) == 2.86
    assert round(float(row["alpha_ev"]), 2) == 0.20
    assert row["odds_source"] == "polymarket"
    assert row["polymarket_market_id"] == "PM-URY"
    assert diagnostics["mapped_alpha_rows"] == 1
    assert diagnostics["rows_with_market_odds"] == 1
    assert diagnostics["reason_if_zero"] == ""


def test_top_alpha_empty_state_reports_filters_when_mapped_rows_are_filtered_out() -> None:
    mapped_alpha = pd.DataFrame(
        [
            {
                "market_id": "PM-URY",
                "market": "1X2",
                "selection": "Uruguay",
                "polymarket_price_cents": 35.0,
                "alpha_gap_cents": 2.0,
            }
        ]
    )

    message = top_alpha_empty_state_message(
        pd.DataFrame(),
        pd.DataFrame(),
        mapped_alpha,
        {"reason_no_alpha_rows": "No Polymarket event resolved."},
        min_liquidity=100.0,
        min_alpha_gap=5.0,
    )

    assert message == "No top alpha signals passed the current sidebar filters (liquidity >= 100, alpha gap >= 5c)."


def test_top_alpha_empty_state_uses_polymarket_diagnostic_when_no_prices_exist() -> None:
    message = top_alpha_empty_state_message(
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        {"reason_no_alpha_rows": "Event resolved; relevant market found but no usable price."},
    )

    assert message == "Event resolved; relevant market found but no usable price."


def test_top_alpha_empty_state_has_generic_selected_match_price_message() -> None:
    message = top_alpha_empty_state_message(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {})

    assert message == "No fixture-specific Polymarket price rows are available for this selected match."


def test_markets_tab_export_dataframe_equals_display_dataframe() -> None:
    joined = join_polymarket_prices_to_model_markets(_model_markets().head(3), _event_markets(), "Uruguay", "Spain")
    table, _diagnostics = build_markets_tab_joined_dataframe(
        _model_markets().head(3),
        joined,
        {"slug_resolution_status": "resolved", "event_markets_loaded_count": 5, "joined_markets_count": 3},
    )

    pd.testing.assert_frame_equal(markets_tab_export_dataframe(table), table)


def test_markets_tab_diagnostics_explain_no_slug() -> None:
    table, diagnostics = build_markets_tab_joined_dataframe(
        _model_markets().head(1),
        pd.DataFrame(),
        {"slug_resolution_status": "unresolved", "event_markets_loaded_count": 0, "joined_markets_count": 0},
    )

    assert table["market_odds"].isna().all()
    assert diagnostics["market_join_status"] == "no_event_resolved"
    assert diagnostics["reason_if_zero"] == "No Polymarket event resolved."


def test_markets_tab_diagnostics_explain_mapping_failure() -> None:
    no_match = _event_markets().loc[_event_markets()["market_type"] == "moneyline"].copy()
    model = _model_markets().loc[_model_markets()["market"] == "BTTS"].copy()
    joined = join_polymarket_prices_to_model_markets(model, no_match, "Uruguay", "Spain")
    table, diagnostics = build_markets_tab_joined_dataframe(
        model,
        joined,
        {"slug_resolution_status": "resolved", "event_markets_loaded_count": len(no_match), "joined_markets_count": 0},
    )

    assert table["market_odds"].isna().all()
    assert diagnostics["market_join_status"] == "event_found_no_relevant_market"
    assert diagnostics["reason_if_zero"] == "Event resolved; no relevant market found."


def test_polymarket_alpha_table_not_empty_when_markets_match() -> None:
    joined = join_polymarket_prices_to_model_markets(_model_markets(), _event_markets(), "Uruguay", "Spain")
    alpha_rows = polymarket_alpha_rows(joined)

    assert not alpha_rows.empty
    assert "market_price_cents" in alpha_rows.columns
    assert "polymarket_price_cents" in alpha_rows.columns
    assert "alpha_ev" in alpha_rows.columns
    assert "signal_strength" in alpha_rows.columns
    assert alpha_rows["polymarket_price_cents"].notna().all()


def test_tournament_context_not_displayed_in_dashboard_by_default() -> None:
    app_source = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")

    assert "Tournament Context" not in app_source
    assert "tournament_context" not in app_source


def test_market_value_tables_do_not_require_context_columns() -> None:
    joined = join_polymarket_prices_to_model_markets(_model_markets(), _event_markets(), "Uruguay", "Spain")
    groups = joined_market_groups(joined)

    all_columns = {column for group in groups.values() for column in group.columns}
    assert "Odds / Price" in all_columns
    assert "EV / Alpha Gap" in all_columns
    assert "Score" in all_columns
    assert "Signal" in all_columns
    assert not any(column.startswith("Context") for column in all_columns)


def test_build_alpha_for_fixture_user_supplied_slug_overrides_search(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.polymarket_slug_join.fetch_polymarket_event_by_slug_with_diagnostics",
        lambda *_args, **_kwargs: (_synthetic_event(), {"method": "synthetic", "warnings": []}),
    )

    joined, diagnostics = build_polymarket_alpha_for_fixture(
        "Uruguay",
        "Spain",
        "2026-06-26",
        "World Cup",
        _model_markets(),
        user_supplied_slug_or_url="https://polymarket.com/sports/world-cup/fifwc-ury-esp-2026-06-26",
    )

    assert diagnostics["user_supplied_slug"] == "fifwc-ury-esp-2026-06-26"
    assert diagnostics["resolved_slug"] == "fifwc-ury-esp-2026-06-26"
    assert diagnostics["slug_resolution_status"] == "resolved"
    assert diagnostics["event_markets_loaded_count"] > 0
    assert diagnostics["joined_markets_count"] > 0
    assert joined["market_price_cents"].notna().any()


def test_top_alpha_and_polymarket_alpha_not_empty_when_prices_join(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.polymarket_slug_join.fetch_polymarket_event_by_slug_with_diagnostics",
        lambda *_args, **_kwargs: (_synthetic_event(), {"method": "synthetic", "warnings": []}),
    )

    joined, diagnostics = build_polymarket_alpha_for_fixture(
        "Uruguay",
        "Spain",
        "2026-06-26",
        "World Cup",
        _model_markets(),
        user_supplied_slug_or_url="fifwc-ury-esp-2026-06-26",
    )
    alpha_rows = polymarket_alpha_rows(joined)

    assert diagnostics["top_alpha_rows_count"] == len(alpha_rows)
    assert not alpha_rows.empty
    assert diagnostics["reason_no_alpha_rows"] == ""


def test_empty_alpha_returns_diagnostics_instead_of_silent_no_rows(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.polymarket_slug_join.fetch_polymarket_event_by_slug_with_diagnostics",
        lambda *_args, **_kwargs: (None, {"method": "synthetic", "warnings": ["No event found for slug."]}),
    )

    joined, diagnostics = build_polymarket_alpha_for_fixture(
        "Uruguay",
        "Spain",
        "2026-06-26",
        "World Cup",
        _model_markets(),
        user_supplied_slug_or_url="fifwc-ury-esp-2026-06-26",
    )
    alpha_rows = polymarket_alpha_rows(joined)

    assert alpha_rows.empty
    assert diagnostics["slug_resolution_status"] == "resolved"
    assert diagnostics["event_markets_loaded_count"] == 0
    assert diagnostics["joined_markets_count"] == 0
    assert diagnostics["market_join_status"] == "event_found_no_relevant_market"
    assert diagnostics["reason_no_alpha_rows"] == "Event resolved; no relevant market found."


def test_html_fallback_extracts_colombia_draw_portugal_moneyline() -> None:
    html = """
    <html><head><title>Colombia - Portugal | Polymarket</title></head>
    <body><section>Moneyline COL 28¢ Draw 25¢ PRT 48¢</section></body></html>
    """

    markets = extract_markets_from_polymarket_event_html(html, "fifwc-col-prt-2026-06-27")

    moneyline = markets.loc[markets["market_type"] == "moneyline"]
    assert set(moneyline["outcome_name"]) == {"Colombia", "Draw", "Portugal"}
    assert set(moneyline["price_cents"]) == {28.0, 25.0, 48.0}


def test_partial_html_extraction_drives_non_empty_alpha_diagnostics(monkeypatch) -> None:
    html = "<html><body>Moneyline COL 28¢ Draw 25¢ PRT 48¢</body></html>"
    monkeypatch.setattr(
        "src.polymarket_slug_join.fetch_polymarket_event_by_slug_with_diagnostics",
        lambda *_args, **_kwargs: (None, {"method": "synthetic", "warnings": []}),
    )
    monkeypatch.setattr("src.polymarket_slug_join.load_polymarket_event_page_html", lambda *_args, **_kwargs: html)

    joined, diagnostics = build_polymarket_alpha_for_fixture(
        "Colombia",
        "Portugal",
        "2026-06-27",
        "World Cup",
        _model_markets(),
    )
    alpha_rows = polymarket_alpha_rows(joined)

    assert diagnostics["registry_match_found"] is True
    assert diagnostics["event_markets_loaded_count"] == 3
    assert diagnostics["event_market_types_found"] == ["moneyline"]
    assert diagnostics["joined_markets_count"] == 3
    assert diagnostics["top_alpha_rows_count"] == len(alpha_rows)
    assert not alpha_rows.empty


def test_user_supplied_url_overrides_registry_for_colombia_portugal(monkeypatch) -> None:
    html = "<html><body>Moneyline COL 28¢ Draw 25¢ PRT 48¢</body></html>"
    monkeypatch.setattr(
        "src.polymarket_slug_join.fetch_polymarket_event_by_slug_with_diagnostics",
        lambda *_args, **_kwargs: (None, {"method": "synthetic", "warnings": []}),
    )
    monkeypatch.setattr("src.polymarket_slug_join.load_polymarket_event_page_html", lambda *_args, **_kwargs: html)

    joined, diagnostics = build_polymarket_alpha_for_fixture(
        "Colombia",
        "Portugal",
        "2026-06-27",
        "World Cup",
        _model_markets(),
        user_supplied_slug_or_url="https://polymarket.com/sports/world-cup/fifwc-col-prt-2026-06-27",
    )

    assert diagnostics["slug_source"] == "user_supplied_slug_or_url"
    assert diagnostics["resolved_slug"] == "fifwc-col-prt-2026-06-27"
    assert joined["market_price_cents"].notna().sum() == 3


def test_event_market_loader_extracts_outcome_rows(monkeypatch) -> None:
    event = _synthetic_event()
    cache_calls = []

    monkeypatch.setattr(
        "src.polymarket_slug_join.fetch_polymarket_event_by_slug_with_diagnostics",
        lambda *_args, **_kwargs: (event, {"method": "gamma_events_slug_filter", "warnings": []}),
    )
    monkeypatch.setattr(
        "src.polymarket_slug_join.write_polymarket_discovery_cache",
        lambda events, markets: cache_calls.append((events, markets.copy())),
    )

    markets = load_polymarket_event_markets_by_slug("fifwc-ury-esp-2026-06-26")

    assert set(markets["market_type"]) == {"moneyline", "total"}
    assert "Uruguay" in set(markets["outcome_name"])
    assert "Over 2.5" in set(markets["outcome_name"])
    assert len(cache_calls) == 1
    assert cache_calls[0][0] == [event]
    assert set(cache_calls[0][1]["market_id"]) == {"ML-URY", "TOT-OVER", "TOT-UNDER"}


def _synthetic_event() -> dict:
    return {
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
            {
                "id": "TOT-UNDER",
                "slug": "fifwc-ury-esp-2026-06-26-under-2-5",
                "question": "Will total goals be under 2.5?",
                "outcomes": '["Yes", "No"]',
                "outcomePrices": '["0.52", "0.48"]',
            },
        ],
    }
