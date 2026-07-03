from __future__ import annotations

import pandas as pd

from src.market_tables import group_market_alpha
from src.report_metrics import build_expected_goals_df, calculate_goal_distribution, calculate_league_context
from src.timeline import calculate_score_timeline


def test_goal_distribution_uses_score_matrix_marginals() -> None:
    matrix = pd.DataFrame(
        [
            {"home_goals": 0, "away_goals": 0, "prob": 0.10},
            {"home_goals": 1, "away_goals": 0, "prob": 0.20},
            {"home_goals": 1, "away_goals": 1, "prob": 0.30},
            {"home_goals": 2, "away_goals": 1, "prob": 0.40},
        ]
    )

    out = calculate_goal_distribution(matrix, max_goals=2)

    assert out.loc[out["goals"] == 1, "home_probability"].iloc[0] == 0.50
    assert out.loc[out["goals"] == 1, "away_probability"].iloc[0] == 0.70


def test_score_timeline_shape_and_bounds() -> None:
    out = calculate_score_timeline(1.4, 1.1)

    assert list(out.columns) == ["minute", "home_goal_probability", "away_goal_probability", "any_goal_probability"]
    assert out["minute"].iloc[0] == 0
    assert out["minute"].iloc[-1] == 90
    assert out["any_goal_probability"].between(0, 1).all()
    assert out["any_goal_probability"].is_monotonic_increasing


def test_league_context_uses_fallback_when_history_missing() -> None:
    result = {
        "hxg": 1.3,
        "axg": 1.1,
        "probs": {"home_win": 0.45, "draw": 0.28, "away_win": 0.27, "btts_yes": 0.50, "over_2_5": 0.47},
    }

    out = calculate_league_context(pd.DataFrame(), result)

    assert len(out) == 6
    assert out["source"].eq("fallback baseline").all()
    assert abs(out.loc[out["Metric"] == "Goals per match", "This match"].iloc[0] - 2.4) < 1e-9


def test_expected_goals_report_uses_base_and_adjusted_components_without_todo() -> None:
    result = {
        "home": "Alpha",
        "away": "Beta",
        "hxg": 1.6,
        "axg": 0.8,
        "components": {
            "home_raw_xg_before_adjustments": 1.4,
            "away_raw_xg_before_adjustments": 0.9,
        },
    }

    out = build_expected_goals_df(result)

    assert out.loc[out["team"] == "Alpha", "base_xg"].iloc[0] == 1.4
    assert out.loc[out["team"] == "Alpha", "adjusted_xg"].iloc[0] == 1.6
    assert {"rating_adjusted_xg", "form_adjusted_xg", "venue_adjusted_xg", "final_xg"}.issubset(out.columns)
    assert "TODO" not in out.attrs.get("note", "")


def test_group_market_alpha_builds_polymarket_group() -> None:
    alpha = pd.DataFrame(
        [
            {
                "market": "Total",
                "selection": "Over 2.5",
                "market_odds": 2.1,
                "fair_odds": 1.9,
                "alpha_ev": 0.08,
            }
        ]
    )
    polymarket = pd.DataFrame(
        [
            {
                "question": "Will Team A beat Team B?",
                "polymarket_side": "YES",
                "polymarket_price_cents": 42.0,
                "fair_price_cents": 49.0,
                "alpha_gap_cents": 7.0,
                "mapping_confidence": "high",
                "liquidity": 1000,
                "signal_strength": "Moderate",
            }
        ]
    )

    groups = group_market_alpha(alpha, polymarket)

    assert len(groups["High Scoring"]) == 1
    assert len(groups["Polymarket Alpha"]) == 1
    assert groups["Polymarket Alpha"].iloc[0]["Signal"] == "Moderate"


def test_group_market_alpha_uses_explicit_missing_value_statuses() -> None:
    alpha = pd.DataFrame(
        [
            {
                "market": "Total",
                "selection": "Under 2.5",
                "market_odds": pd.NA,
                "fair_odds": 1.8,
                "alpha_ev": pd.NA,
            }
        ]
    )

    groups = group_market_alpha(alpha, pd.DataFrame())
    row = groups["Low Scoring"].iloc[0]

    assert row["Odds / Price"] == "No local odds"
    assert row["EV / Alpha Gap"] == "Model-only"
    assert "" not in row.astype(str).tolist()
