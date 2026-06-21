from __future__ import annotations

import sys

import pandas as pd

import scripts.audit_behavior_drivers as behavior_driver_script
import scripts.audit_model_inputs as model_input_script
from src.behavior_driver_report import (
    audit_final_model_inputs,
    behavior_quality_warnings,
    calculate_competition_breakdown,
    calculate_opponent_tier_breakdown,
    classify_opponent_tier,
    get_behavior_driver_matches,
)


def _matches() -> pd.DataFrame:
    rows = []
    specs = [
        ("m1", "2026-06-01", "FIFA World Cup", "world_cup", "Elite", 4, 1, 1780, 1900),
        ("m2", "2026-05-01", "Friendly", "friendly", "Weak", 5, 0, 1780, 1300),
        ("m3", "2026-04-01", "Nations League", "nations_league", "Average", 1, 1, 1780, 1600),
        ("m4", "2026-03-01", "Qualifier", "world_cup_qualifier", "Strong", 0, 1, 1780, 1750),
    ]
    for match_id, date_utc, competition, competition_type, opponent, gf, ga, team_elo, opp_elo in specs:
        rows.append(
            {
                "match_id": match_id,
                "date_utc": date_utc,
                "competition": competition,
                "competition_type": competition_type,
                "team": "Alpha",
                "opponent": opponent,
                "team_goals": gf,
                "opponent_goals": ga,
                "team_elo_pre": team_elo,
                "opponent_elo": opp_elo,
                "is_home": 0,
                "is_neutral": 1,
            }
        )
    return pd.DataFrame(rows)


def _behavior() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "team": "Alpha",
                "attack_index": 0.9,
                "attack_index_final": 0.9,
                "attack_index_raw": 0.9,
                "attack_index_residual_robust": 0.6,
                "defense_index": 0.6,
                "recent_form_index": 0.7,
                "top_3_attack_residual_share": 0.7,
                "top_3_defense_residual_share": 0.2,
                "overall_data_quality": "high",
                "behavior_warning": "",
            }
        ]
    )


def _raw_ratings() -> pd.DataFrame:
    return pd.DataFrame([{"team": "Alpha", "attack": 0.5, "defense": 0.55, "recent_form": 0.45}])


def _adjusted_ratings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "team": "Alpha",
                "manual_attack": 0.5,
                "manual_defense": 0.55,
                "manual_recent_form": 0.45,
                "attack": 0.62,
                "defense": 0.57,
                "recent_form": 0.58,
                "behavior_attack_index": 0.9,
                "behavior_defense_index": 0.6,
                "behavior_recent_form_index": 0.7,
                "behavior_attack_delta": 0.12,
                "behavior_defense_delta": 0.02,
                "behavior_recent_form_delta": 0.13,
                "behavior_attack_cap_hit": True,
                "behavior_defense_cap_hit": False,
                "behavior_recent_form_cap_hit": False,
                "behavior_blend_used": True,
            }
        ]
    )


def test_opponent_tier_classification() -> None:
    assert classify_opponent_tier(1900) == "elite"
    assert classify_opponent_tier(1750) == "strong"
    assert classify_opponent_tier(1600) == "average"
    assert classify_opponent_tier(1300) == "weak"
    assert classify_opponent_tier(None) == "unknown"


def test_competition_breakdown_returns_competition_types() -> None:
    breakdown = calculate_competition_breakdown(_matches(), "Alpha", "2026-06-21")

    assert "competition_type" in breakdown.columns
    assert {"world_cup", "friendly", "nations_league", "world_cup_qualifier"}.issubset(set(breakdown["competition_type"]))


def test_opponent_tier_breakdown_returns_tiers() -> None:
    breakdown = calculate_opponent_tier_breakdown(_matches(), "Alpha", "2026-06-21")

    assert "opponent_tier" in breakdown.columns
    assert {"elite", "strong", "average", "weak"}.issubset(set(breakdown["opponent_tier"]))


def test_driver_matches_are_sorted_by_contribution() -> None:
    drivers = get_behavior_driver_matches(_matches(), "Alpha", "2026-06-21", n=4, sort_by="attack")

    contributions = drivers["attack_contribution"].tolist()
    assert contributions == sorted(contributions, reverse=True)


def test_final_model_input_audit_shows_adjusted_values() -> None:
    audit = audit_final_model_inputs(_raw_ratings(), _adjusted_ratings(), _behavior())
    row = audit.iloc[0]

    assert row["manual_attack"] == 0.5
    assert row["behavior_attack_final"] == 0.9
    assert row["adjusted_attack_used_by_model"] == 0.62
    assert bool(row["attack_delta_cap_applied"]) is True
    assert bool(row["behavior_blend_used"]) is True


def test_warning_triggers_when_behavior_high_but_manual_prior_low() -> None:
    warning = behavior_quality_warnings(_behavior().iloc[0], _raw_ratings().iloc[0], _adjusted_ratings().iloc[0])

    assert "high_behavior_low_manual_prior" in warning
    assert "behavior_final_above_0_85_but_manual_below_0_65" in warning
    assert "top3_residual_share_high" in warning


def test_no_crash_with_missing_residual_columns() -> None:
    sparse = _matches().drop(columns=["team_elo_pre", "opponent_elo"])
    drivers = get_behavior_driver_matches(sparse, "Alpha", "2026-06-21", n=2)

    assert not drivers.empty
    assert "goals_for_residual_robust" in drivers.columns


def test_audit_behavior_drivers_script_runs_without_crashing(monkeypatch) -> None:
    monkeypatch.setattr(behavior_driver_script, "load_historical_matches", lambda use_cache=False: _matches())
    monkeypatch.setattr(behavior_driver_script, "load_team_behavior", _behavior)
    monkeypatch.setattr(behavior_driver_script, "read_csv_with_columns", lambda path, columns: _raw_ratings())
    monkeypatch.setattr(sys, "argv", ["audit_behavior_drivers.py", "--teams", "Alpha", "--reference-date", "2026-06-21"])

    behavior_driver_script.main()


def test_audit_model_inputs_script_runs_without_crashing(monkeypatch) -> None:
    monkeypatch.setattr(model_input_script, "read_csv_with_columns", lambda path, columns: _raw_ratings())
    monkeypatch.setattr(model_input_script, "get_team_ratings", _adjusted_ratings)
    monkeypatch.setattr(model_input_script, "load_team_behavior", _behavior)

    model_input_script.main()
