from __future__ import annotations

import pandas as pd

import scripts.audit_performance_residuals as residual_audit_module
from src.performance_residuals import (
    add_performance_residuals,
    attack_index_from_goal_residual,
    blowout_weight,
    calculate_expected_goals_from_elo,
    calculate_expected_result_from_elo,
    cap_residual,
    defense_index_from_goal_residual,
    final_index_from_raw_and_residual,
)
from src.team_behavior import build_team_behavior_table


def _residual_match_rows() -> pd.DataFrame:
    rows = []
    for i in range(12):
        rows.append(
            {
                "match_id": f"m{i}",
                "date_utc": (pd.Timestamp("2026-06-01") - pd.Timedelta(days=i * 20)).date().isoformat(),
                "competition": "Friendly",
                "competition_type": "friendly",
                "team": "Alpha",
                "opponent": f"Weak {i}",
                "team_goals": 1,
                "opponent_goals": 2,
                "team_elo_pre": 1900,
                "opponent_elo": 1800,
                "is_home": 0,
                "is_neutral": 1,
            }
        )
    return pd.DataFrame(rows)


def _blowout_match_rows() -> pd.DataFrame:
    rows = []
    for i in range(12):
        is_blowout = i == 0
        rows.append(
            {
                "match_id": f"b{i}",
                "date_utc": (pd.Timestamp("2026-06-01") - pd.Timedelta(days=i * 10)).date().isoformat(),
                "competition": "Friendly",
                "competition_type": "friendly",
                "team": "Alpha",
                "opponent": f"Opp {i}",
                "team_goals": 8 if is_blowout else 1,
                "opponent_goals": 0 if is_blowout else 1,
                "team_elo_pre": 1230,
                "opponent_elo": 1500,
                "is_home": 0,
                "is_neutral": 1,
            }
        )
    return pd.DataFrame(rows)


def test_expected_score_from_elo_is_standard_elo_probability() -> None:
    assert calculate_expected_result_from_elo(1500, 1500, neutral=True) == 0.5
    assert calculate_expected_result_from_elo(1600, 1500, neutral=True) > 0.5
    assert calculate_expected_result_from_elo(1500, 1500, neutral=False) > 0.5


def test_expected_goals_increases_with_elo_advantage() -> None:
    favorite = calculate_expected_goals_from_elo(1700, 1500)
    underdog = calculate_expected_goals_from_elo(1300, 1500)

    assert favorite > calculate_expected_goals_from_elo(1500, 1500)
    assert underdog < calculate_expected_goals_from_elo(1500, 1500)


def test_expected_goals_is_capped() -> None:
    assert calculate_expected_goals_from_elo(4000, 900) == 3.50
    assert calculate_expected_goals_from_elo(900, 4000) == 0.25


def test_cap_residual_limits_extreme_values() -> None:
    assert cap_residual(3.2, cap=1.75) == 1.75
    assert cap_residual(-3.2, cap=1.75) == -1.75
    assert cap_residual(0.4, cap=1.75) == 0.4


def test_blowout_weight_downweights_large_margins() -> None:
    assert blowout_weight(2, 1) == 1.0
    assert blowout_weight(5, 1) < 1.0
    assert blowout_weight(10, 0) >= 0.55


def test_positive_goals_for_residual_improves_attack() -> None:
    assert attack_index_from_goal_residual(0.6) > 0.5
    assert attack_index_from_goal_residual(-0.6) < 0.5


def test_positive_goals_against_residual_lowers_defense() -> None:
    assert defense_index_from_goal_residual(0.6) < 0.5
    assert defense_index_from_goal_residual(-0.6) > 0.5


def test_residual_final_index_remains_between_zero_and_one() -> None:
    assert 0.0 <= final_index_from_raw_and_residual(2.0, 2.0) <= 1.0
    assert 0.0 <= final_index_from_raw_and_residual(-1.0, -1.0) <= 1.0


def test_add_performance_residuals_adds_expected_and_residual_columns() -> None:
    rows = add_performance_residuals(_residual_match_rows().head(1))
    row = rows.iloc[0]

    assert row["expected_result_score"] > 0.5
    assert row["actual_result_score"] == 0.0
    assert row["result_residual"] < 0
    assert row["goals_for_residual"] < 0
    assert row["goals_against_residual"] > 0
    assert "goals_for_residual_capped" in rows.columns
    assert "goals_against_residual_capped" in rows.columns
    assert "blowout_weight" in rows.columns


def test_robust_residual_is_lower_than_raw_residual_for_blowouts() -> None:
    behavior = build_team_behavior_table(
        _blowout_match_rows(),
        pd.DataFrame({"team": ["Alpha"]}),
        "2026-06-21",
    )
    row = behavior.iloc[0]

    assert row["weighted_goal_for_residual_raw"] > row["weighted_goal_for_residual_robust"]
    assert row["attack_index_residual_raw"] > row["attack_index_residual_robust"]


def test_residual_concentration_warning_triggers() -> None:
    behavior = build_team_behavior_table(
        _blowout_match_rows(),
        pd.DataFrame({"team": ["Alpha"]}),
        "2026-06-21",
    )
    row = behavior.iloc[0]

    assert row["top_3_attack_residual_share"] >= 0.60
    assert "Attack residuals are concentrated" in row["residual_concentration_warning"]


def test_warning_triggers_when_opponent_boost_and_residual_disagree() -> None:
    behavior = build_team_behavior_table(
        _residual_match_rows(),
        pd.DataFrame({"team": ["Alpha"]}),
        "2026-06-21",
    )
    row = behavior.iloc[0]

    assert row["attack_index_final"] < row["attack_index_raw"]
    assert "goal residual is negative" in row["residual_warning"]
    assert "goals-against residual is poor" in row["residual_warning"]


def test_performance_residual_audit_script_runs_without_crashing(monkeypatch, tmp_path) -> None:
    behavior = pd.DataFrame(
        [
            {
                "team": "Alpha",
                "matches_used_recent": 12,
                "mean_opponent_elo_recent": 1200,
                "attack_index_raw": 0.8,
                "attack_index_residual_raw": 0.45,
                "attack_index_residual_robust": 0.4,
                "attack_index_final": 0.68,
                "weighted_goal_for_residual_raw": -0.5,
                "weighted_goal_for_residual_robust": -0.3,
                "top_3_attack_residual_share": 0.7,
                "defense_index_raw": 0.6,
                "defense_index_residual_raw": 0.3,
                "defense_index_residual_robust": 0.35,
                "defense_index_final": 0.525,
                "weighted_goal_against_residual_raw": 0.6,
                "weighted_goal_against_residual_robust": 0.45,
                "top_3_defense_residual_share": 0.4,
                "residual_concentration_warning": "Attack residuals are concentrated in the top 3 matches.",
                "residual_warning": "Large raw-vs-residual attack disagreement.",
            }
        ]
    )
    monkeypatch.setattr(residual_audit_module, "load_team_behavior", lambda: behavior)

    audit = residual_audit_module.build_residual_audit_table(["Alpha"])
    report_path = residual_audit_module.write_residual_audit_report(audit, tmp_path / "residuals.md")

    assert not audit.empty
    assert list(audit.columns) == residual_audit_module.AUDIT_COLUMNS
    assert "attack_index_residual_robust" in audit.columns
    assert "top_3_attack_residual_share" in audit.columns
    assert report_path.exists()
