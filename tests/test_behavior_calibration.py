from __future__ import annotations

import pandas as pd

import src.ratings as ratings_module
import scripts.audit_schedule_strength as schedule_audit_module
from scripts.audit_behavior_calibration import AUDIT_COLUMNS, build_audit_table, write_audit_report
from src.behavior_calibration import (
    BehaviorConfig,
    calculate_opponent_quality_modifier,
    filter_recent_team_matches,
)
from src.recency import calculate_match_weight, competition_weight
from src.team_behavior import build_team_behavior_table


def _rating_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "team": "Alpha",
                "elo": 1700,
                "attack": 0.20,
                "defense": 0.80,
                "recent_form": 0.00,
                "fifa_rank_proxy": 50,
                "training_temp_c": 20,
                "training_humidity_pct": 60,
                "data_quality": "manual_csv",
                "last_updated": "2026-06-21",
                "notes": "",
            }
        ]
    )


def _behavior_row(quality: str = "high", n_matches: int = 30) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "team": "Alpha",
                "n_matches": n_matches,
                "matches_used_recent": n_matches,
                "attack_index": 1.0,
                "defense_index": 0.0,
                "recent_form_index": 1.0,
                "overall_data_quality": quality,
                "opponent_elo_coverage_recent": 1.0,
                "schedule_strength_label": "average",
            }
        ]
    )


def _match_rows(count: int = 6) -> pd.DataFrame:
    rows = []
    for i in range(count):
        rows.append(
            {
                "match_id": f"m{i}",
                "date_utc": (pd.Timestamp("2026-06-01") - pd.Timedelta(days=i * 30)).date().isoformat(),
                "competition": "Friendly" if i % 2 else "FIFA World Cup",
                "competition_type": "friendly" if i % 2 else "world_cup",
                "team": "Alpha",
                "opponent": f"Opp {i}",
                "team_goals": 2,
                "opponent_goals": 1,
                "opponent_elo": 1500,
            }
        )
    rows.append(
        {
            "match_id": "old",
            "date_utc": "2010-01-01",
            "competition": "FIFA World Cup",
            "competition_type": "world_cup",
            "team": "Alpha",
            "opponent": "Old Opp",
            "team_goals": 6,
            "opponent_goals": 0,
            "opponent_elo": 1200,
        }
    )
    return pd.DataFrame(rows)


def test_recent_filter_excludes_old_matches() -> None:
    filtered = filter_recent_team_matches(_match_rows(), "Alpha", "2026-06-21", BehaviorConfig(lookback_years=4))

    assert "Old Opp" not in filtered["opponent"].tolist()
    assert filtered["date_utc"].min() >= pd.Timestamp("2022-06-21")


def test_recent_filter_respects_max_matches() -> None:
    filtered = filter_recent_team_matches(_match_rows(12), "Alpha", "2026-06-21", BehaviorConfig(max_matches=5))

    assert len(filtered) == 5


def test_friendlies_are_down_weighted() -> None:
    assert competition_weight("friendly") == 0.45
    reference = "2026-06-21"
    friendly = calculate_match_weight(pd.Series({"date_utc": "2026-06-01", "competition_type": "friendly"}), reference)
    other = calculate_match_weight(pd.Series({"date_utc": "2026-06-01", "competition_type": "other"}), reference)

    assert friendly < other


def test_opponent_quality_modifier_is_capped() -> None:
    assert calculate_opponent_quality_modifier(1800) > calculate_opponent_quality_modifier(1500)
    assert calculate_opponent_quality_modifier(1200) < calculate_opponent_quality_modifier(1500)
    assert calculate_opponent_quality_modifier(2500, strength=1.0) == 1.25
    assert calculate_opponent_quality_modifier(500, strength=1.0) == 0.75


def test_behavior_index_remains_between_zero_and_one() -> None:
    behavior = build_team_behavior_table(_match_rows(12), pd.DataFrame({"team": ["Alpha"]}), "2026-06-21")
    row = behavior.iloc[0]

    assert 0.0 <= row["attack_index"] <= 1.0
    assert 0.0 <= row["defense_index"] <= 1.0
    assert row["matches_available_all_time"] > row["matches_used_recent"]


def test_insufficient_sample_uses_manual_rating_only(monkeypatch) -> None:
    monkeypatch.setattr(ratings_module, "load_team_behavior", lambda: _behavior_row("insufficient", 4))

    adjusted = ratings_module._apply_behavior_blend(_rating_rows(), manual_base=_rating_rows())
    row = adjusted.iloc[0]

    assert row["attack"] == 0.20
    assert row["defense"] == 0.80
    assert row["recent_form"] == 0.00
    assert not bool(row["behavior_blend_used"])


def test_behavior_delta_caps_work(monkeypatch) -> None:
    monkeypatch.setattr(ratings_module, "load_team_behavior", lambda: _behavior_row("high", 30))

    adjusted = ratings_module._apply_behavior_blend(_rating_rows(), manual_base=_rating_rows())
    row = adjusted.iloc[0]

    assert row["attack"] == 0.32
    assert row["defense"] == 0.68
    assert row["recent_form"] == 0.18
    assert bool(row["behavior_blend_cap_hit"])
    assert row["behavior_attack_delta"] == 0.12
    assert row["behavior_defense_delta"] == -0.12
    assert row["behavior_recent_form_delta"] == 0.18


def test_weak_schedule_warning_works() -> None:
    rows = _match_rows(12)
    rows["opponent_elo"] = 1250

    behavior = build_team_behavior_table(rows, pd.DataFrame({"team": ["Alpha"]}), "2026-06-21")
    row = behavior.iloc[0]

    assert row["schedule_strength_label"] == "weak"
    assert "weak opponents" in row["schedule_strength_warning"]


def test_adjusted_attack_index_is_lower_against_weak_opponents() -> None:
    rows = _match_rows(12)
    rows["team_goals"] = 4
    rows["opponent_goals"] = 0
    rows["opponent_elo"] = 1200

    behavior = build_team_behavior_table(rows, pd.DataFrame({"team": ["Alpha"]}), "2026-06-21")
    row = behavior.iloc[0]

    assert row["attack_index_adjusted"] < row["attack_index_raw"]
    assert "Attack index reduced" in row["opponent_adjustment_warning"]


def test_weak_schedule_reduces_attack_blend(monkeypatch) -> None:
    weak_behavior = _behavior_row("high", 30)
    weak_behavior["schedule_strength_label"] = "weak"
    weak_behavior["attack_index_raw"] = 1.0
    weak_behavior["attack_index_adjusted"] = 0.8
    weak_behavior["attack_index"] = 0.8
    monkeypatch.setattr(ratings_module, "load_team_behavior", lambda: weak_behavior)

    adjusted = ratings_module._apply_behavior_blend(_rating_rows(), manual_base=_rating_rows())
    row = adjusted.iloc[0]

    assert row["attack"] == 0.29
    assert row["behavior_attack_delta"] == 0.09


def test_audit_script_runs_without_crashing(tmp_path) -> None:
    audit = build_audit_table(["England"])
    report_path = write_audit_report(audit, tmp_path / "audit.md")

    assert list(audit.columns) == AUDIT_COLUMNS
    assert report_path.exists()


def test_schedule_audit_script_runs_without_crashing(monkeypatch, tmp_path) -> None:
    behavior = pd.DataFrame(
        [
            {
                "team": "Alpha",
                "matches_used_recent": 12,
                "mean_opponent_elo_recent": 1400,
                "strong_opponent_match_count": 0,
                "weak_opponent_match_count": 8,
                "schedule_strength_label": "weak",
                "attack_index_raw": 0.9,
                "attack_index_adjusted": 0.8,
                "defense_index_raw": 0.6,
                "defense_index_adjusted": 0.55,
                "recent_form_index": 0.7,
                "schedule_strength_warning": "Recent schedule skews toward weak opponents.",
            }
        ]
    )
    monkeypatch.setattr(schedule_audit_module, "load_team_behavior", lambda: behavior)

    audit = schedule_audit_module.build_schedule_audit_table(["Alpha"])
    report_path = schedule_audit_module.write_schedule_audit_report(audit, tmp_path / "schedule.md")

    assert not audit.empty
    assert report_path.exists()
