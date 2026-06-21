from __future__ import annotations

import sys

import pandas as pd

from src import external_benchmark_calibration as calib


def _ratings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "team": "Generated",
                "elo": 1700,
                "attack": 0.55,
                "defense": 0.56,
                "recent_form": 0.57,
                "fifa_rank_proxy": 60,
                "training_temp_c": 20,
                "training_humidity_pct": 60,
                "data_quality": "generated_from_behavior",
                "last_updated": "2026-06-21",
                "notes": "",
            },
            {
                "team": "Manual",
                "elo": 1800,
                "attack": 0.70,
                "defense": 0.69,
                "recent_form": 0.68,
                "fifa_rank_proxy": 20,
                "training_temp_c": 20,
                "training_humidity_pct": 60,
                "data_quality": "manual_csv",
                "last_updated": "2026-06-21",
                "notes": "",
            },
            {
                "team": "Reviewed",
                "elo": 1850,
                "attack": 0.72,
                "defense": 0.71,
                "recent_form": 0.70,
                "fifa_rank_proxy": 15,
                "training_temp_c": 20,
                "training_humidity_pct": 60,
                "data_quality": "manual_reviewed",
                "last_updated": "2026-06-21",
                "notes": "",
            },
        ]
    )


def _behavior() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"team": "Generated", "attack_index_final": 0.80, "defense_index_final": 0.78, "recent_form_index": 0.77},
            {"team": "Manual", "attack_index_final": 0.73, "defense_index_final": 0.72, "recent_form_index": 0.71},
            {"team": "Reviewed", "attack_index_final": 0.75, "defense_index_final": 0.74, "recent_form_index": 0.73},
        ]
    )


def _priors() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "team": "Generated",
                "reference_attack": pd.NA,
                "reference_defense": pd.NA,
                "reference_recent_form": pd.NA,
                "reference_overall_strength": 0.90,
                "source": "test",
                "last_updated": "2026-06-21",
                "notes": "",
            },
            {
                "team": "Manual",
                "reference_attack": pd.NA,
                "reference_defense": pd.NA,
                "reference_recent_form": pd.NA,
                "reference_overall_strength": 0.85,
                "source": "test",
                "last_updated": "2026-06-21",
                "notes": "",
            },
            {
                "team": "Reviewed",
                "reference_attack": pd.NA,
                "reference_defense": pd.NA,
                "reference_recent_form": pd.NA,
                "reference_overall_strength": 0.88,
                "source": "test",
                "last_updated": "2026-06-21",
                "notes": "",
            },
        ]
    )


def test_strength_scaling_and_combining_external_sources() -> None:
    rank_strength = calib.scale_fifa_rank_to_strength(1)
    points_strength = calib.scale_fifa_points_to_strength(1900)
    elo_strength = calib.scale_external_elo_to_strength(2200)

    assert rank_strength == 0.9
    assert points_strength == 0.9
    assert elo_strength == 0.9
    assert calib.combine_external_strengths(rank_strength, points_strength, elo_strength) == 0.9
    assert pd.isna(calib.combine_external_strengths())


def test_missing_benchmark_warning() -> None:
    row = calib.calibrate_rating_against_external("Generated", _ratings().iloc[0], _behavior().iloc[0], pd.Series(dtype="object"))

    assert "missing_external_benchmark" in row["warning"]
    assert row["calibrated_attack"] == row["current_attack"]


def test_generated_rating_calibration_and_delta_cap() -> None:
    proposals = calib.build_external_calibration_proposals(_ratings().head(1), _behavior(), _priors())
    row = proposals.iloc[0]

    assert row["data_quality_proposed"] == "external_benchmark_calibrated"
    assert "generated_rating_calibrated" in row["warning"]
    assert "large_delta_capped" in row["warning"]
    assert row["attack_delta"] == 0.08
    assert row["calibrated_attack"] == 0.63


def test_manual_existing_preserved_by_default_write_gate() -> None:
    ratings = _ratings().loc[_ratings()["team"] == "Manual"].copy()
    proposals = calib.build_external_calibration_proposals(ratings, _behavior(), _priors())

    updated = calib.apply_external_calibration_proposals(ratings, proposals)

    assert updated.iloc[0]["attack"] == 0.70
    assert proposals.iloc[0]["warning"].find("manual_existing_calibration_suggested") >= 0


def test_manual_reviewed_not_overwritten_by_default() -> None:
    ratings = _ratings().loc[_ratings()["team"] == "Reviewed"].copy()
    proposals = calib.build_external_calibration_proposals(ratings, _behavior(), _priors())

    updated = calib.apply_external_calibration_proposals(ratings, proposals, include_reviewed=False)

    assert updated.iloc[0]["attack"] == 0.72
    assert "manual_reviewed_preserved" in proposals.iloc[0]["warning"]


def test_write_modifies_only_allowed_rows() -> None:
    proposals = calib.build_external_calibration_proposals(_ratings(), _behavior(), _priors())

    updated = calib.apply_external_calibration_proposals(_ratings(), proposals)

    generated = updated.loc[updated["team"] == "Generated"].iloc[0]
    manual = updated.loc[updated["team"] == "Manual"].iloc[0]
    reviewed = updated.loc[updated["team"] == "Reviewed"].iloc[0]
    assert generated["data_quality"] == "external_benchmark_calibrated"
    assert generated["attack"] == 0.63
    assert manual["data_quality"] == "manual_csv"
    assert manual["attack"] == 0.70
    assert reviewed["data_quality"] == "manual_reviewed"


def test_external_benchmark_calibrated_rows_are_idempotent() -> None:
    ratings = _ratings().loc[_ratings()["team"] == "Generated"].copy()
    ratings.loc[:, "attack"] = 0.63
    ratings.loc[:, "defense"] = 0.64
    ratings.loc[:, "recent_form"] = 0.67
    ratings.loc[:, "data_quality"] = "external_benchmark_calibrated"

    proposals = calib.build_external_calibration_proposals(ratings, _behavior(), _priors())
    updated = calib.apply_external_calibration_proposals(ratings, proposals)

    assert proposals.iloc[0]["attack_delta"] == 0.0
    assert proposals.iloc[0]["data_quality_proposed"] == "external_benchmark_calibrated"
    assert updated.iloc[0]["attack"] == 0.63


def test_include_manual_existing_allows_manual_rows() -> None:
    ratings = _ratings().loc[_ratings()["team"] == "Manual"].copy()
    proposals = calib.build_external_calibration_proposals(ratings, _behavior(), _priors())

    updated = calib.apply_external_calibration_proposals(ratings, proposals, include_manual_existing=True)

    assert updated.iloc[0]["data_quality"] == "external_benchmark_calibrated"
    assert updated.iloc[0]["attack"] > 0.70


def test_calibration_scripts_run_without_crashing(monkeypatch, tmp_path, capsys) -> None:
    import scripts.audit_external_benchmark_calibration as audit_script
    import scripts.calibrate_ratings_from_external as calibrate_script

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(calibrate_script, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(calibrate_script, "DATA_DIR", data_dir)
    monkeypatch.setattr(calibrate_script, "load_team_ratings_csv", lambda *args, **kwargs: _ratings())
    monkeypatch.setattr(calibrate_script, "load_team_behavior", lambda: _behavior())
    monkeypatch.setattr(calibrate_script, "load_external_priors", lambda: _priors())
    monkeypatch.setattr(sys, "argv", ["calibrate_ratings_from_external.py", "--output", str(data_dir / "team_ratings_external_calibrated_proposed.csv")])

    calibrate_script.main()

    output = capsys.readouterr().out
    assert "External benchmark calibration" in output
    assert (data_dir / "team_ratings_external_calibrated_proposed.csv").exists()

    monkeypatch.setattr(audit_script, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(audit_script, "load_team_ratings_csv", lambda *args, **kwargs: _ratings())
    monkeypatch.setattr(audit_script, "load_team_behavior", lambda: _behavior())
    monkeypatch.setattr(audit_script, "load_external_priors", lambda: _priors())
    monkeypatch.setattr(audit_script, "today_iso", lambda: "2026-06-21")

    audit_script.main()

    output = capsys.readouterr().out
    assert "teams with external benchmark" in output
    assert (tmp_path / "reports" / "external_benchmark_calibration_audit_2026-06-21.md").exists()
