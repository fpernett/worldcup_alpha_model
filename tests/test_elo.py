from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.elo import add_elo_to_historical_matches, calculate_rolling_elo, elo_k_factor, expected_result


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _long_match(
    match_id: str,
    date_utc: str,
    home: str,
    away: str,
    home_goals: int,
    away_goals: int,
    competition_type: str = "other",
    is_neutral: int = 1,
) -> list[dict[str, object]]:
    common = {
        "match_id": match_id,
        "date_utc": date_utc,
        "competition": competition_type,
        "competition_type": competition_type,
        "is_neutral": is_neutral,
        "source": "test",
        "last_updated": "2026-06-21",
    }
    return [
        {
            **common,
            "team": home,
            "opponent": away,
            "is_home": 1,
            "team_goals": home_goals,
            "opponent_goals": away_goals,
        },
        {
            **common,
            "team": away,
            "opponent": home,
            "is_home": 0,
            "team_goals": away_goals,
            "opponent_goals": home_goals,
        },
    ]


def test_elo_updates_after_win_loss_and_draw() -> None:
    matches = pd.DataFrame(
        _long_match("m1", "2024-01-01", "Alpha", "Beta", 2, 0, "world_cup")
        + _long_match("m2", "2024-02-01", "Alpha", "Beta", 1, 1, "world_cup")
    )

    elo = calculate_rolling_elo(matches)
    alpha_first = elo.loc[(elo["match_id"] == "m1") & (elo["team"] == "Alpha")].iloc[0]
    beta_first = elo.loc[(elo["match_id"] == "m1") & (elo["team"] == "Beta")].iloc[0]
    alpha_draw = elo.loc[(elo["match_id"] == "m2") & (elo["team"] == "Alpha")].iloc[0]
    beta_draw = elo.loc[(elo["match_id"] == "m2") & (elo["team"] == "Beta")].iloc[0]

    assert alpha_first["team_elo_post"] > alpha_first["team_elo_pre"]
    assert beta_first["team_elo_post"] < beta_first["team_elo_pre"]
    assert alpha_draw["elo_change"] < 0
    assert beta_draw["elo_change"] > 0


def test_neutral_site_home_advantage_disabled() -> None:
    neutral = expected_result(1500, 1500, is_neutral=True)
    non_neutral = expected_result(1500, 1500, is_neutral=False)

    assert neutral == 0.5
    assert non_neutral > neutral


def test_stronger_competition_has_larger_k_factor() -> None:
    assert elo_k_factor("world_cup") > elo_k_factor("friendly")
    assert elo_k_factor("world_cup_qualifier") > elo_k_factor("nations_league")


def test_long_format_rows_receive_pre_match_elo() -> None:
    matches = pd.DataFrame(_long_match("m1", "2024-01-01", "Alpha", "Beta", 2, 0, "world_cup"))

    enhanced = add_elo_to_historical_matches(matches)

    assert pd.to_numeric(enhanced["team_elo_pre"], errors="coerce").notna().all()
    assert pd.to_numeric(enhanced["opponent_elo"], errors="coerce").notna().all()
    alpha = enhanced.loc[enhanced["team"] == "Alpha"].iloc[0]
    beta = enhanced.loc[enhanced["team"] == "Beta"].iloc[0]
    assert alpha["team_elo_pre"] == beta["opponent_elo"]
    assert beta["team_elo_pre"] == alpha["opponent_elo"]


def test_rebuild_elo_script_runs_without_crashing(tmp_path) -> None:
    input_path = tmp_path / "historical_matches.csv"
    output_path = tmp_path / "historical_matches_out.csv"
    pd.DataFrame(_long_match("m1", "2024-01-01", "Alpha", "Beta", 2, 0, "world_cup")).to_csv(input_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/rebuild_elo.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert output_path.exists()
