from __future__ import annotations

import sys

import pandas as pd

from src.fifa_ranking_import import (
    build_external_strength_from_fifa_matches,
    load_fifa_ranking_snapshot,
    match_fifa_rankings_to_required_teams,
)


def _required() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"team": "United States", "canonical_team": "United States", "required_for_model": True},
            {"team": "South Korea", "canonical_team": "South Korea", "required_for_model": True},
            {"team": "Ivory Coast", "canonical_team": "Ivory Coast", "required_for_model": True},
            {"team": "DR Congo", "canonical_team": "DR Congo", "required_for_model": True},
            {"team": "England", "canonical_team": "England", "required_for_model": True},
            {"team": "Missing Team", "canonical_team": "Missing Team", "required_for_model": True},
        ]
    )


def test_load_fifa_ranking_snapshot_detects_flexible_columns(tmp_path) -> None:
    path = tmp_path / "snapshot.csv"
    path.write_text(
        "\n".join(
            [
                "country,position,total_points,date",
                "USA,12,1670.5,2026-06-21",
                "Korea Republic,23,1530.25,2026-06-21",
            ]
        ),
        encoding="utf-8",
    )

    snapshot = load_fifa_ranking_snapshot(path)

    assert list(snapshot.columns) == ["team", "fifa_rank", "fifa_points", "source", "last_updated"]
    assert snapshot.loc[0, "team"] == "USA"
    assert snapshot.loc[0, "fifa_rank"] == 12.0
    assert snapshot.loc[0, "fifa_points"] == 1670.5
    assert snapshot.loc[0, "last_updated"] == "2026-06-21"


def test_team_alias_matching_for_common_fifa_names() -> None:
    snapshot = pd.DataFrame(
        [
            {"team": "USA", "fifa_rank": 11, "fifa_points": 1700, "source": "snapshot", "last_updated": "2026-06-21"},
            {"team": "Korea Republic", "fifa_rank": 22, "fifa_points": 1550, "source": "snapshot", "last_updated": "2026-06-21"},
            {"team": "Côte d’Ivoire", "fifa_rank": 35, "fifa_points": 1480, "source": "snapshot", "last_updated": "2026-06-21"},
            {"team": "Congo DR", "fifa_rank": 40, "fifa_points": 1440, "source": "snapshot", "last_updated": "2026-06-21"},
            {"team": "England", "fifa_rank": 5, "fifa_points": 1800, "source": "snapshot", "last_updated": "2026-06-21"},
        ]
    )

    matches = match_fifa_rankings_to_required_teams(snapshot, _required(), pd.DataFrame(columns=["alias", "canonical"]))

    by_team = matches.set_index("canonical_team")
    assert by_team.loc["United States", "match_status"] == "matched"
    assert by_team.loc["South Korea", "match_status"] == "matched"
    assert by_team.loc["Ivory Coast", "match_status"] == "matched"
    assert by_team.loc["DR Congo", "match_status"] == "matched"
    assert by_team.loc["Missing Team", "match_status"] == "unmatched_required_team"


def test_missing_rank_or_points_warning() -> None:
    snapshot = pd.DataFrame(
        [{"team": "England", "fifa_rank": 5, "fifa_points": pd.NA, "source": "snapshot", "last_updated": "2026-06-21"}]
    )
    required = pd.DataFrame([{"team": "England", "canonical_team": "England", "required_for_model": True}])

    matches = match_fifa_rankings_to_required_teams(snapshot, required, pd.DataFrame(columns=["alias", "canonical"]))

    assert matches.iloc[0]["match_status"] == "missing_rank_or_points"
    assert "rank or points is missing" in matches.iloc[0]["match_note"]


def test_external_strength_output_format() -> None:
    snapshot = pd.DataFrame(
        [{"team": "USA", "fifa_rank": 11, "fifa_points": 1700, "source": "snapshot", "last_updated": "2026-06-21"}]
    )
    required = pd.DataFrame([{"team": "United States", "canonical_team": "United States", "required_for_model": True}])
    matches = match_fifa_rankings_to_required_teams(snapshot, required, pd.DataFrame(columns=["alias", "canonical"]))

    output = build_external_strength_from_fifa_matches(matches)

    assert list(output.columns) == ["team", "fifa_rank", "fifa_points", "external_elo", "source", "last_updated", "notes"]
    assert output.iloc[0]["team"] == "United States"
    assert output.iloc[0]["fifa_rank"] == 11.0
    assert output.iloc[0]["fifa_points"] == 1700.0


def test_import_script_runs_on_small_synthetic_snapshot(monkeypatch, tmp_path, capsys) -> None:
    import scripts.import_fifa_rankings as script

    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True)
    snapshot_path = raw_dir / "fifa_rankings_snapshot.csv"
    output_path = raw_dir / "external_team_strength_from_fifa.csv"
    snapshot_path.write_text(
        "\n".join(
            [
                "ranked_team,rank,pts,source,last_updated",
                "USA,11,1700,synthetic,2026-06-21",
                "Korea Republic,22,1550,synthetic,2026-06-21",
                "Unused Team,99,1200,synthetic,2026-06-21",
            ]
        ),
        encoding="utf-8",
    )
    required = pd.DataFrame(
        [
            {"team": "United States", "canonical_team": "United States", "required_for_model": True},
            {"team": "South Korea", "canonical_team": "South Korea", "required_for_model": True},
        ]
    )

    monkeypatch.setattr(script, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(script, "DATA_DIR", data_dir)
    monkeypatch.setattr(script, "_load_required_teams", lambda args: required)
    monkeypatch.setattr(script, "today_iso", lambda: "2026-06-21")
    monkeypatch.setattr(sys, "argv", ["import_fifa_rankings.py", "--input", str(snapshot_path), "--output", str(output_path)])

    script.main()

    output = capsys.readouterr().out
    strength = pd.read_csv(output_path)
    report = pd.read_csv(tmp_path / "reports" / "fifa_ranking_import_2026-06-21.csv")
    assert "matched required teams: 2" in output
    assert set(strength["team"]) == {"United States", "South Korea"}
    assert "unused_fifa_row" in set(report["match_status"])
