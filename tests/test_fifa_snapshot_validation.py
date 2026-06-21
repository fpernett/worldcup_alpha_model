from __future__ import annotations

import sys

import pandas as pd

from src.fifa_snapshot_validation import (
    build_missing_fifa_snapshot_template,
    validate_fifa_snapshot,
)


def _required(teams: list[str] | None = None) -> pd.DataFrame:
    teams = teams or ["United States", "South Korea", "Iran", "Turkey", "Ivory Coast", "DR Congo"]
    return pd.DataFrame(
        [{"team": team, "canonical_team": team, "required_for_model": True} for team in teams]
    )


def test_complete_snapshot_returns_ready_for_import() -> None:
    snapshot = pd.DataFrame(
        [
            {"team": "Argentina", "rank": 1, "points": 1877.27, "source": "snapshot", "date": "2026-06-21"},
            {"team": "Brazil", "rank": 6, "points": 1765.86, "source": "snapshot", "date": "2026-06-21"},
        ]
    )

    validation, summary = validate_fifa_snapshot(snapshot, _required(["Argentina", "Brazil"]), pd.DataFrame())

    assert bool(summary["ready_for_import"]) is True
    assert summary["required_teams"] == 2
    assert summary["matched_required_teams"] == 2
    assert set(validation["match_status"]) == {"matched"}


def test_missing_required_team_returns_not_ready() -> None:
    snapshot = pd.DataFrame([{"team": "Argentina", "fifa_rank": 1, "fifa_points": 1877.27}])

    validation, summary = validate_fifa_snapshot(snapshot, _required(["Argentina", "Brazil"]), pd.DataFrame())

    assert bool(summary["ready_for_import"]) is False
    assert summary["missing_required_teams"] == 1
    brazil = validation.loc[validation["canonical_team"] == "Brazil"].iloc[0]
    assert brazil["match_status"] == "missing_from_snapshot"


def test_missing_rank_is_detected() -> None:
    snapshot = pd.DataFrame([{"team": "Argentina", "fifa_points": 1877.27}])

    validation, summary = validate_fifa_snapshot(snapshot, _required(["Argentina"]), pd.DataFrame())

    assert bool(summary["ready_for_import"]) is False
    assert summary["matched_missing_rank"] == 1
    assert validation.iloc[0]["match_status"] == "matched_missing_rank"


def test_missing_points_is_detected() -> None:
    snapshot = pd.DataFrame([{"team": "Argentina", "fifa_rank": 1}])

    validation, summary = validate_fifa_snapshot(snapshot, _required(["Argentina"]), pd.DataFrame())

    assert bool(summary["ready_for_import"]) is False
    assert summary["matched_missing_points"] == 1
    assert validation.iloc[0]["match_status"] == "matched_missing_points"


def test_alias_matching_for_common_fifa_names() -> None:
    snapshot = pd.DataFrame(
        [
            {"ranked_team": "USA", "position": 11, "pts": 1700.0},
            {"ranked_team": "Korea Republic", "position": 22, "pts": 1550.0},
            {"ranked_team": "IR Iran", "position": 20, "pts": 1560.0},
            {"ranked_team": "Türkiye", "position": 24, "pts": 1540.0},
            {"ranked_team": "Côte d’Ivoire", "position": 35, "pts": 1480.0},
            {"ranked_team": "Congo DR", "position": 40, "pts": 1440.0},
        ]
    )

    validation, summary = validate_fifa_snapshot(snapshot, _required(), pd.DataFrame())

    assert bool(summary["ready_for_import"]) is True
    assert summary["matched_required_teams"] == 6
    assert set(validation["canonical_team"]) == {"United States", "South Korea", "Iran", "Turkey", "Ivory Coast", "DR Congo"}
    assert set(validation["match_status"]) == {"matched"}


def test_unused_fifa_rows_are_reported() -> None:
    snapshot = pd.DataFrame(
        [
            {"team": "Argentina", "fifa_rank": 1, "fifa_points": 1877.27},
            {"team": "Unused Team", "fifa_rank": 99, "fifa_points": 1200.0},
        ]
    )

    validation, summary = validate_fifa_snapshot(snapshot, _required(["Argentina"]), pd.DataFrame())

    assert summary["unused_fifa_rows"] == 1
    unused = validation.loc[validation["match_status"] == "unused_fifa_row"].iloc[0]
    assert unused["snapshot_team"] == "Unused Team"


def test_ambiguous_matches_are_reported() -> None:
    snapshot = pd.DataFrame(
        [
            {"team": "USA", "fifa_rank": 11, "fifa_points": 1700.0},
            {"team": "United States", "fifa_rank": 12, "fifa_points": 1699.0},
        ]
    )

    validation, summary = validate_fifa_snapshot(snapshot, _required(["United States"]), pd.DataFrame())

    assert bool(summary["ready_for_import"]) is False
    assert summary["ambiguous_matches"] == 1
    assert validation.iloc[0]["match_status"] == "ambiguous_match"


def test_missing_output_template_contains_missing_and_incomplete_rows() -> None:
    snapshot = pd.DataFrame(
        [
            {"team": "Argentina", "fifa_rank": 1},
        ]
    )
    validation, _summary = validate_fifa_snapshot(snapshot, _required(["Argentina", "Brazil"]), pd.DataFrame())

    template = build_missing_fifa_snapshot_template(validation)

    assert list(template.columns) == ["team", "fifa_rank", "fifa_points", "source", "last_updated", "notes"]
    assert set(template["team"]) == {"Argentina", "Brazil"}
    argentina = template.loc[template["team"] == "Argentina"].iloc[0]
    assert float(argentina["fifa_rank"]) == 1.0


def test_validation_script_runs_and_writes_missing_output(monkeypatch, tmp_path, capsys) -> None:
    import scripts.validate_fifa_snapshot as script

    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True)
    snapshot_path = raw_dir / "fifa_rankings_snapshot.csv"
    missing_output = raw_dir / "fifa_snapshot_missing_teams.csv"
    snapshot_path.write_text(
        "\n".join(
            [
                "country,rank,total_points,source,last_updated",
                "USA,11,1700,synthetic,2026-06-21",
                "Unused Team,99,1200,synthetic,2026-06-21",
            ]
        ),
        encoding="utf-8",
    )
    required = _required(["United States", "Brazil"])

    monkeypatch.setattr(script, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(script, "DATA_DIR", data_dir)
    monkeypatch.setattr(script, "_load_required_teams", lambda args: required)
    monkeypatch.setattr(script, "today_iso", lambda: "2026-06-21")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_fifa_snapshot.py",
            "--input",
            str(snapshot_path),
            "--missing-output",
            str(missing_output),
        ],
    )

    script.main()

    output = capsys.readouterr().out
    validation_report = pd.read_csv(tmp_path / "reports" / "fifa_snapshot_validation_2026-06-21.csv")
    missing = pd.read_csv(missing_output)
    assert "Ready for import: no" in output
    assert "Missing required teams: 1" in output
    assert "unused_fifa_row" in set(validation_report["match_status"])
    assert list(missing.columns) == ["team", "fifa_rank", "fifa_points", "source", "last_updated", "notes"]
    assert "Brazil" in set(missing["team"])
