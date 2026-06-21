from __future__ import annotations

import sys

import pandas as pd

from src import external_prior_import


def _existing_priors() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "team": "Argentina",
                "reference_attack": pd.NA,
                "reference_defense": pd.NA,
                "reference_recent_form": pd.NA,
                "reference_overall_strength": pd.NA,
                "source": "",
                "last_updated": "",
                "notes": "",
            },
            {
                "team": "Brazil",
                "reference_attack": 0.70,
                "reference_defense": 0.72,
                "reference_recent_form": 0.68,
                "reference_overall_strength": 0.74,
                "source": "existing",
                "last_updated": "2026-01-01",
                "notes": "existing",
            },
        ]
    )


def test_fifa_rank_scaling() -> None:
    assert external_prior_import.scale_fifa_rank_to_strength(1) == 0.9
    assert external_prior_import.scale_fifa_rank_to_strength(210) == 0.35
    assert external_prior_import.scale_fifa_rank_to_strength(999) == 0.35


def test_fifa_points_scaling() -> None:
    assert external_prior_import.scale_fifa_points_to_strength(900) == 0.35
    assert external_prior_import.scale_fifa_points_to_strength(1900) == 0.9
    assert external_prior_import.scale_fifa_points_to_strength(1400) == 0.625


def test_external_elo_scaling() -> None:
    assert external_prior_import.scale_external_elo_to_strength(1200) == 0.35
    assert external_prior_import.scale_external_elo_to_strength(2200) == 0.9
    assert external_prior_import.scale_external_elo_to_strength(1700) == 0.625


def test_combining_sources() -> None:
    combined = external_prior_import.combine_external_strengths(fifa_rank=1, fifa_points=1900, external_elo=2200)
    assert combined == 0.9

    mixed = external_prior_import.combine_external_strengths(fifa_points=1400, external_elo=1700)
    assert mixed == 0.625


def test_missing_data_handling() -> None:
    assert pd.isna(external_prior_import.combine_external_strengths())

    input_df = pd.DataFrame([{"team": "Argentina", "source": "manual"}])
    proposed, diagnostics = external_prior_import.build_external_prior_updates(input_df, _existing_priors())

    assert pd.isna(proposed.iloc[0]["reference_overall_strength"])
    assert bool(diagnostics.iloc[0]["update_ready"]) is False
    assert "missing rank/Elo data" in diagnostics.iloc[0]["warning"]


def test_optional_attack_defense_form_are_only_set_when_supplied() -> None:
    input_df = pd.DataFrame(
        [
            {
                "team": "Argentina",
                "fifa_rank": 1,
                "reference_attack": 0.81,
                "reference_defense": 0.77,
            }
        ]
    )

    proposed, diagnostics = external_prior_import.build_external_prior_updates(input_df, _existing_priors())

    assert proposed.iloc[0]["reference_attack"] == 0.81
    assert proposed.iloc[0]["reference_defense"] == 0.77
    assert pd.isna(proposed.iloc[0]["reference_recent_form"])
    assert bool(diagnostics.iloc[0]["update_ready"]) is True


def test_write_updates_only_matching_ready_teams() -> None:
    input_df = pd.DataFrame(
        [
            {"team": "Argentina", "fifa_rank": 1, "source": "rank source"},
            {"team": "Missing Team", "fifa_rank": 2, "source": "rank source"},
            {"team": "Brazil", "source": "blank source"},
        ]
    )
    proposed, diagnostics = external_prior_import.build_external_prior_updates(input_df, _existing_priors())

    updated = external_prior_import.merge_external_prior_updates(_existing_priors(), proposed, diagnostics)

    argentina = updated.loc[updated["team"] == "Argentina"].iloc[0]
    brazil = updated.loc[updated["team"] == "Brazil"].iloc[0]
    assert argentina["reference_overall_strength"] == 0.9
    assert argentina["source"] == "rank source"
    assert brazil["reference_overall_strength"] == 0.74
    assert "Missing Team" not in set(updated["team"])


def test_merge_casts_blank_text_columns_before_string_assignment() -> None:
    existing = pd.DataFrame(
        {
            "team": ["Argentina"],
            "reference_attack": [float("nan")],
            "reference_defense": [float("nan")],
            "reference_recent_form": [float("nan")],
            "reference_overall_strength": [float("nan")],
            "source": [float("nan")],
            "last_updated": [float("nan")],
            "notes": [float("nan")],
        }
    )
    assert str(existing["source"].dtype) == "float64"
    proposed = pd.DataFrame(
        [
            {
                "team": "Argentina",
                "reference_overall_strength": 0.893,
                "source": "FIFA ranking snapshot via Sofascore",
                "last_updated": "2026-06-21",
                "notes": "External overall-strength prior only",
            }
        ]
    )
    diagnostics = pd.DataFrame([{"canonical_team": "Argentina", "update_ready": True}])

    updated = external_prior_import.merge_external_prior_updates(existing, proposed, diagnostics)

    assert updated.loc[0, "source"] == "FIFA ranking snapshot via Sofascore"
    assert updated.loc[0, "last_updated"] == "2026-06-21"
    assert updated.loc[0, "reference_overall_strength"] == 0.893
    assert pd.api.types.is_numeric_dtype(updated["reference_overall_strength"])


def test_generated_rating_disagreement() -> None:
    proposed = pd.DataFrame([{"team": "Argentina", "reference_overall_strength": 0.40}])
    ratings = pd.DataFrame([{"team": "Argentina", "attack": 0.75, "defense": 0.72, "recent_form": 0.70}])

    disagreements = external_prior_import.calculate_generated_rating_disagreements(proposed, ratings)

    assert disagreements.iloc[0]["warning"] == "large disagreement with generated ratings"


def test_import_script_runs_without_write(monkeypatch, tmp_path, capsys) -> None:
    import scripts.import_external_priors as script

    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True)
    input_path = raw_dir / "external_team_strength.csv"
    output_path = data_dir / "team_rating_external_priors.csv"
    input_path.write_text("team,fifa_rank,fifa_points,external_elo,source,last_updated,notes\nArgentina,1,,,manual,2026-06-21,\n", encoding="utf-8")
    _existing_priors().to_csv(output_path, index=False)

    monkeypatch.setattr(script, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(script, "DATA_DIR", data_dir)
    monkeypatch.setattr(script, "load_team_ratings_csv", lambda: pd.DataFrame([{"team": "Argentina", "attack": 0.7, "defense": 0.7, "recent_form": 0.7}]))
    monkeypatch.setattr(sys, "argv", ["import_external_priors.py", "--input", str(input_path), "--output", str(output_path)])

    script.main()

    output = capsys.readouterr().out
    canonical = pd.read_csv(output_path)
    proposed = pd.read_csv(data_dir / "team_rating_external_priors_proposed.csv")
    assert "was not modified" in output
    assert pd.isna(canonical.loc[canonical["team"] == "Argentina", "reference_overall_strength"].iloc[0])
    assert proposed.loc[proposed["team"] == "Argentina", "reference_overall_strength"].iloc[0] == 0.9


def test_import_script_write_updates_matching_teams(monkeypatch, tmp_path, capsys) -> None:
    import scripts.import_external_priors as script

    data_dir = tmp_path / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True)
    input_path = raw_dir / "external_team_strength.csv"
    output_path = data_dir / "team_rating_external_priors.csv"
    input_path.write_text("team,fifa_rank,fifa_points,external_elo,source,last_updated,notes\nArgentina,1,,,manual,2026-06-21,\n", encoding="utf-8")
    _existing_priors().to_csv(output_path, index=False)

    monkeypatch.setattr(script, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(script, "DATA_DIR", data_dir)
    monkeypatch.setattr(script, "load_team_ratings_csv", lambda: pd.DataFrame())
    monkeypatch.setattr(sys, "argv", ["import_external_priors.py", "--input", str(input_path), "--output", str(output_path), "--write"])

    script.main()

    output = capsys.readouterr().out
    canonical = pd.read_csv(output_path)
    assert "teams updated: 1" in output
    assert canonical.loc[canonical["team"] == "Argentina", "reference_overall_strength"].iloc[0] == 0.9
