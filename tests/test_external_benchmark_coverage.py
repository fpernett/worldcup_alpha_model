from __future__ import annotations

import sys

import pandas as pd

from src.external_benchmark_calibration import (
    audit_external_benchmark_coverage,
    build_external_strength_template,
    external_benchmark_coverage_summary,
)


def _required() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "team": "Alpha",
                "canonical_team": "Alpha",
                "source_fixtures": True,
                "source_historical": True,
                "source_behavior": True,
                "source_backtest": False,
                "required_for_model": True,
            },
            {
                "team": "Beta",
                "canonical_team": "Beta",
                "source_fixtures": False,
                "source_historical": True,
                "source_behavior": True,
                "source_backtest": True,
                "required_for_model": True,
            },
            {
                "team": "Gamma",
                "canonical_team": "Gamma",
                "source_fixtures": False,
                "source_historical": True,
                "source_behavior": True,
                "source_backtest": False,
                "required_for_model": True,
            },
            {
                "team": "History Only",
                "canonical_team": "History Only",
                "source_fixtures": False,
                "source_historical": True,
                "source_behavior": False,
                "source_backtest": False,
                "required_for_model": False,
            },
        ]
    )


def _priors() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "team": "Alpha",
                "reference_attack": pd.NA,
                "reference_defense": pd.NA,
                "reference_recent_form": pd.NA,
                "reference_overall_strength": 0.82,
                "source": "snapshot",
                "last_updated": "2026-06-21",
                "notes": "",
            },
            {
                "team": "Beta",
                "reference_attack": pd.NA,
                "reference_defense": pd.NA,
                "reference_recent_form": pd.NA,
                "reference_overall_strength": pd.NA,
                "source": "",
                "last_updated": "",
                "notes": "",
            },
            {
                "team": "Gamma",
                "reference_attack": pd.NA,
                "reference_defense": pd.NA,
                "reference_recent_form": pd.NA,
                "reference_overall_strength": pd.NA,
                "source": "",
                "last_updated": "",
                "notes": "",
            },
        ]
    )


def _ratings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "team": "Alpha",
                "elo": 1800,
                "attack": 0.70,
                "defense": 0.70,
                "recent_form": 0.70,
                "fifa_rank_proxy": 20,
                "training_temp_c": 20,
                "training_humidity_pct": 60,
                "data_quality": "external_benchmark_calibrated",
                "last_updated": "2026-06-21",
                "notes": "",
            },
            {
                "team": "Beta",
                "elo": 1700,
                "attack": 0.55,
                "defense": 0.55,
                "recent_form": 0.55,
                "fifa_rank_proxy": 70,
                "training_temp_c": 20,
                "training_humidity_pct": 60,
                "data_quality": "generated_from_behavior",
                "last_updated": "2026-06-21",
                "notes": "",
            },
            {
                "team": "Gamma",
                "elo": 1600,
                "attack": 0.50,
                "defense": 0.50,
                "recent_form": 0.50,
                "fifa_rank_proxy": 100,
                "training_temp_c": 20,
                "training_humidity_pct": 60,
                "data_quality": "manual_csv",
                "last_updated": "2026-06-21",
                "notes": "",
            },
        ]
    )


def test_missing_external_benchmark_detection_and_summary() -> None:
    coverage = audit_external_benchmark_coverage(_required(), _priors(), _ratings())
    summary = external_benchmark_coverage_summary(coverage)

    assert summary["required_teams"] == 3
    assert summary["teams_with_external_benchmark"] == 1
    assert summary["teams_missing_external_benchmark"] == 2
    beta = coverage.loc[coverage["team"] == "Beta"].iloc[0]
    assert bool(beta["has_external_benchmark"]) is False
    assert beta["priority"] == "high"
    assert beta["recommended_action"] == "fill FIFA rank/points or public Elo snapshot"


def test_priority_classification_for_less_relevant_required_team() -> None:
    coverage = audit_external_benchmark_coverage(_required(), _priors(), _ratings())

    gamma = coverage.loc[coverage["team"] == "Gamma"].iloc[0]
    assert gamma["priority"] == "medium"
    assert gamma["recommended_action"] == "fill benchmark when available"


def test_template_creation_includes_only_missing_by_default() -> None:
    coverage = audit_external_benchmark_coverage(_required(), _priors(), _ratings())

    template = build_external_strength_template(coverage)

    assert list(template.columns) == ["team", "fifa_rank", "fifa_points", "external_elo", "source", "last_updated", "notes"]
    assert set(template["team"]) == {"Beta", "Gamma"}
    assert "Alpha" not in set(template["team"])


def test_all_required_teams_option_includes_covered_team() -> None:
    coverage = audit_external_benchmark_coverage(_required(), _priors(), _ratings())

    template = build_external_strength_template(coverage, all_required_teams=True)

    assert set(template["team"]) == {"Alpha", "Beta", "Gamma"}


def test_import_script_does_not_overwrite_existing_priors_without_write(tmp_path, monkeypatch) -> None:
    import scripts.import_external_priors as script

    data = tmp_path / "data"
    raw = data / "raw"
    raw.mkdir(parents=True)
    input_path = raw / "external_team_strength_missing_template.csv"
    output_path = data / "team_rating_external_priors.csv"
    input_path.write_text(
        "\n".join(
            [
                "team,fifa_rank,fifa_points,external_elo,source,last_updated,notes",
                "Beta,10,1800,,manual snapshot,2026-06-21,test",
            ]
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "team": "Beta",
                "reference_attack": pd.NA,
                "reference_defense": pd.NA,
                "reference_recent_form": pd.NA,
                "reference_overall_strength": 0.50,
                "source": "old",
                "last_updated": "2026-01-01",
                "notes": "",
            }
        ]
    ).to_csv(output_path, index=False)

    monkeypatch.setattr(script, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(script, "load_team_ratings_csv", lambda *args, **kwargs: _ratings())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "import_external_priors.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
    )

    script.main()

    unchanged = pd.read_csv(output_path)
    proposed = pd.read_csv(data / "team_rating_external_priors_proposed.csv")
    assert float(unchanged.loc[0, "reference_overall_strength"]) == 0.50
    assert float(proposed.loc[0, "reference_overall_strength"]) != 0.50


def test_coverage_scripts_run_without_crashing(monkeypatch, tmp_path, capsys) -> None:
    import scripts.audit_external_benchmark_coverage as audit_script
    import scripts.build_external_team_strength_template as template_script

    monkeypatch.setattr(audit_script, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(audit_script, "_load_required_teams", lambda args: _required())
    monkeypatch.setattr(audit_script, "load_external_priors", lambda: _priors())
    monkeypatch.setattr(audit_script, "load_team_ratings_csv", lambda: _ratings())
    monkeypatch.setattr(audit_script, "today_iso", lambda: "2026-06-21")
    monkeypatch.setattr(sys, "argv", ["audit_external_benchmark_coverage.py"])

    audit_script.main()

    output = capsys.readouterr().out
    assert "required teams: 3" in output
    assert (tmp_path / "reports" / "external_benchmark_coverage_audit_2026-06-21.md").exists()

    output_path = tmp_path / "data" / "raw" / "external_team_strength_missing_template.csv"
    monkeypatch.setattr(template_script, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(template_script, "_load_required_teams", lambda args: _required())
    monkeypatch.setattr(template_script, "load_external_priors", lambda: _priors())
    monkeypatch.setattr(template_script, "load_team_ratings_csv", lambda: _ratings())
    monkeypatch.setattr(sys, "argv", ["build_external_team_strength_template.py", "--output", str(output_path)])

    template_script.main()

    output = capsys.readouterr().out
    assert "template rows written: 2" in output
    assert output_path.exists()
