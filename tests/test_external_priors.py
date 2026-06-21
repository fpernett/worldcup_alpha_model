from __future__ import annotations

import sys

import pandas as pd

from src import external_priors


def _ratings(data_quality: str = "generated_from_behavior") -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "team": "Argentina",
                "elo": 1850,
                "attack": 0.70,
                "defense": 0.68,
                "recent_form": 0.66,
                "data_quality": data_quality,
                "last_updated": "2026-06-21",
                "notes": "rating",
            }
        ]
    )


def _proposals() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "team": "Argentina",
                "rating_status": "generated_from_behavior",
                "reviewed_attack": 0.74,
                "reviewed_defense": 0.70,
                "reviewed_recent_form": 0.72,
            }
        ]
    )


def test_external_prior_loading_with_blank_rows(tmp_path) -> None:
    path = tmp_path / "priors.csv"
    path.write_text(
        "team,reference_attack,reference_defense,reference_recent_form,reference_overall_strength,source,last_updated,notes\n"
        "Argentina,,,,,,,placeholder\n",
        encoding="utf-8",
    )

    priors = external_priors.load_external_priors(path)

    assert list(priors.columns) == external_priors.EXTERNAL_PRIOR_COLUMNS
    assert priors.iloc[0]["team"] == "Argentina"
    assert pd.isna(priors.iloc[0]["reference_attack"])


def test_missing_external_prior_warning() -> None:
    priors = pd.DataFrame([{"team": "Argentina"}])

    comparison = external_priors.compare_ratings_to_external_priors(_ratings(), _proposals(), priors)
    warning = comparison.iloc[0]["overall_warning"]

    assert "missing external prior" in warning
    assert "generated rating without external check" in warning
    assert comparison.iloc[0]["recommended_action"] == "fill external prior before approval"


def test_large_disagreement_warning() -> None:
    priors = pd.DataFrame(
        [
            {
                "team": "Argentina",
                "reference_attack": 0.50,
                "reference_defense": 0.88,
                "reference_recent_form": 0.70,
            }
        ]
    )

    comparison = external_priors.compare_ratings_to_external_priors(_ratings(), _proposals(), priors)
    warning = comparison.iloc[0]["overall_warning"]

    assert comparison.iloc[0]["attack_disagreement"] == 0.24
    assert comparison.iloc[0]["defense_disagreement"] == -0.18
    assert "large attack disagreement" in warning
    assert "large defense disagreement" in warning
    assert "proposal above external prior" in warning
    assert "proposal below external prior" in warning


def test_manual_review_worksheet_creation() -> None:
    audit = pd.DataFrame(
        [{"team": "Argentina", "rating_status": "generated_from_behavior", "priority": "high"}]
    )
    priors = pd.DataFrame(
        [
            {
                "team": "Argentina",
                "reference_attack": 0.72,
                "reference_defense": 0.68,
                "reference_recent_form": 0.70,
            }
        ]
    )

    sheet = external_priors.build_manual_rating_review_sheet(_ratings(), audit, _proposals(), priors, ["Argentina"])
    row = sheet.iloc[0]

    assert row["recommended_attack"] == 0.73
    assert row["recommended_defense"] == 0.69
    assert row["recommended_recent_form"] == 0.712
    assert row["review_status"] == "pending_human_approval"


def test_approved_only_write_behavior() -> None:
    sheet = pd.DataFrame(
        [
            {
                "team": "Argentina",
                "recommended_attack": 0.76,
                "recommended_defense": 0.74,
                "recommended_recent_form": 0.73,
                "review_status": "approved",
                "human_notes": "Reviewed externally.",
            }
        ]
    )

    updated = external_priors.apply_approved_review_sheet(_ratings(), sheet)
    row = updated.iloc[0]

    assert row["attack"] == 0.76
    assert row["defense"] == 0.74
    assert row["recent_form"] == 0.73
    assert row["data_quality"] == "manual_reviewed"
    assert "Reviewed externally" in row["notes"]


def test_non_approved_rows_are_not_written() -> None:
    sheet = pd.DataFrame(
        [
            {
                "team": "Argentina",
                "recommended_attack": 0.40,
                "recommended_defense": 0.40,
                "recommended_recent_form": 0.40,
                "review_status": "needs_review",
            }
        ]
    )

    updated = external_priors.apply_approved_review_sheet(_ratings(), sheet)

    assert updated.iloc[0]["attack"] == 0.70
    assert updated.iloc[0]["data_quality"] == "generated_from_behavior"


def test_manual_reviewed_rows_are_not_overwritten() -> None:
    sheet = pd.DataFrame(
        [
            {
                "team": "Argentina",
                "recommended_attack": 0.40,
                "recommended_defense": 0.40,
                "recommended_recent_form": 0.40,
                "review_status": "approved",
            }
        ]
    )

    updated = external_priors.apply_approved_review_sheet(_ratings("manual_reviewed"), sheet)

    assert updated.iloc[0]["attack"] == 0.70
    assert updated.iloc[0]["data_quality"] == "manual_reviewed"


def test_audit_external_priors_script_runs(monkeypatch, tmp_path, capsys) -> None:
    import scripts.audit_external_priors as script

    comparison = pd.DataFrame(
        [
            {
                "team": "Argentina",
                "overall_warning": "missing external prior; generated rating without external check",
            }
        ]
    )
    monkeypatch.setattr(script, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(script, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(script, "load_team_ratings_csv", lambda: _ratings())
    monkeypatch.setattr(script, "load_review_proposals", lambda path: _proposals())
    monkeypatch.setattr(script, "load_external_priors", lambda: pd.DataFrame())
    monkeypatch.setattr(script, "compare_ratings_to_external_priors", lambda *args, **kwargs: comparison)
    monkeypatch.setattr(
        script,
        "external_prior_review_summary",
        lambda *args, **kwargs: {
            "teams_with_external_priors": 0,
            "teams_missing_external_priors": 1,
            "generated_without_external_check": 1,
            "large_disagreement_warnings": 0,
            "worksheet_rows": 0,
            "approved_review_rows": 0,
        },
    )
    monkeypatch.setattr(script, "today_iso", lambda: "2026-06-21")
    monkeypatch.setattr(sys, "argv", ["audit_external_priors.py"])

    script.main()

    output = capsys.readouterr().out
    assert "teams missing external priors: Argentina" in output
    assert (tmp_path / "reports" / "external_prior_audit_2026-06-21.md").exists()


def test_build_review_sheet_script_runs_without_modifying_source(monkeypatch, tmp_path, capsys) -> None:
    import scripts.build_rating_review_sheet as script

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    ratings_path = data_dir / "team_ratings.csv"
    _ratings().to_csv(ratings_path, index=False)
    output_path = data_dir / "manual_rating_review_sheet.csv"

    monkeypatch.setattr(script, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(script, "DATA_DIR", data_dir)
    monkeypatch.setattr(script, "load_team_ratings_csv", lambda path=None: pd.read_csv(ratings_path))
    monkeypatch.setattr(script, "load_team_behavior", lambda: pd.DataFrame())
    monkeypatch.setattr(script, "load_historical_matches", lambda use_cache=False: pd.DataFrame())
    monkeypatch.setattr(script, "load_review_proposals", lambda path: _proposals())
    monkeypatch.setattr(script, "load_external_priors", lambda: pd.DataFrame())
    monkeypatch.setattr(
        script,
        "audit_generated_ratings",
        lambda *args, **kwargs: pd.DataFrame(
            [{"team": "Argentina", "rating_status": "generated_from_behavior", "priority": "high"}]
        ),
    )
    monkeypatch.setattr(sys, "argv", ["build_rating_review_sheet.py", "--teams", "Argentina", "--output", str(output_path)])

    script.main()

    output = capsys.readouterr().out
    unchanged = pd.read_csv(ratings_path)
    sheet = pd.read_csv(output_path)
    assert "was not modified" in output
    assert unchanged.iloc[0]["data_quality"] == "generated_from_behavior"
    assert sheet.iloc[0]["team"] == "Argentina"
