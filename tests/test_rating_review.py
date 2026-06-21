from __future__ import annotations

import sys

import pandas as pd

from src import rating_review


def _ratings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "team": "Argentina",
                "elo": 1850,
                "attack": 0.70,
                "defense": 0.68,
                "recent_form": 0.66,
                "data_quality": "generated_from_behavior",
                "notes": "Generated",
            },
            {
                "team": "England",
                "elo": 1860,
                "attack": 0.76,
                "defense": 0.74,
                "recent_form": 0.70,
                "data_quality": "manual_csv",
                "notes": "Manual",
            },
            {
                "team": "Brazil",
                "elo": 1900,
                "attack": 0.82,
                "defense": 0.79,
                "recent_form": 0.72,
                "data_quality": "manual_reviewed",
                "notes": "Reviewed",
            },
        ]
    )


def _behavior() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "team": "Argentina",
                "attack_index_final": 0.82,
                "defense_index_final": 0.78,
                "recent_form_index": 0.74,
                "matches_used_recent": 35,
                "mean_opponent_elo_recent": 1710,
            },
            {
                "team": "England",
                "attack_index_final": 0.77,
                "defense_index_final": 0.73,
                "recent_form_index": 0.69,
                "matches_used_recent": 30,
                "mean_opponent_elo_recent": 1740,
            },
        ]
    )


def test_generated_rating_detection(monkeypatch) -> None:
    monkeypatch.setattr(rating_review, "read_csv_with_columns", lambda path, columns: pd.DataFrame(columns=columns))
    backtest = pd.DataFrame([{"home": "Argentina", "away": "England"}])

    audit = rating_review.audit_generated_ratings(_ratings(), _behavior(), pd.DataFrame(), backtest)
    argentina = audit.loc[audit["team"] == "Argentina"].iloc[0]
    england = audit.loc[audit["team"] == "England"].iloc[0]

    assert argentina["rating_status"] == "generated_from_behavior"
    assert argentina["priority"] == "high"
    assert "generated rating requires review" in argentina["warning"]
    assert england["rating_status"] == "manual_existing"


def test_priority_classification() -> None:
    assert rating_review.classify_review_priority("France", "generated_from_behavior") == "high"
    assert rating_review.classify_review_priority("Lower Team", "generated_from_behavior", fixture_match_count=1) == "medium"
    assert rating_review.classify_review_priority("Archive Team", "generated_from_behavior") == "low"
    assert rating_review.classify_review_priority("England", "manual_existing") == "reviewed"


def test_manual_reviewed_rows_are_not_overwritten() -> None:
    proposals = pd.DataFrame(
        [
            {
                "team": "Brazil",
                "reviewed_attack": 0.40,
                "reviewed_defense": 0.40,
                "reviewed_recent_form": 0.40,
            }
        ]
    )

    updated = rating_review.apply_review_proposals(_ratings(), proposals)
    brazil = updated.loc[updated["team"] == "Brazil"].iloc[0]

    assert brazil["attack"] == 0.82
    assert brazil["data_quality"] == "manual_reviewed"


def test_review_proposals_stay_within_rating_bounds() -> None:
    ratings = pd.DataFrame(
        [
            {
                "team": "Extreme",
                "elo": 2300,
                "attack": 0.95,
                "defense": 0.95,
                "recent_form": 0.95,
                "data_quality": "generated_from_behavior",
            }
        ]
    )
    behavior = pd.DataFrame([{"team": "Extreme", "attack_index_final": 1.0, "defense_index_final": 1.0, "recent_form_index": 1.0}])

    proposal = rating_review.propose_reviewed_rating("Extreme", ratings, behavior)

    assert 0.35 <= proposal["reviewed_attack"] <= 0.90
    assert 0.35 <= proposal["reviewed_defense"] <= 0.90
    assert 0.35 <= proposal["reviewed_recent_form"] <= 0.90
    assert proposal["recommended_data_quality"] == "manual_review_candidate"


def test_proposal_script_writes_output_without_modifying_source_unless_write(monkeypatch, tmp_path, capsys) -> None:
    import scripts.propose_reviewed_ratings as script

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    ratings_path = data_dir / "team_ratings.csv"
    _ratings().to_csv(ratings_path, index=False)
    output_path = data_dir / "team_ratings_review_proposals.csv"

    monkeypatch.setattr(script, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(script, "DATA_DIR", data_dir)
    monkeypatch.setattr(script, "load_team_ratings_csv", lambda path=None: pd.read_csv(ratings_path))
    monkeypatch.setattr(script, "load_team_behavior", lambda: _behavior())
    monkeypatch.setattr(sys, "argv", ["propose_reviewed_ratings.py", "--teams", "Argentina", "--output", str(output_path)])

    script.main()

    output = capsys.readouterr().out
    unchanged = pd.read_csv(ratings_path)
    proposals = pd.read_csv(output_path)
    assert "was not modified" in output
    assert unchanged.loc[unchanged["team"] == "Argentina", "data_quality"].iloc[0] == "generated_from_behavior"
    assert proposals.loc[proposals["team"] == "Argentina", "review_action"].iloc[0] == "review_candidate"


def test_proposal_script_write_updates_generated_rows_only(monkeypatch, tmp_path, capsys) -> None:
    import scripts.propose_reviewed_ratings as script

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    ratings_path = data_dir / "team_ratings.csv"
    _ratings().to_csv(ratings_path, index=False)

    monkeypatch.setattr(script, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(script, "DATA_DIR", data_dir)
    monkeypatch.setattr(script, "load_team_ratings_csv", lambda path=None: pd.read_csv(ratings_path))
    monkeypatch.setattr(script, "load_team_behavior", lambda: _behavior())
    monkeypatch.setattr(sys, "argv", ["propose_reviewed_ratings.py", "--teams", "Argentina", "England", "--write"])

    script.main()

    output = capsys.readouterr().out
    written = pd.read_csv(ratings_path)
    argentina = written.loc[written["team"] == "Argentina"].iloc[0]
    england = written.loc[written["team"] == "England"].iloc[0]
    assert "updated generated rows: 1" in output
    assert argentina["data_quality"] == "manual_review_candidate"
    assert england["data_quality"] == "manual_csv"


def test_rating_review_summary_handles_missing_columns() -> None:
    summary = rating_review.rating_review_summary(pd.DataFrame())

    assert summary["manual_reviewed_rows"] == 0
    assert summary["generated_from_behavior_rows"] == 0


def test_audit_rating_review_script_runs(monkeypatch, tmp_path, capsys) -> None:
    import scripts.audit_rating_review as script

    audit = pd.DataFrame(
        [
            {
                "team": "Argentina",
                "rating_status": "generated_from_behavior",
                "priority": "high",
                "warning": "generated rating requires review",
                "backtest_match_count": 1,
                "fixture_match_count": 0,
            }
        ]
    )
    monkeypatch.setattr(script, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(script, "load_team_ratings_csv", lambda: _ratings())
    monkeypatch.setattr(script, "load_team_behavior", lambda: _behavior())
    monkeypatch.setattr(script, "load_historical_matches", lambda use_cache=False: pd.DataFrame())
    monkeypatch.setattr(script, "load_completed_matches_for_backtest", lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr(script, "audit_generated_ratings", lambda *args, **kwargs: audit)
    monkeypatch.setattr(
        script,
        "rating_review_summary",
        lambda *args, **kwargs: {
            "manual_reviewed_rows": 1,
            "manual_existing_rows": 1,
            "generated_from_behavior_rows": 1,
            "neutral_placeholder_rows": 0,
            "high_priority_needing_review": 1,
        },
    )
    monkeypatch.setattr(script, "today_iso", lambda: "2026-06-21")
    monkeypatch.setattr(sys, "argv", ["audit_rating_review.py"])

    script.main()

    output = capsys.readouterr().out
    assert "generated ratings: 1" in output
    assert (tmp_path / "reports" / "rating_review_audit_2026-06-21.md").exists()
