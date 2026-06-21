from __future__ import annotations

import sys

import pandas as pd

from src import rating_coverage
from src.ratings import neutral_team_rating


def test_collect_required_teams_from_fixtures_and_backtest(monkeypatch) -> None:
    ratings = pd.DataFrame([{"team": "Alpha"}])

    def fake_read_csv(path, columns):
        if str(path).endswith("team_ratings.csv"):
            return ratings
        return pd.DataFrame(columns=columns)

    monkeypatch.setattr(rating_coverage, "read_csv_with_columns", fake_read_csv)
    fixtures = pd.DataFrame([{"home": "Alpha", "away": "Beta"}])
    historical = pd.DataFrame([{"team": "Gamma", "opponent": "Delta"}])
    behavior = pd.DataFrame([{"team": "Beta"}, {"team": "Gamma"}])
    backtest = pd.DataFrame([{"home": "Germany", "away": "Curacao"}])

    required = rating_coverage.collect_required_teams(fixtures, historical, behavior, backtest)

    beta = required.loc[required["team"] == "Beta"].iloc[0]
    gamma = required.loc[required["team"] == "Gamma"].iloc[0]
    germany = required.loc[required["team"] == "Germany"].iloc[0]
    assert bool(beta["source_fixtures"]) is True
    assert bool(beta["required_for_model"]) is True
    assert bool(gamma["source_historical"]) is True
    assert bool(gamma["required_for_model"]) is False
    assert bool(germany["source_backtest"]) is True


def test_missing_rating_detection() -> None:
    required = pd.DataFrame(
        [
            {"team": "Alpha", "canonical_team": "Alpha", "required_for_model": True, "source_behavior": False, "source_fixtures": True},
            {"team": "Beta", "canonical_team": "Beta", "required_for_model": True, "source_behavior": True, "source_fixtures": False},
        ]
    )
    ratings = pd.DataFrame([{"team": "Alpha", "attack": 0.7, "defense": 0.6, "recent_form": 0.6, "elo": 1800, "data_quality": "manual_csv"}])

    audit = rating_coverage.audit_rating_coverage(required, ratings, pd.DataFrame(columns=["alias", "canonical"]))

    beta = audit.loc[audit["team"] == "Beta"].iloc[0]
    assert bool(beta["in_team_ratings"]) is False
    assert "missing rating row" in beta["warning"]
    assert "behavior exists but manual rating missing" in beta["warning"]


def test_existing_ratings_are_not_overwritten() -> None:
    existing = pd.DataFrame(
        [
            {"team": "Alpha", "elo": 1800, "attack": 0.7, "defense": 0.6, "recent_form": 0.6},
        ]
    )
    proposals = pd.DataFrame(
        [
            {"team": "Alpha", "elo": 1900, "attack": 0.9, "defense": 0.9, "recent_form": 0.9},
            {"team": "Beta", "elo": 1700, "attack": 0.55, "defense": 0.55, "recent_form": 0.55},
        ]
    )

    updated = rating_coverage.append_missing_team_ratings(existing, proposals)

    alpha = updated.loc[updated["team"] == "Alpha"].iloc[0]
    assert alpha["attack"] == 0.7
    assert "Beta" in set(updated["team"])


def test_behavior_based_rating_proposal() -> None:
    behavior = pd.DataFrame(
        [
            {
                "team": "Beta",
                "attack_index_final": 0.90,
                "defense_index_final": 0.80,
                "recent_form_index": 0.70,
            }
        ]
    )
    historical = pd.DataFrame([{"team": "Beta", "date_utc": "2026-01-01", "team_elo_pre": 1750}])

    proposal = rating_coverage.propose_missing_team_rating("Beta", behavior, historical, pd.DataFrame())

    assert proposal["attack"] == 0.655
    assert proposal["defense"] == 0.625
    assert proposal["recent_form"] == 0.595
    assert proposal["elo"] == 1750.0
    assert proposal["data_quality"] == "generated_from_behavior"


def test_neutral_proposal_when_behavior_missing() -> None:
    proposal = rating_coverage.propose_missing_team_rating("No Data", pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    assert proposal["attack"] == 0.55
    assert proposal["defense"] == 0.55
    assert proposal["recent_form"] == 0.55
    assert proposal["data_quality"] == "generated_neutral_placeholder_low"


def test_alias_proposal_detects_bosnia_and_curacao() -> None:
    required = pd.DataFrame(
        [
            {"team": "Bosnia-H.", "canonical_team": "Bosnia and Herzegovina", "required_for_model": True},
            {"team": "Curacao", "canonical_team": "Curaçao", "required_for_model": True},
        ]
    )
    ratings = pd.DataFrame([{"team": "Bosnia and Herzegovina"}, {"team": "Curaçao"}])

    aliases = rating_coverage.propose_team_aliases(required, ratings, pd.DataFrame(columns=["alias", "canonical"]))

    pairs = set(zip(aliases["alias"], aliases["canonical"]))
    assert ("Bosnia-H.", "Bosnia and Herzegovina") in pairs
    assert ("Curacao", "Curaçao") in pairs


def test_append_missing_aliases_only() -> None:
    existing = pd.DataFrame([{"alias": "USA", "canonical": "United States"}])
    proposals = pd.DataFrame(
        [
            {"alias": "USA", "canonical": "United States", "reason": "existing"},
            {"alias": "Curacao", "canonical": "Curaçao", "reason": "new"},
        ]
    )

    updated = rating_coverage.append_missing_aliases(existing, proposals)

    assert len(updated) == 2
    assert "Curacao" in set(updated["alias"])


def test_rating_fallback_flag_is_exposed() -> None:
    fallback = neutral_team_rating("Missing Team")

    assert bool(fallback["used_neutral_fallback"]) is True
    assert fallback["rating_warning"] == "neutral fallback used"


def test_proposal_script_write_appends_missing_only(monkeypatch, tmp_path, capsys) -> None:
    import scripts.propose_missing_team_ratings as script

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    existing = pd.DataFrame(
        [
            {
                "team": "Alpha",
                "elo": 1800,
                "attack": 0.7,
                "defense": 0.6,
                "recent_form": 0.6,
                "fifa_rank_proxy": 20,
                "training_temp_c": 20,
                "training_humidity_pct": 60,
                "data_quality": "manual_csv",
                "last_updated": "2026-06-21",
                "notes": "",
            }
        ]
    )
    proposals = pd.DataFrame(
        [
            {
                "team": "Alpha",
                "elo": 1900,
                "attack": 0.9,
                "defense": 0.9,
                "recent_form": 0.9,
                "fifa_rank_proxy": 1,
                "training_temp_c": 20,
                "training_humidity_pct": 60,
                "data_quality": "generated_from_behavior",
                "last_updated": "2026-06-21",
                "notes": "Generated",
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
                "data_quality": "generated_neutral_placeholder_low",
                "last_updated": "2026-06-21",
                "notes": "Generated",
            },
        ]
    )

    monkeypatch.setattr(script, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(script, "DATA_DIR", data_dir)
    monkeypatch.setattr(script, "load_historical_matches", lambda use_cache=False: pd.DataFrame())
    monkeypatch.setattr(script, "load_team_behavior", lambda: pd.DataFrame())
    monkeypatch.setattr(script, "load_team_ratings_csv", lambda path=None: existing)
    monkeypatch.setattr(script, "load_team_name_aliases_df", lambda include_defaults=False: pd.DataFrame(columns=["alias", "canonical"]))
    monkeypatch.setattr(script, "collect_required_teams", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(script, "audit_rating_coverage", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(script, "build_missing_rating_proposals", lambda *args, **kwargs: proposals)
    monkeypatch.setattr(script, "propose_team_aliases", lambda *args, **kwargs: pd.DataFrame(columns=["alias", "canonical", "reason"]))
    monkeypatch.setattr(script, "read_csv_with_columns", lambda path, columns: pd.DataFrame(columns=columns))
    monkeypatch.setattr(sys, "argv", ["propose_missing_team_ratings.py", "--write"])

    script.main()

    output = capsys.readouterr().out
    written = pd.read_csv(data_dir / "team_ratings.csv")
    assert "appended rating rows: 1" in output
    assert written.loc[written["team"] == "Alpha", "attack"].iloc[0] == 0.7
    assert "Beta" in set(written["team"])


def test_audit_rating_coverage_script_runs(monkeypatch, tmp_path, capsys) -> None:
    import scripts.audit_rating_coverage as script

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    audit = pd.DataFrame(
        [
            {
                "team": "Beta",
                "canonical_team": "Beta",
                "in_team_ratings": False,
                "warning": "missing rating row",
                "possible_alias_match": "",
            }
        ]
    )
    monkeypatch.setattr(script, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(script, "DATA_DIR", data_dir)
    monkeypatch.setattr(script, "read_csv_with_columns", lambda path, columns: pd.DataFrame(columns=columns))
    monkeypatch.setattr(script, "load_historical_matches", lambda use_cache=False: pd.DataFrame())
    monkeypatch.setattr(script, "load_team_behavior", lambda: pd.DataFrame())
    monkeypatch.setattr(script, "load_completed_matches_for_backtest", lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr(script, "load_team_ratings_csv", lambda: pd.DataFrame())
    monkeypatch.setattr(script, "load_team_name_aliases_df", lambda include_defaults=False: pd.DataFrame(columns=["alias", "canonical"]))
    monkeypatch.setattr(script, "collect_required_teams", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(script, "audit_rating_coverage", lambda *args, **kwargs: audit)
    monkeypatch.setattr(script, "propose_team_aliases", lambda *args, **kwargs: pd.DataFrame(columns=["alias", "canonical", "reason"]))
    monkeypatch.setattr(
        script,
        "rating_coverage_summary",
        lambda *args, **kwargs: {
            "required_teams": 1,
            "teams_in_team_ratings": 0,
            "missing_manual_ratings": 1,
            "generated_rating_rows": 0,
            "neutral_fallback_risk_count": 1,
            "alias_warning_count": 0,
            "recommended_fixes": "Add missing generated rows and review aliases.",
        },
    )
    monkeypatch.setattr(script, "today_iso", lambda: "2026-06-21")
    monkeypatch.setattr(sys, "argv", ["audit_rating_coverage.py"])

    script.main()

    output = capsys.readouterr().out
    assert "required teams: 1" in output
    assert (tmp_path / "reports" / "rating_coverage_audit_2026-06-21.md").exists()
