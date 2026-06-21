from __future__ import annotations

import sys

import pandas as pd

from src.backtest_diagnostics import (
    audit_backtest_match_inputs,
    audit_backtest_rating_coverage,
    audit_backtest_team_aliases,
    build_backtest_qa_summary,
)


def _completed() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "match_id": "M1",
                "date_utc": "2026-06-14",
                "home": "Germany",
                "away": "Curacao",
                "home_goals": 7,
                "away_goals": 1,
                "competition": "FIFA World Cup",
            }
        ]
    )


def _predictions(home_prob: float = 0.365, draw_prob: float = 0.270, away_prob: float = 0.365) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_mode": "baseline_manual",
                "match_id": "M1",
                "home_win_prob": home_prob,
                "draw_prob": draw_prob,
                "away_win_prob": away_prob,
                "actual_result_prob": home_prob,
            },
            {
                "model_mode": "behavior_adjusted",
                "match_id": "M1",
                "home_win_prob": home_prob + 0.01,
                "draw_prob": draw_prob,
                "away_win_prob": away_prob - 0.01,
                "actual_result_prob": home_prob + 0.01,
            },
        ]
    )


def test_missing_manual_rating_detection() -> None:
    manual = pd.DataFrame([{"team": "Germany", "attack": 0.8, "defense": 0.75, "recent_form": 0.7}])
    adjusted = manual.copy()
    behavior = pd.DataFrame([{"team": "Germany"}])

    coverage = audit_backtest_rating_coverage(_completed(), manual, adjusted, behavior, pd.DataFrame())
    curacao = coverage.loc[coverage["team"] == "Curacao"].iloc[0]

    assert bool(curacao["in_manual_ratings"]) is False
    assert "missing manual rating" in curacao["warning"]


def test_neutral_fallback_detection() -> None:
    manual = pd.DataFrame([{"team": "Germany", "attack": 0.8, "defense": 0.75, "recent_form": 0.7}])
    adjusted = pd.DataFrame(
        [
            {"team": "Germany", "attack": 0.8, "defense": 0.75, "recent_form": 0.7},
            {"team": "Curacao", "attack": 0.55, "defense": 0.55, "recent_form": 0.55, "data_quality": "neutral_fixture_base"},
        ]
    )

    coverage = audit_backtest_rating_coverage(_completed(), manual, adjusted, pd.DataFrame(), pd.DataFrame())
    curacao = coverage.loc[coverage["team"] == "Curacao"].iloc[0]

    assert bool(curacao["used_neutral_fallback"]) is True
    assert "neutral fallback used" in curacao["warning"]


def test_alias_mismatch_detection() -> None:
    ratings = pd.DataFrame([{"team": "Curaçao", "attack": 0.5, "defense": 0.5, "recent_form": 0.5}])
    aliases = pd.DataFrame([{"alias": "Curacao", "canonical": "Curaçao"}])

    alias_audit = audit_backtest_team_aliases(_completed(), ratings, aliases)
    row = alias_audit.loc[alias_audit["team_in_backtest"] == "Curacao"].iloc[0]

    assert bool(row["exact_match_in_ratings"]) is False
    assert row["possible_alias"] == "Curaçao"
    assert row["recommendation"] == ""


def test_flat_probability_warning() -> None:
    manual = pd.DataFrame([{"team": "Germany", "attack": 0.55, "defense": 0.55, "recent_form": 0.55}])
    adjusted = manual.copy()

    audit = audit_backtest_match_inputs(_predictions(), _completed(), manual, adjusted)
    row = audit.iloc[0]

    assert "probabilities_too_flat" in row["probability_warning"]
    assert "missing_rating_probabilities_flat" in row["probability_warning"]


def test_suspicious_favorite_warning() -> None:
    completed = pd.DataFrame(
        [{"match_id": "M1", "date_utc": "2026-06-14", "home": "Strong", "away": "Weak", "home_goals": 1, "away_goals": 0}]
    )
    manual = pd.DataFrame(
        [
            {"team": "Strong", "attack": 0.9, "defense": 0.85, "recent_form": 0.8},
            {"team": "Weak", "attack": 0.4, "defense": 0.4, "recent_form": 0.4},
        ]
    )
    predictions = pd.DataFrame(
        [
            {"model_mode": "baseline_manual", "match_id": "M1", "home_win_prob": 0.45, "draw_prob": 0.3, "away_win_prob": 0.25, "actual_result_prob": 0.45},
            {"model_mode": "behavior_adjusted", "match_id": "M1", "home_win_prob": 0.45, "draw_prob": 0.3, "away_win_prob": 0.25, "actual_result_prob": 0.45},
        ]
    )

    audit = audit_backtest_match_inputs(predictions, completed, manual, manual)

    warning = audit.iloc[0]["probability_warning"]
    assert "favorite_manual_rating_gap_high_but_probability_low" in warning
    assert "elite_vs_weak_probability_low" in warning


def test_germany_curacao_style_case_is_flagged_when_team_lacks_ratings() -> None:
    manual = pd.DataFrame([{"team": "Germany", "attack": 0.82, "defense": 0.78, "recent_form": 0.74}])
    adjusted = manual.copy()

    audit = audit_backtest_match_inputs(_predictions(), _completed(), manual, adjusted)
    row = audit.iloc[0]

    assert "away missing manual rating" in row["rating_coverage_warning"]
    assert "probabilities_too_flat" in row["probability_warning"]


def test_qa_summary_counts_problems() -> None:
    manual = pd.DataFrame([{"team": "Germany", "attack": 0.8, "defense": 0.75, "recent_form": 0.7}])
    adjusted = manual.copy()
    coverage = audit_backtest_rating_coverage(_completed(), manual, adjusted, pd.DataFrame(), pd.DataFrame())
    match_audit = audit_backtest_match_inputs(_predictions(), _completed(), manual, adjusted)
    aliases = audit_backtest_team_aliases(_completed(), manual, pd.DataFrame())

    summary = build_backtest_qa_summary(_completed(), coverage, match_audit, aliases)

    assert summary["neutral_fallback_count"] >= 1
    assert summary["suspicious_probability_count"] >= 1
    assert "unreliable" in summary["qa_warning"]


def test_qa_script_runs_without_crashing(monkeypatch, tmp_path, capsys) -> None:
    import scripts.audit_backtest_quality as script

    fake_result = {
        "matches": _completed(),
        "qa_rating_coverage": pd.DataFrame(
            [
                {
                    "team": "Curacao",
                    "warning": "missing manual rating; neutral fallback used",
                    "used_neutral_fallback": True,
                    "in_team_behavior": False,
                }
            ]
        ),
        "qa_match_inputs": audit_backtest_match_inputs(_predictions(), _completed(), pd.DataFrame(), pd.DataFrame()),
        "qa_aliases": pd.DataFrame(
            [
                {
                    "team_in_backtest": "Curacao",
                    "recommendation": "Add alias",
                }
            ]
        ),
        "qa_summary": {"completed_matches": 1, "unique_teams": 2, "qa_warning": "Backtest metrics may be unreliable."},
    }
    monkeypatch.setattr(script, "run_backtest", lambda **kwargs: fake_result)
    monkeypatch.setattr(script, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(script, "today_iso", lambda: "2026-06-21")
    monkeypatch.setattr(sys, "argv", ["audit_backtest_quality.py", "--start-date", "2026-06-14", "--end-date", "2026-06-14"])

    script.main()

    output = capsys.readouterr().out
    assert "completed matches used: 1" in output
    assert (tmp_path / "reports" / "backtest_quality_audit_2026-06-21.md").exists()
