from __future__ import annotations

import sys

import pandas as pd

from src.polymarket_match_search import (
    build_polymarket_match_queries,
    find_best_polymarket_match_candidates,
    score_polymarket_market_for_match,
)


def test_build_queries_are_matchup_specific() -> None:
    queries = build_polymarket_match_queries("Colombia", "DR Congo", "World Cup")

    assert "Colombia DR Congo" in queries
    assert "Colombia vs DR Congo" in queries
    assert "COL CDR" in queries
    assert "COL COD" in queries
    assert "fifwc-col-cdr" in queries
    assert "World Cup Colombia DR Congo" in queries
    assert "Colombia" not in queries
    assert "DR Congo" not in queries


def test_correct_sports_market_scores_high() -> None:
    market = {
        "market_id": "PM-COL-COD",
        "question": "Will Colombia beat DR Congo in the World Cup?",
        "slug": "will-colombia-beat-dr-congo-world-cup",
        "event_title": "Colombia vs DR Congo",
        "category": "Sports",
    }

    score = score_polymarket_market_for_match(market, "Colombia", "DR Congo", "World Cup")

    assert score["matched_home"] is True
    assert score["matched_away"] is True
    assert score["matched_sports_terms"] is True
    assert score["confidence"] == "high"
    assert score["rejected"] is False


def test_political_market_is_rejected() -> None:
    market = {
        "market_id": "PM-COL-POL",
        "question": "Will the next president of Colombia be from the left?",
        "slug": "next-president-colombia-left",
        "event_title": "Colombia Elections",
        "category": "Politics",
    }

    score = score_polymarket_market_for_match(market, "Colombia", "DR Congo", "World Cup")

    assert score["rejected"] is True
    assert score["confidence"] == "reject"
    assert "political/non-sports" in score["reject_reason"]


def test_country_only_market_is_not_accepted() -> None:
    market = {
        "market_id": "PM-COL-INFLATION",
        "question": "Will Colombia inflation rise in 2026?",
        "slug": "colombia-inflation-2026",
        "category": "Economics",
    }

    score = score_polymarket_market_for_match(market, "Colombia", "DR Congo", "World Cup")

    assert score["rejected"] is True
    assert score["confidence"] == "reject"


def test_one_team_only_sports_market_is_not_high_confidence() -> None:
    market = {
        "market_id": "PM-COL-QUALIFY",
        "question": "Will Colombia qualify from their World Cup group?",
        "slug": "colombia-qualify-world-cup-group",
        "category": "Sports",
    }

    score = score_polymarket_market_for_match(market, "Colombia", "DR Congo", "World Cup")

    assert score["confidence"] != "high"
    assert score["rejected"] is True


def test_alias_matching_for_common_team_names() -> None:
    cases = [
        ("United States", "Australia", "Will USA beat Australia in the World Cup?"),
        ("DR Congo", "Colombia", "Will Congo DR beat Colombia in the World Cup?"),
        ("DR Congo", "Colombia", "Will CDR beat COL in the World Cup?"),
        ("DR Congo", "Colombia", "Will COD beat COL in the World Cup?"),
        ("South Korea", "Mexico", "Will Korea Republic beat Mexico in the World Cup?"),
    ]
    for home, away, question in cases:
        score = score_polymarket_market_for_match(
            {"market_id": question, "question": question, "event_title": f"{home} vs {away}", "category": "Sports"},
            home,
            away,
            "World Cup",
        )
        assert score["matched_home"] is True
        assert score["matched_away"] is True
        assert score["confidence"] == "high"


def test_candidate_finder_sorts_and_excludes_rejected_by_default() -> None:
    markets = pd.DataFrame(
        [
            {
                "market_id": "POL",
                "question": "Will the next president of Colombia be from the left?",
                "slug": "colombia-president-left",
                "event_title": "Colombia election",
                "category": "Politics",
            },
            {
                "market_id": "MATCH",
                "question": "Will Colombia beat DR Congo in the World Cup?",
                "slug": "will-colombia-beat-dr-congo",
                "event_title": "Colombia vs DR Congo",
                "category": "Sports",
            },
        ]
    )

    def loader(**kwargs):
        return markets

    candidates = find_best_polymarket_match_candidates(
        "Colombia",
        "DR Congo",
        competition="World Cup",
        retrieval_mode="query_only",
        market_loader=loader,
    )

    assert list(candidates["market_id"]) == ["MATCH"]
    assert candidates.iloc[0]["confidence"] == "high"


def test_candidate_finder_can_include_rejected_for_diagnostics() -> None:
    markets = pd.DataFrame(
        [
            {
                "market_id": "POL",
                "question": "Will Colombia inflation rise in 2026?",
                "slug": "colombia-inflation-2026",
                "category": "Economics",
            }
        ]
    )

    def loader(**kwargs):
        return markets

    candidates = find_best_polymarket_match_candidates(
        "Colombia",
        "DR Congo",
        competition="World Cup",
        include_rejected=True,
        retrieval_mode="query_only",
        market_loader=loader,
    )

    assert len(candidates) == 1
    assert bool(candidates.iloc[0]["rejected"]) is True


def test_audit_script_runs_with_synthetic_candidates(monkeypatch, tmp_path, capsys) -> None:
    import scripts.audit_polymarket_match_search as script

    candidates = pd.DataFrame(
        [
            {
                "market_id": "MATCH",
                "question": "Will Colombia beat DR Congo in the World Cup?",
                "slug": "will-colombia-beat-dr-congo",
                "event_title": "Colombia vs DR Congo",
                "category": "Sports",
                "score": 100,
                "confidence": "high",
                "matched_home": True,
                "matched_away": True,
                "matched_competition": True,
                "matched_sports_terms": True,
                "rejected": False,
                "reject_reason": "",
                "score_breakdown": "both teams matched; sports terms",
            },
            {
                "market_id": "POL",
                "question": "Will the next president of Colombia be from the left?",
                "slug": "colombia-president-left",
                "event_title": "Colombia election",
                "category": "Politics",
                "score": 0,
                "confidence": "reject",
                "matched_home": True,
                "matched_away": False,
                "matched_competition": False,
                "matched_sports_terms": False,
                "rejected": True,
                "reject_reason": "political/non-sports market",
                "score_breakdown": "only one selected team matched",
            },
        ]
    )

    monkeypatch.setattr(script, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(script, "find_best_polymarket_match_candidates", lambda *args, **kwargs: candidates)
    monkeypatch.setattr(
        sys,
        "argv",
        ["audit_polymarket_match_search.py", "--home", "Colombia", "--away", "DR Congo", "--competition", "World Cup"],
    )

    script.main()

    output = capsys.readouterr().out
    assert "accepted candidates: 1" in output
    assert "rejected candidates: 1" in output
    assert (tmp_path / "reports").exists()
