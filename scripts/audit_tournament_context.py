from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import ModelConfig, run_match_model  # noqa: E402
from src.prediction_ledger import load_results_ledger  # noqa: E402
from src.ratings import get_team_ratings  # noqa: E402
from src.storage import load_csv  # noqa: E402
from src.tournament_context import (  # noqa: E402
    apply_tournament_context_adjustment,
    build_group_standings_asof,
    classify_match_context,
    context_probabilities_display,
    remaining_group_fixtures_asof,
)
from src.utils import today_iso  # noqa: E402
from src.weather import load_venues  # noqa: E402


FIXTURE_COLUMNS = ["match_id", "date_utc", "time_utc", "competition", "group", "home", "away", "venue", "city", "country"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Tournament Context Layer v1 for one fixture.")
    parser.add_argument("--home", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--fixture-date", required=True)
    parser.add_argument("--competition", default="World Cup")
    parser.add_argument("--group", required=True)
    args = parser.parse_args()

    fixtures = load_csv("fixtures.csv", FIXTURE_COLUMNS)
    results = load_results_ledger()
    match = _find_or_build_match(args, fixtures)
    kickoff = f"{match.get('date_utc', args.fixture_date)} {match.get('time_utc', '00:00')} UTC"
    standings = build_group_standings_asof(results, fixtures, args.group, kickoff)
    remaining = remaining_group_fixtures_asof(fixtures, args.group, kickoff)
    context = classify_match_context(args.home, args.away, match, standings, remaining)

    baseline_probs: dict = {}
    context_probs: dict = {}
    diagnostics: dict = {}
    context_display = pd.DataFrame()
    try:
        model_result = run_match_model(match, get_team_ratings(), load_venues(), pd.DataFrame(), ModelConfig())
        baseline_probs = model_result.get("probs", {})
        context_probs, _context_markets, diagnostics = apply_tournament_context_adjustment(
            baseline_probs,
            model_result.get("alpha", pd.DataFrame()),
            context,
        )
        context_display = context_probabilities_display(baseline_probs, context_probs)
    except Exception as exc:
        diagnostics = {"warnings": f"Could not run baseline model: {exc}"}

    csv_path, md_path = _write_reports(args, standings, context, context_display, diagnostics)

    print("Tournament context audit")
    print("group standings as-of kickoff:")
    print(_plain_table(standings))
    print(f"home qualification need: {context.get('home_need', {})}")
    print(f"away qualification need: {context.get('away_need', {})}")
    print(f"context type: {context.get('context_type', '')}")
    if baseline_probs:
        print(
            "baseline probabilities: "
            f"home={baseline_probs.get('home_win'):.4f}, draw={baseline_probs.get('draw'):.4f}, away={baseline_probs.get('away_win'):.4f}"
        )
    if context_probs:
        print(
            "context-adjusted probabilities: "
            f"home={context_probs.get('home_win'):.4f}, draw={context_probs.get('draw'):.4f}, away={context_probs.get('away_win'):.4f}"
        )
    if diagnostics:
        print(f"probability shifts: max={diagnostics.get('max_probability_shift_pp', '')} pp")
        print(f"warnings: {diagnostics.get('warnings', '') or context.get('warnings', '') or 'None'}")
    print(f"report CSV: {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"report: {md_path.relative_to(PROJECT_ROOT)}")


def _find_or_build_match(args: argparse.Namespace, fixtures: pd.DataFrame) -> pd.Series:
    if not fixtures.empty:
        dates = pd.to_datetime(fixtures["date_utc"], errors="coerce").dt.date.astype(str)
        group_key = fixtures["group"].astype(str).str.lower().str.replace("group", "", regex=False).str.strip()
        home_key = fixtures["home"].astype(str).str.lower()
        away_key = fixtures["away"].astype(str).str.lower()
        requested = {args.home.lower(), args.away.lower()}
        mask = (dates == args.fixture_date) & (group_key == args.group.lower().replace("group", "").strip())
        mask &= home_key.combine(away_key, lambda h, a: {h, a} == requested)
        if mask.any():
            return fixtures.loc[mask].iloc[0]
    return pd.Series(
        {
            "match_id": f"audit_{args.fixture_date}_{args.home}_{args.away}".lower().replace(" ", "_"),
            "date_utc": args.fixture_date,
            "time_utc": "23:59",
            "competition": args.competition,
            "group": f"Group {args.group.upper()}",
            "home": args.home,
            "away": args.away,
            "venue": "",
            "city": "",
            "country": "",
        }
    )


def _write_reports(
    args: argparse.Namespace,
    standings: pd.DataFrame,
    context: dict,
    context_display: pd.DataFrame,
    diagnostics: dict,
) -> tuple[Path, Path]:
    reports = PROJECT_ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    today = today_iso()
    csv_path = reports / f"tournament_context_{today}.csv"
    md_path = reports / f"tournament_context_{today}.md"

    summary = pd.DataFrame(
        [
            {
                "fixture": f"{args.home} vs {args.away}",
                "fixture_date": args.fixture_date,
                "group": args.group,
                "home_incentive": context.get("home_incentive_label", ""),
                "away_incentive": context.get("away_incentive_label", ""),
                "context_type": context.get("context_type", ""),
                "tempo_bias": context.get("tempo_bias", ""),
                "draw_bias": context.get("draw_bias", ""),
                "late_open_game_risk": context.get("late_open_game_risk", ""),
                "max_probability_shift_pp": diagnostics.get("max_probability_shift_pp", pd.NA),
                "warnings": diagnostics.get("warnings", "") or context.get("warnings", ""),
                "explanation": context.get("explanation", ""),
            }
        ]
    )
    summary.to_csv(csv_path, index=False)
    md_path.write_text(
        "\n".join(
            [
                f"# Tournament Context Audit - {today}",
                "",
                "This audit is diagnostic. It does not change the baseline model, place trades, or size positions.",
                "",
                "## Summary",
                "",
                _to_markdown(summary),
                "",
                "## Group Standings As Of Kickoff",
                "",
                _to_markdown(standings),
                "",
                "## Home Qualification Need",
                "",
                _to_markdown(pd.DataFrame([context.get("home_need", {})])),
                "",
                "## Away Qualification Need",
                "",
                _to_markdown(pd.DataFrame([context.get("away_need", {})])),
                "",
                "## Probability Shifts",
                "",
                _to_markdown(context_display),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return csv_path, md_path


def _plain_table(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "No rows."
    cols = [col for col in ["group_position", "team", "played", "points", "goal_difference", "goals_for", "goals_against"] if col in df.columns]
    return df[cols].to_string(index=False)


def _to_markdown(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_No rows._"
    display = df.fillna("").astype(str)
    headers = list(display.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in display.values.tolist():
        lines.append("| " + " | ".join(str(value).replace("|", "\\|").replace("\n", "<br>") for value in row) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
