from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import run_backtest  # noqa: E402
from src.backtest_diagnostics import explain_match_probability_issue  # noqa: E402
from src.utils import today_iso  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit rating coverage and probability QA for completed-match backtests.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--competition", default=None)
    parser.add_argument("--teams", nargs="*", default=None)
    args = parser.parse_args()

    result = run_backtest(
        start_date=args.start_date,
        end_date=args.end_date,
        competition=args.competition,
        teams=args.teams,
    )
    matches = _frame(result.get("matches"))
    coverage = _frame(result.get("qa_rating_coverage"))
    match_audit = _frame(result.get("qa_match_inputs"))
    aliases = _frame(result.get("qa_aliases"))
    summary = result.get("qa_summary", {})

    missing_manual = _filter_warning(coverage, "missing manual rating")
    missing_behavior = _filter_warning(coverage, "missing behavior row")
    fallback = coverage.loc[coverage["used_neutral_fallback"].astype(bool)] if not coverage.empty and "used_neutral_fallback" in coverage.columns else pd.DataFrame()
    alias_warnings = aliases.loc[aliases["recommendation"].astype(str) != ""] if not aliases.empty and "recommendation" in aliases.columns else pd.DataFrame()
    flat = _filter_warning(match_audit, "probabilities_too_flat", column="probability_warning")
    suspicious = match_audit.loc[match_audit["probability_warning"].astype(str) != ""] if not match_audit.empty and "probability_warning" in match_audit.columns else pd.DataFrame()
    favorite = suspicious.loc[suspicious["probability_warning"].astype(str).str.contains("favorite|elite_vs_weak", case=False, na=False)] if not suspicious.empty else pd.DataFrame()
    top_suspicious = suspicious.head(10)
    germany_curacao = _germany_curacao(match_audit)

    print(f"completed matches used: {len(matches):,}")
    print(f"unique teams in backtest: {len(coverage):,}")
    print(f"teams missing manual ratings: {_team_list(missing_manual)}")
    print(f"teams missing behavior rows: {_team_list(missing_behavior)}")
    print(f"teams using neutral fallback: {_team_list(fallback)}")
    print(f"possible alias mismatches: {_team_alias_list(alias_warnings)}")
    print(f"matches with flat probabilities: {len(flat):,}")
    print(f"matches with suspicious favorite probabilities: {len(favorite):,}")
    if isinstance(summary, dict) and summary.get("qa_warning"):
        print(f"qa warning: {summary['qa_warning']}")

    print("\nTop 10 suspicious matches")
    print(_printable(top_suspicious).to_string(index=False) if not top_suspicious.empty else "No suspicious matches.")

    if not germany_curacao.empty:
        print("\nGermany vs Curacao diagnostic")
        print(_printable(germany_curacao).to_string(index=False))
        print("reason:", explain_match_probability_issue(germany_curacao.iloc[0]))

    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    today = today_iso()
    matches_path = report_dir / f"backtest_quality_matches_{today}.csv"
    report_path = report_dir / f"backtest_quality_audit_{today}.md"
    match_audit.to_csv(matches_path, index=False)
    report_path.write_text(
        "\n\n".join(
            [
                f"# Backtest Quality Audit - {today}",
                "",
                "This audit checks whether completed-match backtests are using matched team names, manual ratings, behavior rows, and non-flat probabilities.",
                "",
                "## Summary",
                "",
                _to_markdown(pd.DataFrame([summary]) if isinstance(summary, dict) and summary else pd.DataFrame()),
                "",
                "## Teams Missing Manual Ratings",
                "",
                _to_markdown(_printable(missing_manual)),
                "",
                "## Teams Missing Behavior Rows",
                "",
                _to_markdown(_printable(missing_behavior)),
                "",
                "## Teams Using Neutral Fallback",
                "",
                _to_markdown(_printable(fallback)),
                "",
                "## Possible Alias Mismatches",
                "",
                _to_markdown(_printable(alias_warnings)),
                "",
                "## Top 10 Suspicious Matches",
                "",
                _to_markdown(_printable(top_suspicious)),
                "",
                "## Germany vs Curacao Diagnostic",
                "",
                _to_markdown(_printable(germany_curacao)),
                "",
                explain_match_probability_issue(germany_curacao.iloc[0]) if not germany_curacao.empty else "_Germany vs Curacao was not present in this backtest window._",
                "",
                "Backtest metrics should not be trusted until rating coverage and team-name QA are clean.",
            ]
        ),
        encoding="utf-8",
    )
    print(f"\nmatch QA CSV: {matches_path.relative_to(PROJECT_ROOT)}")
    print(f"report: {report_path.relative_to(PROJECT_ROOT)}")


def _frame(value: object) -> pd.DataFrame:
    return value if isinstance(value, pd.DataFrame) else pd.DataFrame()


def _filter_warning(df: pd.DataFrame, pattern: str, column: str = "warning") -> pd.DataFrame:
    if df.empty or column not in df.columns:
        return pd.DataFrame(columns=df.columns)
    return df.loc[df[column].astype(str).str.contains(pattern, case=False, na=False)].copy()


def _germany_curacao(match_audit: pd.DataFrame) -> pd.DataFrame:
    if match_audit.empty:
        return match_audit
    home = match_audit["home"].astype(str).str.lower()
    away = match_audit["away"].astype(str).str.lower()
    mask = (
        home.str.contains("germany", na=False)
        & (away.str.contains("cura", na=False) | away.str.contains("curacao", na=False))
    ) | (
        away.str.contains("germany", na=False)
        & (home.str.contains("cura", na=False) | home.str.contains("curacao", na=False))
    )
    return match_audit.loc[mask].copy()


def _team_list(df: pd.DataFrame) -> str:
    if df.empty or "team" not in df.columns:
        return "None"
    return ", ".join(df["team"].dropna().astype(str).tolist()) or "None"


def _team_alias_list(df: pd.DataFrame) -> str:
    if df.empty or "team_in_backtest" not in df.columns:
        return "None"
    return ", ".join(df["team_in_backtest"].dropna().astype(str).tolist()) or "None"


def _printable(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(lambda value: "" if pd.isna(value) else round(float(value), 4))
    return out


def _to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        values = [str(row.get(col, "")).replace("|", "\\|") for col in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
