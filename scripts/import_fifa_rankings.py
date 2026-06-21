from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import load_completed_matches_for_backtest  # noqa: E402
from src.config import DATA_DIR  # noqa: E402
from src.external_prior_import import (  # noqa: E402
    EXTERNAL_STRENGTH_INPUT_COLUMNS,
    build_external_prior_updates,
    merge_external_prior_updates,
)
from src.external_priors import load_external_priors  # noqa: E402
from src.fifa_ranking_import import (  # noqa: E402
    build_external_strength_from_fifa_matches,
    fifa_ranking_import_summary,
    load_fifa_ranking_snapshot,
    match_fifa_rankings_to_required_teams,
    update_external_strength_template_from_fifa,
)
from src.historical_data import load_historical_matches  # noqa: E402
from src.rating_coverage import collect_required_teams  # noqa: E402
from src.team_behavior import load_team_behavior  # noqa: E402
from src.team_names import load_team_name_aliases_df  # noqa: E402
from src.utils import read_csv_with_columns, today_iso  # noqa: E402


DEFAULT_INPUT = DATA_DIR / "raw" / "fifa_rankings_snapshot.csv"
DEFAULT_OUTPUT = DATA_DIR / "raw" / "external_team_strength_from_fifa.csv"
DEFAULT_TEMPLATE = DATA_DIR / "raw" / "external_team_strength_missing_template.csv"
DEFAULT_EXTERNAL_PRIORS = DATA_DIR / "team_rating_external_priors.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a local FIFA ranking snapshot for required teams.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--update-template", action="store_true", help="Fill blank cells in the missing-team strength template.")
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    parser.add_argument("--write-external-priors", action="store_true", help="Update data/team_rating_external_priors.csv using matched FIFA rows.")
    parser.add_argument("--external-priors", default=str(DEFAULT_EXTERNAL_PRIORS))
    parser.add_argument("--start-date", default="2026-06-11")
    parser.add_argument("--end-date", default="2026-06-21")
    parser.add_argument("--competition", default="World Cup")
    args = parser.parse_args()

    input_path = _resolve_path(args.input)
    output_path = _resolve_path(args.output)
    template_path = _resolve_path(args.template)
    external_priors_path = _resolve_path(args.external_priors)

    required = _load_required_teams(args)
    aliases = load_team_name_aliases_df(include_defaults=True)
    fifa_snapshot = load_fifa_ranking_snapshot(input_path)
    matches = match_fifa_rankings_to_required_teams(fifa_snapshot, required, aliases)
    external_strength = build_external_strength_from_fifa_matches(matches)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    external_strength.to_csv(output_path, index=False)

    if args.update_template:
        existing_template = read_csv_with_columns(template_path, EXTERNAL_STRENGTH_INPUT_COLUMNS)
        updated_template = update_external_strength_template_from_fifa(existing_template, external_strength)
        template_path.parent.mkdir(parents=True, exist_ok=True)
        updated_template.to_csv(template_path, index=False)

    updated_external_prior_count = 0
    if args.write_external_priors:
        existing_priors = load_external_priors(external_priors_path)
        proposed, diagnostics = build_external_prior_updates(external_strength, existing_priors)
        updated_priors = merge_external_prior_updates(existing_priors, proposed, diagnostics)
        updated_external_prior_count = _changed_rows(existing_priors, updated_priors)
        external_priors_path.parent.mkdir(parents=True, exist_ok=True)
        updated_priors.to_csv(external_priors_path, index=False)

    required_count = _required_count(required)
    summary = fifa_ranking_import_summary(matches, required_count, len(fifa_snapshot))
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    today = today_iso()
    report_csv = reports_dir / f"fifa_ranking_import_{today}.csv"
    report_md = reports_dir / f"fifa_ranking_import_{today}.md"
    matches.to_csv(report_csv, index=False)
    report_md.write_text(_markdown_report(today, summary, matches, output_path, args, updated_external_prior_count), encoding="utf-8")

    print(f"required teams: {summary['required_teams']}")
    print(f"FIFA rows loaded: {summary['fifa_rows_loaded']}")
    print(f"matched required teams: {summary['matched_required_teams']}")
    print(f"missing required teams: {_team_list(_status_rows(matches, 'unmatched_required_team'))}")
    print(f"unmatched FIFA rows: {summary['unmatched_fifa_rows']}")
    print(f"ambiguous matches: {_team_list(_status_rows(matches, 'ambiguous'))}")
    print(f"teams missing rank/points: {_team_list(_status_rows(matches, 'missing_rank_or_points'))}")
    print(f"output file path: {_display_path(output_path)}")
    if args.update_template:
        print(f"updated template: {_display_path(template_path)}")
    if args.write_external_priors:
        print(f"external prior rows updated: {updated_external_prior_count}")
        print(f"updated external priors: {_display_path(external_priors_path)}")
    else:
        print("data/team_rating_external_priors.csv was not modified. Pass --write-external-priors to update it.")
    print(f"report CSV: {_display_path(report_csv)}")
    print(f"report: {_display_path(report_md)}")


def _load_required_teams(args: argparse.Namespace) -> pd.DataFrame:
    fixtures = read_csv_with_columns(DATA_DIR / "fixtures.csv", ["home", "away"])
    historical = load_historical_matches(use_cache=False)
    behavior = load_team_behavior()
    backtest = load_completed_matches_for_backtest(start_date=args.start_date, end_date=args.end_date)
    if args.competition and not backtest.empty and "competition" in backtest.columns:
        needle = str(args.competition).strip().lower()
        backtest = backtest.loc[backtest["competition"].astype(str).str.lower().str.contains(needle, na=False)].copy()
    return collect_required_teams(fixtures, historical, behavior, backtest)


def _required_count(required: pd.DataFrame) -> int:
    if required.empty:
        return 0
    model_required = required.copy()
    if "required_for_model" in model_required.columns:
        model_required = model_required.loc[model_required["required_for_model"].astype(bool)].copy()
    team_col = "canonical_team" if "canonical_team" in model_required.columns else "team"
    return int(model_required[team_col].dropna().astype(str).str.strip().replace("", pd.NA).dropna().nunique())


def _changed_rows(before: pd.DataFrame, after: pd.DataFrame) -> int:
    if before.empty or after.empty or "team" not in before.columns or "team" not in after.columns:
        return 0
    before_by_team = before.set_index(before["team"].astype(str).str.lower(), drop=False)
    changed = 0
    for _, row in after.iterrows():
        key = str(row.get("team", "") or "").strip().lower()
        if not key or key not in before_by_team.index:
            continue
        old = before_by_team.loc[key]
        if isinstance(old, pd.DataFrame):
            old = old.iloc[0]
        if any(_text_value(row.get(col, "")) != _text_value(old.get(col, "")) for col in after.columns if col in old.index):
            changed += 1
    return changed


def _status_rows(matches: pd.DataFrame, status: str) -> pd.DataFrame:
    if matches.empty or "match_status" not in matches.columns:
        return pd.DataFrame()
    return matches.loc[matches["match_status"].astype(str) == status].copy()


def _team_list(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "None"
    col = "canonical_team" if "canonical_team" in df.columns else "team"
    if col not in df.columns:
        return "None"
    return ", ".join(df[col].dropna().astype(str).tolist()) or "None"


def _markdown_report(
    today: str,
    summary: dict[str, object],
    matches: pd.DataFrame,
    output_path: Path,
    args: argparse.Namespace,
    updated_external_prior_count: int,
) -> str:
    parts = [
        f"# FIFA Ranking Import - {today}",
        "",
        "This report matches a local FIFA ranking snapshot to required project teams. It does not change model formulas or trading behavior.",
        "",
        "## Summary",
        "",
        _to_markdown(pd.DataFrame([summary])),
        "",
        "## Files",
        "",
        f"- Input snapshot: `{args.input}`",
        f"- External-strength output: `{_display_path(output_path)}`",
        f"- Template updated: `{bool(args.update_template)}`",
        f"- External prior rows updated: `{updated_external_prior_count}`",
        "",
        "## Missing Required Teams",
        "",
        _to_markdown(_printable(_status_rows(matches, "unmatched_required_team"))),
        "",
        "## Ambiguous Matches",
        "",
        _to_markdown(_printable(_status_rows(matches, "ambiguous"))),
        "",
        "## Teams Missing Rank Or Points",
        "",
        _to_markdown(_printable(_status_rows(matches, "missing_rank_or_points"))),
        "",
        "## Unused FIFA Rows",
        "",
        _to_markdown(_printable(_status_rows(matches, "unused_fifa_row").head(50))),
    ]
    return "\n".join(parts)


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


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return PROJECT_ROOT / resolved


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _text_value(value) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


if __name__ == "__main__":
    main()
