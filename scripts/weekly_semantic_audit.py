"""Generate a local weekly semantic audit report.

This script is intentionally local and read-only except for writing one
Markdown report under reports/semantic_audits/. It does not call APIs, require
API keys, modify model inputs, update CSV data, create commits, or send
notifications.
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports" / "semantic_audits"
AUDIT_OUTPUT_PREFIXES = ("reports/semantic_audits/",)
AUDIT_OUTPUT_EXACT_PATHS = {"reports/"}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.schema_registry import (  # noqa: E402
    API_SOURCE_FILES,
    DASHBOARD_DEFINITION_FILES,
    DOCUMENTATION_FILES,
    EXPECTED_CSV_SCHEMAS,
    METRIC_DEFINITION_FILES,
    MODEL_DEFINITION_FILES,
    POLYMARKET_MAPPING_FILES,
    SEMANTIC_LAYER_DOCUMENTS,
)


@dataclass
class CommandResult:
    label: str
    command: str
    returncode: int
    stdout: str
    stderr: str
    skipped: bool = False


@dataclass
class SchemaCheck:
    path: str
    exists: bool
    actual_columns: list[str]
    expected_columns: list[str]
    missing_columns: list[str]
    extra_columns: list[str]
    order_matches: bool


def run_command(args: list[str], cwd: Path = PROJECT_ROOT) -> CommandResult:
    """Run a command and capture output without raising."""
    try:
        completed = subprocess.run(
            args,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        return CommandResult(
            label=args[0],
            command=" ".join(args),
            returncode=127,
            stdout="",
            stderr=str(exc),
        )

    return CommandResult(
        label=args[0],
        command=" ".join(args),
        returncode=completed.returncode,
        stdout=completed.stdout.strip(),
        stderr=completed.stderr.strip(),
    )


def git(args: list[str]) -> CommandResult:
    return run_command(["git", *args])


def command_output(args: list[str], fallback: str = "") -> str:
    result = run_command(args)
    if result.returncode != 0:
        return fallback
    return result.stdout.strip()


def git_output(args: list[str], fallback: str = "") -> str:
    result = git(args)
    if result.returncode != 0:
        return fallback
    return result.stdout.strip()


def report_date_slug(now: datetime) -> str:
    return now.date().isoformat()


def find_last_audit() -> Path | None:
    if not REPORT_DIR.exists():
        return None
    reports = sorted(REPORT_DIR.glob("weekly_semantic_audit_*.md"))
    if not reports:
        return None
    return reports[-1]


def extract_latest_commit(report_path: Path | None) -> str | None:
    if report_path is None or not report_path.exists():
        return None

    text = report_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Latest commit hash:\s*`?([0-9a-fA-F]{7,40})`?", text)
    if not match:
        return None
    return match.group(1)


def git_commit_exists(commit_hash: str | None) -> bool:
    if not commit_hash:
        return False
    result = git(["cat-file", "-e", f"{commit_hash}^{{commit}}"])
    return result.returncode == 0


def parse_status_paths(status_text: str) -> list[str]:
    paths: list[str] = []
    for line in status_text.splitlines():
        if not line.strip():
            continue
        if len(line) >= 3 and line[2] == " ":
            raw_path = line[3:].strip()
        elif len(line) >= 2 and line[1] == " ":
            raw_path = line[2:].strip()
        else:
            raw_path = line.strip()
        if " -> " in raw_path:
            old_path, new_path = raw_path.split(" -> ", 1)
            paths.extend([old_path.strip(), new_path.strip()])
        else:
            paths.append(raw_path)
    return paths


def unique_sorted(paths: list[str]) -> list[str]:
    filtered = []
    for path in paths:
        if not path:
            continue
        if path in AUDIT_OUTPUT_EXACT_PATHS:
            continue
        if any(path.startswith(prefix) for prefix in AUDIT_OUTPUT_PREFIXES):
            continue
        filtered.append(path)
    return sorted(set(filtered))


def changed_files_since_last_audit(last_commit: str | None) -> tuple[list[str], str]:
    committed_files: list[str] = []
    basis = "No previous audit found; using latest commit plus current Git status."

    if git_commit_exists(last_commit):
        diff_result = git(["diff", "--name-only", f"{last_commit}..HEAD"])
        if diff_result.returncode == 0:
            committed_files = [line.strip() for line in diff_result.stdout.splitlines() if line.strip()]
            basis = f"Compared against previous audit commit {last_commit}."
    else:
        diff_tree = git(["diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"])
        if diff_tree.returncode == 0:
            committed_files = [line.strip() for line in diff_tree.stdout.splitlines() if line.strip()]

    status_result = git(["status", "--porcelain"])
    status_files = parse_status_paths(status_result.stdout) if status_result.returncode == 0 else []

    return unique_sorted([*committed_files, *status_files]), basis


def read_csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            return [column.strip() for column in next(reader)]
        except StopIteration:
            return []


def check_csv_schemas() -> list[SchemaCheck]:
    checks: list[SchemaCheck] = []
    for relative_path, expected_columns in EXPECTED_CSV_SCHEMAS.items():
        path = PROJECT_ROOT / relative_path
        if path.exists():
            actual_columns = read_csv_header(path)
        else:
            actual_columns = []

        checks.append(
            SchemaCheck(
                path=relative_path,
                exists=path.exists(),
                actual_columns=actual_columns,
                expected_columns=expected_columns,
                missing_columns=[col for col in expected_columns if col not in actual_columns],
                extra_columns=[col for col in actual_columns if col not in expected_columns],
                order_matches=actual_columns == expected_columns,
            )
        )
    return checks


def semantic_docs_changed(last_audit_path: Path | None, changed_files: list[str]) -> list[str]:
    changed: list[str] = []
    last_audit_mtime = last_audit_path.stat().st_mtime if last_audit_path and last_audit_path.exists() else None

    for raw_path in sorted(SEMANTIC_LAYER_DOCUMENTS):
        path = Path(raw_path).expanduser()
        if not path.exists():
            continue
        if last_audit_mtime is None or path.stat().st_mtime > last_audit_mtime:
            changed.append(raw_path)

    for path in changed_files:
        if "worldcup-alpha-model-semantic-layer" in path and path not in changed:
            changed.append(path)

    return sorted(set(changed))


def filter_changed(changed_files: list[str], tracked_files: set[str]) -> list[str]:
    tracked = {path.rstrip("/") for path in tracked_files}
    return [path for path in changed_files if path in tracked]


def validation_commands() -> list[CommandResult]:
    results: list[CommandResult] = []
    app_path = PROJECT_ROOT / "app.py"
    src_files = sorted((PROJECT_ROOT / "src").glob("*.py"))
    test_files = sorted((PROJECT_ROOT / "tests").glob("test_*.py"))

    if app_path.exists():
        result = run_command([sys.executable, "-m", "py_compile", "app.py"])
        result.label = "Compile app.py"
        results.append(result)
    else:
        results.append(CommandResult("Compile app.py", "python -m py_compile app.py", 0, "", "app.py missing", True))

    if src_files:
        relative_src_files = [str(path.relative_to(PROJECT_ROOT)) for path in src_files]
        result = run_command([sys.executable, "-m", "py_compile", *relative_src_files])
        result.label = "Compile src/*.py"
        results.append(result)
    else:
        results.append(CommandResult("Compile src/*.py", "python -m py_compile src/*.py", 0, "", "No src files found", True))

    if test_files:
        result = run_command([sys.executable, "-m", "pytest", "-q"])
        result.label = "pytest"
        results.append(result)
    else:
        results.append(CommandResult("pytest", "python -m pytest -q", 0, "", "No tests found", True))

    return results


def markdown_list(items: list[str], empty_message: str = "None detected.") -> str:
    if not items:
        return f"- {empty_message}\n"
    return "".join(f"- `{item}`\n" for item in items)


def escape_table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def truncate_output(text: str, max_lines: int = 40) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join([*lines[: max_lines // 2], "...", *lines[-max_lines // 2 :]])


def schema_status(check: SchemaCheck) -> str:
    if not check.exists:
        return "missing file"
    if check.missing_columns:
        return "missing columns"
    if check.extra_columns or not check.order_matches:
        return "schema drift"
    return "ok"


def build_schema_table(checks: list[SchemaCheck]) -> str:
    lines = [
        "| File | Status | Missing columns | Extra columns | Order matches |",
        "| --- | --- | --- | --- | --- |",
    ]
    for check in checks:
        missing = ", ".join(check.missing_columns) if check.missing_columns else "-"
        extra = ", ".join(check.extra_columns) if check.extra_columns else "-"
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{check.path}`",
                    schema_status(check),
                    escape_table_cell(missing),
                    escape_table_cell(extra),
                    "yes" if check.order_matches else "no",
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def build_validation_section(results: list[CommandResult]) -> str:
    blocks: list[str] = []
    for result in results:
        if result.skipped:
            status = "skipped"
        elif result.returncode == 0:
            status = "passed"
        else:
            status = "failed"

        blocks.append(f"### {result.label}\n")
        blocks.append(f"- Command: `{result.command}`\n")
        blocks.append(f"- Status: {status}\n")
        if result.stdout:
            blocks.append("\n```text\n" + truncate_output(result.stdout) + "\n```\n")
        if result.stderr:
            blocks.append("\n```text\n" + truncate_output(result.stderr) + "\n```\n")
    return "\n".join(blocks)


def build_recommendations(
    schema_checks: list[SchemaCheck],
    model_changed: list[str],
    metric_changed: list[str],
    api_changed: list[str],
    mapping_changed: list[str],
    dashboard_changed: list[str],
    docs_changed: list[str],
    semantic_docs: list[str],
    validation_results: list[CommandResult],
) -> list[str]:
    recommendations: list[str] = []

    if any(check.missing_columns or not check.exists for check in schema_checks):
        recommendations.append("Resolve missing CSV files or required columns, or update `src/schema_registry.py` if the schema change is intentional.")
    if any(check.extra_columns or not check.order_matches for check in schema_checks):
        recommendations.append("Review CSV schema drift and decide whether README and semantic-layer table definitions need updates.")
    if model_changed:
        recommendations.append("Review model-definition changes and update expected-goals, scoreline, confidence, or climate caveats in the semantic layer if behavior changed.")
    if metric_changed:
        recommendations.append("Review metric-definition changes and update fair odds, alpha, sensitivity, report, or backtesting definitions where needed.")
    if api_changed:
        recommendations.append("Review API/source logic changes and update source precedence, cache, fallback, and source-label documentation if needed.")
    if mapping_changed:
        recommendations.append("Review Polymarket mapping and price-normalization rules, especially YES/NO translation and low-confidence warnings.")
    if dashboard_changed:
        recommendations.append("Review dashboard output definitions and README screenshots/prose if tab labels, tables, or report sections changed.")
    if docs_changed or semantic_docs:
        recommendations.append("Reconcile README, AGENTS.md, and semantic-layer documentation so local code remains the source of truth.")
    if any(result.returncode != 0 and not result.skipped for result in validation_results):
        recommendations.append("Fix validation failures before treating the semantic layer as current.")

    if not recommendations:
        recommendations.append("No urgent semantic-layer updates detected. Keep this audit report as the baseline for the next run.")

    return recommendations


def write_report() -> Path:
    now = datetime.now(UTC)
    slug = report_date_slug(now)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    last_audit_path = find_last_audit()
    last_commit = extract_latest_commit(last_audit_path)
    changed_files, comparison_basis = changed_files_since_last_audit(last_commit)
    schema_checks = check_csv_schemas()
    validation_results = validation_commands()

    branch = git_output(["branch", "--show-current"], "unknown")
    latest_commit = git_output(["rev-parse", "--short", "HEAD"], "unknown")
    git_status = git_output(["status", "--short"], "")
    recent_commits = git_output(["log", "--oneline", "-5"], "")

    model_changed = filter_changed(changed_files, MODEL_DEFINITION_FILES)
    metric_changed = filter_changed(changed_files, METRIC_DEFINITION_FILES)
    api_changed = filter_changed(changed_files, API_SOURCE_FILES)
    mapping_changed = filter_changed(changed_files, POLYMARKET_MAPPING_FILES)
    dashboard_changed = filter_changed(changed_files, DASHBOARD_DEFINITION_FILES)
    docs_changed = filter_changed(changed_files, DOCUMENTATION_FILES)
    semantic_changed = semantic_docs_changed(last_audit_path, changed_files)

    missing_required = [
        f"{check.path}: {', '.join(check.missing_columns)}"
        for check in schema_checks
        if check.missing_columns
    ]
    missing_files = [
        f"{check.path}: file missing"
        for check in schema_checks
        if not check.exists
    ]

    recommendations = build_recommendations(
        schema_checks,
        model_changed,
        metric_changed,
        api_changed,
        mapping_changed,
        dashboard_changed,
        docs_changed,
        semantic_changed,
        validation_results,
    )

    report_path = REPORT_DIR / f"weekly_semantic_audit_{slug}.md"
    lines: list[str] = [
        f"# Weekly Semantic Audit - {slug}",
        "",
        "## Audit Metadata",
        "",
        f"- Audit date: `{now.isoformat(timespec='seconds')}`",
        f"- Git branch: `{branch}`",
        f"- Latest commit hash: `{latest_commit}`",
        f"- Previous audit report: `{last_audit_path.relative_to(PROJECT_ROOT) if last_audit_path else 'none'}`",
        f"- Change detection basis: {comparison_basis}",
        "",
        "## Recent Commits",
        "",
        "```text",
        recent_commits or "No recent commits found.",
        "```",
        "",
        "## Git Status",
        "",
        "```text",
        git_status or "Clean worktree.",
        "```",
        "",
        "## Files Changed Since Last Audit",
        "",
        markdown_list(changed_files),
        "## CSV Schema Checks",
        "",
        build_schema_table(schema_checks),
        "## Missing Required Columns",
        "",
        markdown_list([*missing_files, *missing_required]),
        "## Model-Definition Files Changed",
        "",
        markdown_list(model_changed),
        "## Metric-Definition Files Changed",
        "",
        markdown_list(metric_changed),
        "## API / Source Files Changed",
        "",
        markdown_list(api_changed),
        "## Polymarket Mapping Files Changed",
        "",
        markdown_list(mapping_changed),
        "## Dashboard Files Changed",
        "",
        markdown_list(dashboard_changed),
        "## Documentation Files Changed",
        "",
        markdown_list(docs_changed),
        "## Semantic Layer Documentation Changed",
        "",
        markdown_list(semantic_changed),
        "## Validation Results",
        "",
        build_validation_section(validation_results),
        "## Recommended Semantic-Layer Updates",
        "",
        markdown_list(recommendations),
    ]

    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return report_path


def main() -> int:
    report_path = write_report()
    print(f"Wrote semantic audit report: {report_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
