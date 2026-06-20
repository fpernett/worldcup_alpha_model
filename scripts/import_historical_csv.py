from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.historical_csv_importer import HistoricalCsvImportError, import_historical_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Import local historical results CSV into long-format match history.")
    parser.add_argument("--input", required=True, help="Path to source results CSV.")
    parser.add_argument("--source", default="local_csv", help="Source label written to historical rows.")
    parser.add_argument("--teams", nargs="*", default=None, help="Optional team filter. Includes matches where either side matches.")
    parser.add_argument("--rebuild-behavior", action="store_true", help="Rebuild data/team_behavior.csv after import.")
    args = parser.parse_args()

    try:
        diagnostics = import_historical_csv(
            input_path=args.input,
            source=args.source,
            teams=args.teams,
            rebuild_behavior=args.rebuild_behavior,
        )
    except HistoricalCsvImportError as exc:
        print(f"status: failed")
        print(f"error: {exc}")
        raise SystemExit(1)
    except Exception as exc:
        print(f"status: failed")
        print(f"error: {exc}")
        raise SystemExit(1)

    for key, value in diagnostics.items():
        if isinstance(value, list):
            print(f"{key}: {', '.join(str(item) for item in value) if value else ''}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
