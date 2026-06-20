from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


DEFAULT_FOOTBALL_API_URL = "https://api.football-data.org/v4"


def main() -> None:
    try:
        from dotenv import load_dotenv
        import requests
    except Exception as exc:
        print(f"Import error: {exc}")
        return

    parser = argparse.ArgumentParser(description="Debug football-data.org API configuration and responses.")
    parser.add_argument("--start-date", default="2022-11-20")
    parser.add_argument("--end-date", default="2022-11-29")
    parser.add_argument("--competitions", default="WC", help="football-data.org competition code(s), for example WC or WC,EC.")
    args = parser.parse_args()

    try:
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        import os

        configured_url = os.getenv("FOOTBALL_API_URL")
        configured_key = os.getenv("FOOTBALL_API_KEY")
        api_url = (configured_url or DEFAULT_FOOTBALL_API_URL).rstrip("/")
        endpoint = f"{api_url}/matches" if not api_url.endswith("/matches") else api_url

        print(f"FOOTBALL_API_URL configured: {bool(configured_url)}")
        print(f"FOOTBALL_API_KEY configured: {bool(configured_key)}")
        print(f"Request endpoint: {endpoint}")
        print(f"Date range: {args.start_date} to {args.end_date}")
        print(f"Competitions: {args.competitions}")

        params = {
            "dateFrom": args.start_date,
            "dateTo": args.end_date,
            "status": "FINISHED",
        }
        if args.competitions:
            params["competitions"] = args.competitions

        headers = {}
        if configured_key:
            headers["X-Auth-Token"] = configured_key

        try:
            response = requests.get(endpoint, params=params, headers=headers, timeout=25)
        except Exception as exc:
            print(f"Request failed: {exc}")
            return

        print(f"Status code: {response.status_code}")
        text = response.text or ""
        print("First 1000 response characters:")
        print(text[:1000])

        try:
            payload = response.json()
        except Exception as exc:
            print(f"JSON parse failed: {exc}")
            return

        if isinstance(payload, dict):
            print(f"Top-level JSON keys: {list(payload.keys())}")
            matches = payload.get("matches")
            if isinstance(matches, list):
                print(f"matches count: {len(matches)}")
                if matches:
                    first = matches[0]
                    competition = first.get("competition") if isinstance(first, dict) else {}
                    home = first.get("homeTeam") if isinstance(first, dict) else {}
                    away = first.get("awayTeam") if isinstance(first, dict) else {}
                    score = first.get("score") if isinstance(first, dict) else {}
                    print(
                        "first match summary: "
                        f"id={first.get('id', '')}, "
                        f"utcDate={first.get('utcDate', '')}, "
                        f"competition={(competition or {}).get('name', '')}, "
                        f"home={(home or {}).get('name', '')}, "
                        f"away={(away or {}).get('name', '')}, "
                        f"score={(score or {}).get('fullTime', {})}"
                    )
        else:
            print(f"JSON type: {type(payload).__name__}")
    except Exception as exc:
        print(f"Diagnostic failed safely: {exc}")


if __name__ == "__main__":
    main()
