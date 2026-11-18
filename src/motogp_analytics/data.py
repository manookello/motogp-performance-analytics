"""Download MotoGP-class results from the public MotoGP results API used by motogp.com.

The API (``https://api.motogp.pulselive.com/motogp/v1/results``) is **unofficial and
undocumented**. This module is deliberately polite: every response is cached to disk under
``data/raw/motogp`` (so each URL is fetched once), requests are throttled to ~2 per second, and
HTTP 429 / 5xx responses are retried with back-off.

Endpoint chain::

    seasons -> events?seasonUuid= -> categories?eventUuid= -> sessions?eventUuid=&categoryUuid=
            -> session/{id}/classification

Usage::

    python -m motogp_analytics.data --start 2013 --end 2026
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import time
from pathlib import Path

import pandas as pd
import requests

from motogp_analytics.config import (
    DATA_PROCESSED,
    DATA_RAW,
    FIRST_SEASON,
    LAST_SEASON,
    SESSION_TYPES,
)

BASE_URL = "https://api.motogp.pulselive.com/motogp/v1/results"


class MotoGPClient:
    def __init__(
        self,
        cache_dir: Path = DATA_RAW / "motogp",
        min_interval: float = 0.5,
        max_retries: int = 6,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers["User-Agent"] = (
            "motogp-performance-analytics/0.1 (open-source research)"
        )
        self._last_request = 0.0

    def get(self, path: str, params: dict | None = None, refresh: bool = False):
        params = params or {}
        key = path + "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        cache_file = self.cache_dir / f"{hashlib.sha1(key.encode()).hexdigest()}.json"
        if cache_file.exists() and not refresh:
            return json.loads(cache_file.read_text(encoding="utf-8"))

        for attempt in range(self.max_retries):
            wait = self.min_interval - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()
            try:
                resp = self.session.get(f"{BASE_URL}/{path}", params=params, timeout=30)
            except requests.RequestException:
                time.sleep(2 ** (attempt + 1))
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(2 ** (attempt + 2))
                continue
            if resp.status_code == 404:
                payload = None
            else:
                resp.raise_for_status()
                payload = resp.json()
            cache_file.write_text(json.dumps(payload), encoding="utf-8")
            return payload
        raise RuntimeError(f"Giving up on {key}")

    # Convenience wrappers -------------------------------------------------------------
    def seasons(self) -> list[dict]:
        return self.get("seasons", refresh=True)

    def events(self, season_id: str, refresh: bool = False) -> list[dict]:
        return self.get("events", {"seasonUuid": season_id, "isFinished": "true"}, refresh=refresh)

    def categories(self, event_id: str) -> list[dict]:
        return self.get("categories", {"eventUuid": event_id}) or []

    def sessions(self, event_id: str, category_id: str) -> list[dict]:
        return self.get("sessions", {"eventUuid": event_id, "categoryUuid": category_id}) or []

    def classification(self, session_id: str) -> dict | None:
        return self.get(f"session/{session_id}/classification", {"test": "false"})


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------


def lap_time_to_seconds(value) -> float | None:
    """'1:28.782' / '01:28.782' / '39:37.244' / '1:02:03.456' -> seconds."""
    if not value or not isinstance(value, str):
        return None
    try:
        parts = [float(p) for p in value.strip().split(":")]
    except ValueError:
        return None
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds


def _num(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def session_label(session: dict) -> str | None:
    stype, number = session.get("type"), session.get("number")
    if stype == "Q" and number in (1, 2):
        return f"Q{number}"
    if stype in ("SPR", "RAC"):
        return stype
    return None


def parse_classification(payload: dict | None, meta: dict) -> list[dict]:
    rows = []
    if not payload:
        return rows
    for entry in payload.get("classification") or []:
        rider = entry.get("rider") or {}
        team = entry.get("team") or {}
        constructor = entry.get("constructor") or {}
        best = entry.get("best_lap") or {}
        gap = entry.get("gap") or {}
        country = rider.get("country") or {}
        rows.append(
            {
                **meta,
                "position": entry.get("position"),
                "rider_id": rider.get("riders_api_uuid") or rider.get("id"),
                "rider_name": rider.get("full_name"),
                "rider_number": rider.get("number"),
                "rider_country": country.get("iso"),
                "team_name": team.get("name"),
                "constructor": constructor.get("name"),
                "points": _num(entry.get("points")) or 0.0,
                "status": entry.get("status"),
                "total_laps": entry.get("total_laps"),
                "time": entry.get("time"),
                "time_s": lap_time_to_seconds(entry.get("time")),
                "best_lap_s": lap_time_to_seconds(best.get("time")),
                "top_speed": _num(entry.get("top_speed")),
                "average_speed": _num(entry.get("average_speed")),
                "gap_first_s": _num(gap.get("first")),
                "gap_laps": _num(gap.get("lap")),
            }
        )
    return rows


def build_dataset(
    start: int = FIRST_SEASON,
    end: int = LAST_SEASON,
    client: MotoGPClient | None = None,
    out_dir: Path = DATA_PROCESSED,
    category_prefix: str = "MotoGP",
) -> dict[str, pd.DataFrame]:
    """Fetch MotoGP-class Q1/Q2/Sprint/Race classifications and write tidy parquet tables."""
    client = client or MotoGPClient()
    year_to_id = {s["year"]: s["id"] for s in client.seasons()}
    current_year = dt.date.today().year

    event_rows, session_rows, result_rows = [], [], []
    for year in range(start, end + 1):
        events = client.events(year_to_id[year], refresh=year >= current_year) or []
        races = sorted(
            (e for e in events if not e.get("test")), key=lambda e: e.get("date_start") or ""
        )
        for rnd, event in enumerate(races, start=1):
            cats = client.categories(event["id"])
            cat = next((c for c in cats if (c.get("name") or "").startswith(category_prefix)), None)
            if cat is None:
                continue
            circuit = event.get("circuit") or {}
            country = event.get("country") or {}
            event_rows.append(
                {
                    "season": year,
                    "round": rnd,
                    "event_id": event["id"],
                    "event_name": event.get("name"),
                    "event_short": event.get("short_name"),
                    "date_start": event.get("date_start"),
                    "date_end": event.get("date_end"),
                    "circuit_id": circuit.get("id"),
                    "circuit_name": circuit.get("name"),
                    "circuit_place": circuit.get("place"),
                    "country_iso": country.get("iso"),
                    "country_name": country.get("name"),
                }
            )
            for sess in client.sessions(event["id"], cat["id"]):
                label = session_label(sess)
                if label is None or sess.get("type") not in SESSION_TYPES:
                    continue
                cond = sess.get("condition") or {}
                session_rows.append(
                    {
                        "season": year,
                        "round": rnd,
                        "event_id": event["id"],
                        "session_id": sess["id"],
                        "session": label,
                        "session_date": sess.get("date"),
                        "track_condition": cond.get("track"),
                        "air_temp": _num(
                            str(cond.get("air", "")).replace("º", "").replace("°", "")
                        ),
                        "ground_temp": _num(
                            str(cond.get("ground", "")).replace("º", "").replace("°", "")
                        ),
                        "humidity": _num(str(cond.get("humidity", "")).replace("%", "")),
                        "weather": cond.get("weather"),
                    }
                )
                meta = {
                    "season": year,
                    "round": rnd,
                    "event_id": event["id"],
                    "session": label,
                    "session_id": sess["id"],
                    "session_date": sess.get("date"),
                }
                result_rows.extend(parse_classification(client.classification(sess["id"]), meta))
        print(f"{year}: {len(races)} events, {len(result_rows)} result rows so far", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        "events": pd.DataFrame(event_rows),
        "sessions": pd.DataFrame(session_rows),
        "results": pd.DataFrame(result_rows),
    }
    for name, df in frames.items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
    return frames


def load_tables(data_dir: Path = DATA_PROCESSED) -> dict[str, pd.DataFrame]:
    return {
        n: pd.read_parquet(data_dir / f"{n}.parquet") for n in ("events", "sessions", "results")
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=FIRST_SEASON)
    parser.add_argument("--end", type=int, default=LAST_SEASON)
    args = parser.parse_args()
    build_dataset(args.start, args.end)


if __name__ == "__main__":
    main()
