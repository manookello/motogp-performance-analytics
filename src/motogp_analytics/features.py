"""Turn raw session classifications into one tidy row per (race, rider) with pre-race features.

Conventions:

* **Official race** – when a race is red-flagged and restarted, the API returns two ``RAC``
  sessions; the later one is the official classification.
* **Grid** – reconstructed from qualifying: Q2 positions 1-12, then Q1 positions 3+ as 13th
  onwards. Post-qualifying grid penalties are not in the API and are ignored.
* **Finishing position** – classified riders keep their position; non-finishers are placed
  behind them, ordered by laps completed (needed to rank a whole field).
* All form features use ``shift(1)``: only races strictly before the current one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MANUFACTURERS = ["Ducati", "Honda", "Yamaha", "KTM", "Aprilia", "Suzuki"]

FEATURES = [
    "grid",
    "quali_gap_pct",
    "sprint_position",
    "sprint_dnf",
    "rdr_form_3",
    "rdr_form_5",
    "rdr_points_form_5",
    "rdr_podium_rate_10",
    "rdr_dnf_rate_10",
    "rdr_starts",
    "rdr_champ_rank_before",
    "rdr_points_share_before",
    "mfr_points_per_bike_5",
    "mfr_best_finish_5",
    "rdr_circuit_avg_finish",
    "rdr_circuit_starts",
    "wet",
    "ground_temp",
    "round",
]

DNF_FEATURES = [
    "grid",
    "quali_gap_pct",
    "sprint_dnf",
    "rdr_dnf_rate_10",
    "rdr_dnf_rate_30",
    "rdr_starts",
    "mfr_dnf_rate_10",
    "circuit_dnf_rate",
    "wet",
    "air_temp",
    "ground_temp",
    "humidity",
    "round",
]

FEATURE_LABELS = {
    "grid": "Grid position",
    "quali_gap_pct": "Qualifying gap to pole (%)",
    "sprint_position": "Sprint finish (same weekend)",
    "sprint_dnf": "Crashed out of the sprint",
    "rdr_form_3": "Rider avg finish, last 3",
    "rdr_form_5": "Rider avg finish, last 5",
    "rdr_points_form_5": "Rider avg points, last 5",
    "rdr_podium_rate_10": "Rider podium rate, last 10",
    "rdr_dnf_rate_10": "Rider DNF rate, last 10",
    "rdr_dnf_rate_30": "Rider DNF rate, last 30",
    "rdr_starts": "Rider career starts",
    "rdr_champ_rank_before": "Rider championship rank",
    "rdr_points_share_before": "Rider points vs leader",
    "mfr_points_per_bike_5": "Manufacturer points per bike, last 5",
    "mfr_best_finish_5": "Manufacturer best finish, last 5",
    "mfr_dnf_rate_10": "Manufacturer DNF rate, last 10",
    "rdr_circuit_avg_finish": "Rider avg finish at circuit",
    "rdr_circuit_starts": "Rider starts at circuit",
    "circuit_dnf_rate": "Circuit DNF rate (history)",
    "wet": "Wet race",
    "air_temp": "Air temperature",
    "ground_temp": "Track temperature",
    "humidity": "Humidity",
    "round": "Round of season",
}


def manufacturer(name: str | None) -> str:
    """Collapse CRT / Open-class chassis names (FTR, ART, Suter, ...) into 'Other'."""
    return name if name in MANUFACTURERS else "Other"


def official_sessions(results: pd.DataFrame, session: str) -> pd.DataFrame:
    """Rows of the last session of type ``session`` per event (handles restarted races)."""
    s = results[results["session"] == session]
    last = s.sort_values("session_date").groupby("event_id")["session_id"].last()
    return s[s["session_id"].isin(last)].copy()


def finishing_order(race: pd.DataFrame) -> pd.Series:
    """Classified position, then non-finishers ordered by laps completed."""
    classified = race["position"].notna()
    n_classified = classified.groupby(race["session_id"]).transform("sum")
    dnf_rank = (
        race["total_laps"]
        .where(~classified)
        .groupby(race["session_id"])
        .rank(ascending=False, method="first")
    )
    return race["position"].where(classified, n_classified + dnf_rank)


def build_grid(results: pd.DataFrame) -> pd.DataFrame:
    q1 = official_sessions(results, "Q1")
    q2 = official_sessions(results, "Q2")
    g2 = q2[["event_id", "rider_id", "position", "best_lap_s"]].assign(
        grid=q2["position"], session="Q2"
    )
    q1_out = q1[q1["position"] >= 3]
    g1 = q1_out[["event_id", "rider_id", "position", "best_lap_s"]].assign(
        grid=q1_out["position"] + 10, session="Q1"
    )
    grid = pd.concat([g2, g1]).drop_duplicates(["event_id", "rider_id"], keep="first")
    best = (
        pd.concat([q1, q2])
        .groupby(["event_id", "rider_id"])["best_lap_s"]
        .min()
        .rename("best_quali_s")
    )
    grid = grid.drop(columns=["best_lap_s", "position", "session"]).join(
        best, on=["event_id", "rider_id"]
    )
    pole = grid.groupby("event_id")["best_quali_s"].transform("min")
    grid["quali_gap_pct"] = (grid["best_quali_s"] / pole - 1) * 100
    return grid


def _rolling_prev(s: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
    return s.shift(1).rolling(window, min_periods=min_periods).mean()


def build_features(
    events: pd.DataFrame, sessions: pd.DataFrame, results: pd.DataFrame
) -> pd.DataFrame:
    race = official_sessions(results, "RAC")
    race["finish_position"] = finishing_order(race)
    race["dnf"] = race["position"].isna().astype(int)
    race["podium"] = (race["position"] <= 3).astype(int)
    race["win"] = (race["position"] == 1).astype(int)
    race["manufacturer"] = race["constructor"].map(manufacturer)

    ev = events[
        ["event_id", "event_short", "event_name", "date_start", "circuit_id", "circuit_name"]
    ]
    df = race.merge(ev, on="event_id", how="left")
    cond = sessions[
        ["session_id", "track_condition", "weather", "air_temp", "ground_temp", "humidity"]
    ]
    df = df.merge(cond, on="session_id", how="left")
    df["wet"] = df["track_condition"].isin(["Wet", "Wet-Dry"]).astype(int)
    df["date"] = pd.to_datetime(df["date_start"])
    df["race_id"] = df["season"].astype(str) + "-" + df["round"].astype(str).str.zfill(2)

    # grid & qualifying
    df = df.merge(build_grid(results), on=["event_id", "rider_id"], how="left")
    n_riders = df.groupby("race_id")["rider_id"].transform("count")
    df["grid"] = df["grid"].fillna(n_riders)

    # sprint (Saturday, before the race)
    spr = official_sessions(results, "SPR")
    spr = spr.assign(
        sprint_position=spr["position"],
        sprint_dnf=spr["position"].isna().astype(int),
        sprint_points=spr["points"],
    )[["event_id", "rider_id", "sprint_position", "sprint_dnf", "sprint_points"]]
    df = df.merge(spr, on=["event_id", "rider_id"], how="left")
    df["sprint_points"] = df["sprint_points"].fillna(0.0)

    # rider rolling form
    df = df.sort_values(["date", "rider_id"], kind="stable").reset_index(drop=True)
    by_rdr = df.groupby("rider_id", sort=False)
    df["rdr_form_3"] = by_rdr["finish_position"].transform(lambda s: _rolling_prev(s, 3))
    df["rdr_form_5"] = by_rdr["finish_position"].transform(lambda s: _rolling_prev(s, 5))
    df["rdr_points_form_5"] = by_rdr["points"].transform(lambda s: _rolling_prev(s, 5))
    df["rdr_podium_rate_10"] = by_rdr["podium"].transform(lambda s: _rolling_prev(s, 10))
    df["rdr_dnf_rate_10"] = by_rdr["dnf"].transform(lambda s: _rolling_prev(s, 10, 3))
    df["rdr_dnf_rate_30"] = by_rdr["dnf"].transform(lambda s: _rolling_prev(s, 30, 5))
    df["rdr_starts"] = by_rdr.cumcount()

    # championship before the weekend
    df["weekend_points"] = df["points"] + df["sprint_points"]
    season_rdr = df.groupby(["season", "rider_id"], sort=False)["weekend_points"]
    df["rdr_points_before"] = season_rdr.cumsum() - df["weekend_points"]
    leader = df.groupby("race_id")["rdr_points_before"].transform("max")
    df["rdr_points_share_before"] = np.where(leader > 0, df["rdr_points_before"] / leader, np.nan)
    df["rdr_champ_rank_before"] = df.groupby("race_id")["rdr_points_before"].rank(
        ascending=False, method="min"
    )
    df.loc[df["round"] == 1, "rdr_champ_rank_before"] = np.nan

    # manufacturer form (per bike, so 8-bike Ducati and 2-bike Suzuki are comparable)
    mfr = (
        df.groupby(["race_id", "date", "manufacturer"], as_index=False)
        .agg(
            pts_per_bike=("points", "mean"),
            best_finish=("finish_position", "min"),
            dnf_rate=("dnf", "mean"),
        )
        .sort_values(["date", "manufacturer"])
    )
    by_mfr = mfr.groupby("manufacturer", sort=False)
    mfr["mfr_points_per_bike_5"] = by_mfr["pts_per_bike"].transform(lambda s: _rolling_prev(s, 5))
    mfr["mfr_best_finish_5"] = by_mfr["best_finish"].transform(lambda s: _rolling_prev(s, 5))
    mfr["mfr_dnf_rate_10"] = by_mfr["dnf_rate"].transform(lambda s: _rolling_prev(s, 10))
    df = df.merge(
        mfr[
            [
                "race_id",
                "manufacturer",
                "mfr_points_per_bike_5",
                "mfr_best_finish_5",
                "mfr_dnf_rate_10",
            ]
        ],
        on=["race_id", "manufacturer"],
        how="left",
    )

    # circuit history
    df = df.sort_values(["date", "rider_id"], kind="stable").reset_index(drop=True)
    by_rc = df.groupby(["rider_id", "circuit_id"], sort=False)["finish_position"]
    df["rdr_circuit_avg_finish"] = by_rc.transform(lambda s: s.shift(1).expanding().mean())
    df["rdr_circuit_starts"] = by_rc.cumcount()
    circ = (
        df.groupby(["race_id", "date", "circuit_id"], as_index=False)["dnf"]
        .mean()
        .sort_values("date")
    )
    circ["circuit_dnf_rate"] = circ.groupby("circuit_id")["dnf"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    df = df.merge(circ[["race_id", "circuit_dnf_rate"]], on="race_id", how="left")

    return df.sort_values(["date", "finish_position"], kind="stable").reset_index(drop=True)
