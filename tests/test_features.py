import numpy as np
import pandas as pd
import pytest

from motogp_analytics.data import lap_time_to_seconds, session_label
from motogp_analytics.features import (
    build_features,
    build_grid,
    finishing_order,
    manufacturer,
    official_sessions,
)


def test_lap_time_to_seconds():
    assert lap_time_to_seconds("1:28.782") == pytest.approx(88.782)
    assert lap_time_to_seconds("01:28.782") == pytest.approx(88.782)
    assert lap_time_to_seconds("39:37.244") == pytest.approx(2377.244)
    assert lap_time_to_seconds(None) is None


def test_session_label():
    assert session_label({"type": "Q", "number": 2}) == "Q2"
    assert session_label({"type": "RAC", "number": None}) == "RAC"
    assert session_label({"type": "FP", "number": 1}) is None


def test_manufacturer_grouping():
    assert manufacturer("Ducati") == "Ducati"
    assert manufacturer("FTR Kawasaki") == "Other"
    assert manufacturer(None) == "Other"


def test_official_session_keeps_restart():
    res = pd.DataFrame(
        {
            "event_id": ["e", "e", "e", "e"],
            "session": ["RAC"] * 4,
            "session_id": ["first", "first", "restart", "restart"],
            "session_date": [
                "2020-01-01T10:00",
                "2020-01-01T10:00",
                "2020-01-01T11:00",
                "2020-01-01T11:00",
            ],
            "rider_id": ["a", "b", "a", "b"],
        }
    )
    out = official_sessions(res, "RAC")
    assert set(out["session_id"]) == {"restart"}


def test_finishing_order_places_dnfs_last_by_laps():
    race = pd.DataFrame(
        {
            "session_id": ["s"] * 4,
            "position": [1.0, 2.0, np.nan, np.nan],
            "total_laps": [25, 25, 3, 20],
        }
    )
    assert finishing_order(race).tolist() == [1.0, 2.0, 4.0, 3.0]


def _q(session, event, riders, positions, laps):
    return pd.DataFrame(
        {
            "event_id": event,
            "session": session,
            "session_id": f"{event}-{session}",
            "session_date": "2024-01-01",
            "rider_id": riders,
            "position": positions,
            "best_lap_s": laps,
        }
    )


def test_grid_from_q1_q2():
    q2 = _q(
        "Q2",
        "e",
        [f"r{i}" for i in range(1, 13)],
        list(range(1, 13)),
        [90 + i * 0.1 for i in range(12)],
    )
    q1 = _q("Q1", "e", ["r11", "r12", "r13", "r14"], [1, 2, 3, 4], [91.0, 91.1, 91.2, 91.3])
    grid = build_grid(pd.concat([q1, q2])).set_index("rider_id")
    assert grid.loc["r1", "grid"] == 1
    assert grid.loc["r12", "grid"] == 12  # advanced from Q1, grid from Q2
    assert grid.loc["r13", "grid"] == 13
    assert grid.loc["r14", "grid"] == 14
    assert grid.loc["r1", "quali_gap_pct"] == pytest.approx(0.0)


def _season(n_races=5):
    riders = ["a", "b", "c"]
    events, sessions, results = [], [], []
    rng = np.random.default_rng(1)
    for rnd in range(1, n_races + 1):
        eid = f"e{rnd}"
        events.append(
            {
                "event_id": eid,
                "event_short": f"E{rnd}",
                "event_name": f"GP {rnd}",
                "date_start": f"2024-0{rnd}-01",
                "circuit_id": "c",
                "circuit_name": "C",
            }
        )
        sid = f"{eid}-RAC"
        sessions.append(
            {
                "session_id": sid,
                "track_condition": "Dry",
                "weather": "Clear",
                "air_temp": 25.0,
                "ground_temp": 35.0,
                "humidity": 50.0,
            }
        )
        order = rng.permutation(3)
        for pos, i in enumerate(order, start=1):
            results.append(
                {
                    "season": 2024,
                    "round": rnd,
                    "event_id": eid,
                    "session": "RAC",
                    "session_id": sid,
                    "session_date": f"2024-0{rnd}-01",
                    "position": float(pos),
                    "rider_id": riders[i],
                    "rider_name": riders[i].upper(),
                    "rider_number": i,
                    "team_name": "T",
                    "constructor": "Ducati",
                    "points": [25.0, 20.0, 16.0][pos - 1],
                    "status": "INSTND",
                    "total_laps": 20,
                    "best_lap_s": np.nan,
                }
            )
    return pd.DataFrame(events), pd.DataFrame(sessions), pd.DataFrame(results)


def test_rider_form_uses_only_past_races():
    feats = build_features(*_season())
    for rider in "abc":
        d = feats[feats["rider_id"] == rider].sort_values("round")
        finishes = d["finish_position"].to_numpy()
        assert np.isnan(d["rdr_form_3"].iloc[0])
        for i in range(1, len(d)):
            assert d["rdr_form_3"].iloc[i] == pytest.approx(finishes[max(0, i - 3) : i].mean())
        cum_before = d["points"].cumsum().shift(1).fillna(0).to_numpy()
        assert np.allclose(d["rdr_points_before"].to_numpy(), cum_before)
