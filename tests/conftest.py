"""Shared test data: a mini-season of four made-up teams.

Results and stats are generated deterministically (fixed seed). Point-in-time
tests don't need hand-computed values — just a stable, complete season in the
raw football-data.co.uk CSV format.
"""
import numpy as np
import pandas as pd
import pytest

# Three pairing layouts; every three rounds they return with home/away swapped.
_PAIRINGS = [
    [("alfa", "bravo"), ("charlie", "delta")],
    [("alfa", "charlie"), ("bravo", "delta")],
    [("alfa", "delta"), ("bravo", "charlie")],
]


@pytest.fixture
def mini_season_raw():
    """Frame in raw CSV format: 12 rounds x 2 matches, each team plays 12 times
    — enough for the rolling6/rolling8 windows to carry real values."""
    rng = np.random.default_rng(7)
    rows = []
    for r in range(12):
        pairs = _PAIRINGS[r % 3]
        if (r // 3) % 2 == 1:
            pairs = [(away, home) for home, away in pairs]
        date = (pd.Timestamp("2024-08-10") + pd.Timedelta(weeks=r)).strftime("%d/%m/%Y")
        for home, away in pairs:
            fthg, ftag = rng.integers(0, 4), rng.integers(0, 4)
            rows.append({
                "Date": date, "Time": "15:00",
                "HomeTeam": home, "AwayTeam": away,
                "FTHG": fthg, "FTAG": ftag,
                "FTR": "H" if fthg > ftag else ("A" if ftag > fthg else "D"),
                "HS": rng.integers(5, 20), "AS": rng.integers(5, 20),
                "HST": rng.integers(1, 8), "AST": rng.integers(1, 8),
                "HC": rng.integers(0, 10), "AC": rng.integers(0, 10),
                "HY": rng.integers(0, 5), "AY": rng.integers(0, 5),
                "HR": rng.integers(0, 2), "AR": rng.integers(0, 2),
                "AvgH": round(rng.uniform(1.5, 4.0), 2),
                "AvgD": round(rng.uniform(2.8, 4.0), 2),
                "AvgA": round(rng.uniform(1.5, 6.0), 2),
            })
    return pd.DataFrame(rows)
