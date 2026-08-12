"""Fatigue windows: the counts match hand-computed expectations.

Same spirit as test_window_semantics — the fixture is literal so every expected
number can be derived on paper in the comment next to it. What differs is the
kind of window: match-count windows there, calendar-time windows here, over a
fixture list that mixes competitions.

The last test guards a bug this builder actually had: the window sums are written
back positionally, so an implementation that returns its rows grouped rather than
in frame order silently attaches every count to the wrong fixture. Feeding the
same fixtures in a different row order must not change any value.
"""
import numpy as np
import pandas as pd
import pytest

from features.match_team_fatigue import _add_stats, _derive_flags

_WINDOWS = [6, 15, 45]

# alfa's fixture list. Gaps are round numbers of days so every window is obvious.
#                   ts                     stream    is_home  is_neutral
_ALFA = [
    ("2024-08-01 20:00", "league", True,  False),   # d0
    ("2024-08-04 20:00", "league", False, False),   # d1  +3d
    ("2024-08-08 20:00", "europe", False, False),   # d2  +4d
    ("2024-08-11 20:00", "league", True,  False),   # d3  +3d
    ("2024-08-16 20:00", "cup",    True,  True),    # d4  +5d, neutral venue
    ("2024-09-20 20:00", "league", True,  False),   # d5  +35d
]

# A second team in the same frame: its fixtures must not leak into alfa's counts.
_BRAVO = [
    ("2024-08-02 20:00", "league", False, False),
    ("2024-08-09 20:00", "league", True,  False),
    ("2024-08-30 20:00", "other",  True,  True),
]


def _frame(rows_by_team: dict[str, list]) -> pd.DataFrame:
    rows = [
        {"league": "testliga", "season": "2425", "team": team, "ts": pd.Timestamp(ts),
         "stream": stream, "is_home": is_home, "is_neutral": is_neutral, "match_id": None}
        for team, rows in rows_by_team.items()
        for ts, stream, is_home, is_neutral in rows
    ]
    return _derive_flags(pd.DataFrame(rows))


@pytest.fixture
def calendar():
    frame = _frame({"alfa": _ALFA, "bravo": _BRAVO})
    built = _add_stats(frame.sort_values(["team", "ts"]).reset_index(drop=True), _WINDOWS)
    return built[built["team"] == "alfa"].reset_index(drop=True)


def test_counts_only_earlier_fixtures(calendar):
    # 6d window at each of alfa's fixtures: d0 has no history; d1 sees d0 (3d back);
    # d2 sees d1 (4d); d3 sees d2 (3d); d4 sees d3 (5d); d5 sees nothing (35d gap).
    assert list(calendar["games_in_6d"]) == [0, 1, 1, 1, 1, 0]

    # 15d: history accumulates, then the 35d gap empties the window again.
    assert list(calendar["games_in_15d"]) == [0, 1, 2, 3, 4, 0]


def test_window_includes_its_left_edge(calendar):
    # d4 is 2024-08-16 20:00, so the 15d window opens at 2024-08-01 20:00 — exactly d0.
    # A left-closed window counts it: d0, d1, d2, d3 = 4.
    assert calendar.loc[4, "games_in_15d"] == 4

    # The current fixture is never its own history: d4 itself is not among those four.
    assert calendar.loc[4, "cup_games_in_15d"] == 0


def test_streams_split_the_total(calendar):
    # d4's 15d window holds d0, d1, d3 (league) and d2 (europe).
    assert calendar.loc[4, "league_games_in_15d"] == 3
    assert calendar.loc[4, "europe_games_in_15d"] == 1
    assert calendar.loc[4, "cup_games_in_15d"] == 0
    assert calendar.loc[4, "other_games_in_15d"] == 0

    streams = ["league", "cup", "europe", "other"]
    for x in _WINDOWS:
        total = sum(calendar[f"{s}_games_in_{x}d"] for s in streams)
        assert (calendar[f"games_in_{x}d"] == total).all()


def test_neutral_venue_counts_as_away_only_for_awayn(calendar):
    # d5's 45d window opens 2024-08-06 20:00 and holds d2 (europe, away),
    # d3 (league, home) and d4 (cup, neutral).
    assert calendar.loc[5, "games_in_45d"] == 3
    assert calendar.loc[5, "away_games_in_45d"] == 1     # d2 only
    assert calendar.loc[5, "awayn_games_in_45d"] == 2    # d2 + the neutral d4


def test_rest_hours_and_last_match_flags(calendar):
    # Gaps of 3, 4, 3, 5 days, then 35 days which the 200h cap flattens.
    assert np.isnan(calendar.loc[0, "rest_hours"])
    assert list(calendar["rest_hours"][1:]) == [72.0, 96.0, 72.0, 120.0, 200.0]

    # Previous fixture: d1 away, d2 away, d4 neutral (so not "away"), d2 european.
    assert list(calendar["last_match_is_away"][1:]) == [False, True, True, False, False]
    assert list(calendar["last_match_is_europe"][1:]) == [False, False, True, False, False]


def test_values_do_not_depend_on_row_order(calendar):
    """Same fixtures, teams interleaved by kick-off instead of blocked by team."""
    frame = _frame({"alfa": _ALFA, "bravo": _BRAVO})
    interleaved = _add_stats(frame.sort_values("ts").reset_index(drop=True), _WINDOWS)

    other = (interleaved[interleaved["team"] == "alfa"]
             .sort_values("ts").reset_index(drop=True))
    counted = [c for c in calendar.columns if "_in_" in c] + ["rest_hours"]
    pd.testing.assert_frame_equal(calendar[counted], other[counted])
