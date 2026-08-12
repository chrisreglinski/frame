"""Fatigue windows and decayed load: the values match hand-computed expectations.

Same spirit as test_window_semantics — the fixture is literal so every expected
number can be derived on paper in the comment next to it. What differs is the
kind of window: match-count windows there, calendar-time windows here, over a
fixture list that mixes competitions and venues.

Two tests guard bugs this builder actually had. The row-order test covers window
sums written back positionally: an implementation that returns its rows grouped
rather than in frame order silently attaches every count to the wrong fixture.
The decay test recomputes the load from its definition (a direct sum over all
earlier fixtures) and compares it with the O(n) recursion the builder uses.
"""
import numpy as np
import pandas as pd
import pytest

from features.match_team_fatigue import _add_stats, _derive_flags

_WINDOWS = [6, 15, 45]
_TAUS_EXP = [7]
_TAUS_GAUSS = [4]

# alfa's fixture list. Gaps are round numbers of days so every window is obvious,
# and between them they cover all three venues and all three competitions.
#                   ts                     stream    home   neutral abroad -> venue
_ALFA = [
    ("2024-08-01 20:00", "league", True,  False, False),  # d0      home
    ("2024-08-04 20:00", "league", False, False, False),  # d1 +3d  domestic
    ("2024-08-08 20:00", "europe", False, False, True),   # d2 +4d  abroad
    ("2024-08-11 20:00", "league", True,  False, False),  # d3 +3d  home
    ("2024-08-16 20:00", "other",  True,  True,  False),  # d4 +5d  domestic
    ("2024-09-20 20:00", "league", True,  False, False),  # d5 +35d home
]

# A second team in the same frame: its fixtures must not leak into alfa's counts.
_BRAVO = [
    ("2024-08-02 20:00", "league", False, False, False),
    ("2024-08-09 20:00", "league", True,  False, False),
    ("2024-08-30 20:00", "other",  True,  True,  True),
]


def _frame(rows_by_team: dict[str, list]) -> pd.DataFrame:
    rows = [
        {"league": "testliga", "season": "2425", "team": team, "ts": pd.Timestamp(ts),
         "stream": stream, "is_home": is_home, "is_neutral": is_neutral,
         "is_abroad": is_abroad, "match_id": None}
        for team, fixtures in rows_by_team.items()
        for ts, stream, is_home, is_neutral, is_abroad in fixtures
    ]
    return _derive_flags(pd.DataFrame(rows))


@pytest.fixture
def calendar():
    frame = _frame({"alfa": _ALFA, "bravo": _BRAVO})
    built = _add_stats(frame.sort_values(["team", "ts"]).reset_index(drop=True),
                       _WINDOWS, _TAUS_EXP, _TAUS_GAUSS)
    return built[built["team"] == "alfa"].reset_index(drop=True)


def test_fixtures_land_on_the_right_axes(calendar):
    assert list(calendar["venue"]) == [
        "home", "domestic", "abroad", "home", "domestic", "home",
    ]
    assert list(calendar["cell"]) == [
        "league_home", "league_domestic", "europe_abroad",
        "league_home", "other_domestic", "league_home",
    ]


def test_every_venue_is_reachable_in_every_stream():
    """A fixture is home, or away in its own country (domestic), or out of it (abroad).
    The league is never played abroad, so that combination cannot occur."""
    fixtures = [
        ("league", True,  False, False, "home"),
        ("league", False, False, False, "domestic"),
        ("europe", True,  False, False, "home"),
        ("europe", False, False, False, "domestic"),   # away leg, same country
        ("europe", False, False, True,  "abroad"),
        ("other",  True,  False, False, "home"),
        ("other",  True,  True,  False, "domestic"),   # neutral final, home soil
        ("other",  True,  True,  True,  "abroad"),     # neutral final, abroad
    ]
    rows = [("2024-08-%02d 20:00" % (i + 1), stream, home, neutral, abroad)
            for i, (stream, home, neutral, abroad, _) in enumerate(fixtures)]
    built = _derive_flags(_frame({"alfa": rows}))
    assert list(built["venue"]) == [expected for *_, expected in fixtures]


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
    assert calendar.loc[4, "other_games_in_15d"] == 0


def test_both_axes_split_the_total(calendar):
    # d4's 15d window holds d0 and d3 (league at home), d1 (league away in the country)
    # and d2 (a european trip abroad).
    assert calendar.loc[4, "home_games_in_15d"] == 2
    assert calendar.loc[4, "domestic_games_in_15d"] == 1
    assert calendar.loc[4, "abroad_games_in_15d"] == 1
    assert calendar.loc[4, "league_games_in_15d"] == 3
    assert calendar.loc[4, "europe_games_in_15d"] == 1
    assert calendar.loc[4, "other_games_in_15d"] == 0

    for x in _WINDOWS:
        for axis in (["home", "domestic", "abroad"], ["league", "europe", "other"]):
            total = sum(calendar[f"{k}_games_in_{x}d"] for k in axis)
            assert (calendar[f"games_in_{x}d"] == total).all()


def test_neutral_home_country_fixture_is_domestic(calendar):
    # d4 is a neutral-venue final on home soil: not a home match, not abroad.
    # d5's 45d window opens 2024-08-06 20:00 and holds d2, d3 and d4.
    assert calendar.loc[5, "games_in_45d"] == 3
    assert calendar.loc[5, "other_games_in_45d"] == 1
    assert calendar.loc[5, "domestic_games_in_45d"] == 1
    assert calendar.loc[5, "abroad_games_in_45d"] == 1
    assert calendar.loc[5, "home_games_in_45d"] == 1


def test_gaps_to_the_two_previous_fixtures(calendar):
    # Gaps of 3, 4, 3, 5 days, then 35 days which the 200h cap flattens.
    assert np.isnan(calendar.loc[0, "hours_since_last_match"])
    assert list(calendar["hours_since_last_match"][1:]) == [72.0, 96.0, 72.0, 120.0, 200.0]

    # Two fixtures back: 7d, 7d, 8d, then 40d flattened by the higher 500h cap.
    # d1 has only one earlier fixture, so it has no second gap.
    assert calendar["hours_since_2nd_last_match"][:2].isna().all()
    assert list(calendar["hours_since_2nd_last_match"][2:]) == [168.0, 168.0, 192.0, 500.0]


def test_last_match_flags(calendar):
    # Previous fixture: d1 away, d2 away, d4 neutral (so not "away"), d2 european.
    assert list(calendar["last_match_is_away"][1:]) == [False, True, True, False, False]
    assert list(calendar["last_match_is_europe"][1:]) == [False, False, True, False, False]


def test_decayed_load_matches_its_definition(calendar):
    """The builder uses an O(n) recursion; here the load is summed directly instead."""
    ts = calendar["ts"].to_numpy()
    expected = []
    for i in range(len(ts)):
        ages = (ts[i] - ts[:i]) / np.timedelta64(1, "D")
        expected.append(np.exp(-ages / 7).sum())

    assert calendar["games_load_7d"].to_numpy() == pytest.approx(expected)

    # The first gap is 3 days, so d1 carries exactly one decayed fixture.
    assert calendar.loc[1, "games_load_7d"] == pytest.approx(np.exp(-3 / 7))

    # The axes split the load the same way they split the counts.
    for axis in (["home", "domestic", "abroad"], ["league", "europe", "other"]):
        parts = sum(calendar[f"{k}_games_load_7d"] for k in axis)
        assert calendar["games_load_7d"].to_numpy() == pytest.approx(parts.to_numpy())


def test_gaussian_load_matches_its_definition(calendar):
    """Same check for the squared kernel, which has no recursion to lean on."""
    ts = calendar["ts"].to_numpy()
    expected = []
    for i in range(len(ts)):
        ages = (ts[i] - ts[:i]) / np.timedelta64(1, "D")
        expected.append(np.exp(-((ages / 4) ** 2)).sum())

    assert calendar["games_load_gauss_4d"].to_numpy() == pytest.approx(expected)

    # A fixture exactly tau days old contributes 1/e under either kernel; the shapes
    # differ elsewhere, so d1 (3 days back) is heavier here than under exp(-age/4).
    assert calendar.loc[1, "games_load_gauss_4d"] == pytest.approx(np.exp(-((3 / 4) ** 2)))
    assert calendar.loc[1, "games_load_gauss_4d"] > np.exp(-3 / 4)


def test_values_do_not_depend_on_row_order(calendar):
    """Same fixtures, teams interleaved by kick-off instead of blocked by team."""
    frame = _frame({"alfa": _ALFA, "bravo": _BRAVO})
    interleaved = _add_stats(frame.sort_values("ts").reset_index(drop=True),
                             _WINDOWS, _TAUS_EXP, _TAUS_GAUSS)

    other = (interleaved[interleaved["team"] == "alfa"]
             .sort_values("ts").reset_index(drop=True))
    counted = ([c for c in calendar.columns if "_in_" in c or "_load_" in c]
               + ["hours_since_last_match", "hours_since_2nd_last_match"])
    pd.testing.assert_frame_equal(calendar[counted], other[counted])
