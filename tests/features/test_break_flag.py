"""match_after_intl_break: the first gameweek each league plays after a shared break.

The restart date is common to the leagues for a FIFA window but not for the 2022 world
cup, where england came back on boxing day and germany three weeks later. The flag is
therefore resolved per league, by gameweek rather than by date — that is what these
fixtures check, alongside the two pauses that must NOT count: a league-specific one and
the gap between seasons.
"""
import pandas as pd
import pytest

from features.match_info import _attach_break_flag


def _matches(rows):
    return pd.DataFrame(
        [{"league": lg, "season": s, "gameweek": gw, "date": pd.Timestamp(d)}
         for lg, s, gw, d in rows]
    )


@pytest.fixture
def breaks():
    return pd.DataFrame({
        "season": ["2425"],
        "last_match_before": [pd.Timestamp("2024-10-06")],
        "first_match_after": [pd.Timestamp("2024-10-18")],
        "type": ["international"],
    })


def test_flags_the_first_gameweek_after_the_break(breaks):
    out = _attach_break_flag(_matches([
        ("england", "2425", 7, "2024-10-05"),   # last round before
        ("england", "2425", 7, "2024-10-06"),
        ("england", "2425", 8, "2024-10-19"),   # first round after -> flagged
        ("england", "2425", 8, "2024-10-20"),
        ("england", "2425", 9, "2024-10-26"),   # the one after is not
    ]), breaks)
    assert list(out["match_after_intl_break"]) == [False, False, True, True, False]


def test_each_league_resolves_its_own_restart(breaks):
    """A league that comes back later still has its own next gameweek flagged, whole."""
    out = _attach_break_flag(_matches([
        ("england", "2425", 8, "2024-10-19"),
        ("germany", "2425", 7, "2024-11-08"),   # came back much later
        ("germany", "2425", 7, "2024-11-09"),   # same gameweek, still flagged
        ("germany", "2425", 8, "2024-11-16"),
    ]), breaks)
    assert list(out["match_after_intl_break"]) == [True, True, True, False]


def test_other_seasons_and_other_pauses_are_untouched(breaks):
    out = _attach_break_flag(_matches([
        ("england", "2324", 8, "2024-10-19"),   # right dates, wrong season
        ("england", "2425", 3, "2024-09-14"),   # before the break
    ]), breaks)
    assert not out["match_after_intl_break"].any()


def test_no_break_file_leaves_the_flag_off():
    empty = pd.DataFrame(columns=["season", "last_match_before", "first_match_after", "type"])
    out = _attach_break_flag(_matches([("england", "2425", 8, "2024-10-19")]), empty)
    assert list(out["match_after_intl_break"]) == [False]
