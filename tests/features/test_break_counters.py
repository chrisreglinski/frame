"""game_number_after_break / _before_break: each team's league matches counted from the
last break in which every league stopped, and back to the next one.

The counters are resolved per team on purpose. `gameweek` is only the sequential match
number bucketed by half the team count, so a rescheduled fixture leaves two teams in one
bucket with different counts — 10% of matches in the built ABT are such a case. The
fixtures below are built around exactly that, plus the world cup situation where the
leagues resume weeks apart.
"""
import pandas as pd
import pytest

from features.match_info import _attach_break_counters


def _matches(rows):
    return pd.DataFrame(
        [{"league": lg, "season": s, "hmt_name": h, "awt_name": a, "date": pd.Timestamp(d)}
         for lg, s, h, a, d in rows]
    )


@pytest.fixture
def breaks():
    return pd.DataFrame({
        "season": ["2425", "2425"],
        "last_match_before": [pd.Timestamp("2024-10-06"), pd.Timestamp("2024-11-10")],
        "first_match_after": [pd.Timestamp("2024-10-18"), pd.Timestamp("2024-11-22")],
        "type": ["international", "international"],
    })


def test_counts_run_forward_from_each_break(breaks):
    out = _attach_break_counters(_matches([
        ("england", "2425", "alfa", "bravo", "2024-10-19"),   # 1st after break 1
        ("england", "2425", "alfa", "charlie", "2024-10-26"),  # 2nd
        ("england", "2425", "alfa", "delta", "2024-11-02"),    # 3rd
        ("england", "2425", "alfa", "echo", "2024-11-23"),     # 1st after break 2
    ]), breaks)
    assert list(out["hmt_game_number_after_break"]) == [1, 2, 3, 1]


def test_counts_run_backward_to_the_next_break(breaks):
    out = _attach_break_counters(_matches([
        ("england", "2425", "alfa", "bravo", "2024-10-19"),
        ("england", "2425", "alfa", "charlie", "2024-10-26"),
        ("england", "2425", "alfa", "delta", "2024-11-02"),    # last before break 2
    ]), breaks)
    assert list(out["hmt_game_number_before_break"]) == [-3, -2, -1]


def test_edges_of_the_season_are_empty(breaks):
    out = _attach_break_counters(_matches([
        ("england", "2425", "alfa", "bravo", "2024-09-28"),   # before the first break
        ("england", "2425", "alfa", "charlie", "2024-11-30"),  # after the last one
    ]), breaks)
    assert out["hmt_game_number_after_break"].isna().tolist() == [True, False]
    assert out["hmt_game_number_before_break"].isna().tolist() == [False, True]


def test_the_two_sides_can_disagree(breaks):
    """alfa had a fixture rescheduled into the restart week, so by the time the two meet
    it is alfa's second match after the break and bravo's first."""
    out = _attach_break_counters(_matches([
        ("england", "2425", "alfa", "charlie", "2024-10-19"),  # alfa plays early
        ("england", "2425", "alfa", "bravo", "2024-10-26"),    # alfa 2nd, bravo 1st
    ]), breaks)
    assert out.loc[1, "hmt_game_number_after_break"] == 2
    assert out.loc[1, "awt_game_number_after_break"] == 1


def test_leagues_resume_independently(breaks):
    """The world cup case: one league comes back weeks after another, and each counts
    from its own first match."""
    out = _attach_break_counters(_matches([
        ("england", "2425", "alfa", "bravo", "2024-10-19"),
        ("germany", "2425", "gamma", "delta", "2024-11-09"),   # still its 1st after break 1
    ]), breaks)
    assert list(out["hmt_game_number_after_break"]) == [1, 1]


def test_no_break_file_leaves_both_counters_empty():
    empty = pd.DataFrame(columns=["season", "last_match_before", "first_match_after", "type"])
    out = _attach_break_counters(_matches([
        ("england", "2425", "alfa", "bravo", "2024-10-19"),
    ]), empty)
    assert out["hmt_game_number_after_break"].isna().all()
    assert out["hmt_game_number_before_break"].isna().all()
