"""Window semantics: the computed values match hand-computed expectations.

Complements the leakage test: that one checks the *direction* of information
flow, this one checks the *arithmetic* — a rolling6 that quietly averaged five
matches would still pass the perturbation test. One assertion per class of
logic, each expected number derived on paper in the comment next to it.

The fixture is deliberately literal (no seed, no generation): team "alfa"
plays 8 home matches with goals 1..8, so every window value is obvious.
"""
import pandas as pd
import pytest
from pytest import approx

from features.match_team_stats import _add_stats, _build_long

#            match:   1    2    3    4    5    6    7    8
_GOALS_FOR        = [  1,   2,   3,   4,   5,   6,   7,   8]
_SHOTS_FOR        = [ 10,   5,   8,   8,   8,   8,   8,   8]
_SHOTS_ON_TARGET  = [  1,   5,   2,   2,   2,   2,   2,   2]
_RED_FOR          = [  1,   0,   0,   0,   0,   0,   0,   0]


@pytest.fixture
def alfa():
    rows = []
    for k in range(8):
        date = (pd.Timestamp("2024-08-10") + pd.Timedelta(weeks=k)).strftime("%d/%m/%Y")
        rows.append({
            "Date": date, "Time": "15:00",
            "HomeTeam": "alfa", "AwayTeam": f"opponent{k}",
            "FTHG": _GOALS_FOR[k], "FTAG": 0, "FTR": "H",
            "HS": _SHOTS_FOR[k], "AS": 7,
            "HST": _SHOTS_ON_TARGET[k], "AST": 3,
            "HC": 4, "AC": 4, "HY": 1, "AY": 1,
            "HR": _RED_FOR[k], "AR": 0,
            "AvgH": 2.0, "AvgD": 3.5, "AvgA": 4.0,
        })
    raw = pd.DataFrame(rows)
    long = _add_stats(_build_long(raw, "testliga", "2425"))
    # one row per alfa match, indexed by game_number (1..8)
    return long[long["team"] == "alfa"].sort_values("date").set_index("game_number")


def test_season_expanding_mean(alfa):
    # match 4 sees matches 1-3: mean(1, 2, 3) = 2.0
    assert alfa.loc[4, "season_goals_for_avg"] == approx(2.0)
    # match 1 has no history: NaN, not 0
    assert pd.isna(alfa.loc[1, "season_goals_for_avg"])


def test_rolling6_is_exactly_six_matches(alfa):
    # match 7 sees matches 1-6: mean(1..6) = 3.5
    assert alfa.loc[7, "rolling6_goals_for_avg"] == approx(3.5)
    # match 8 sees matches 2-7 (window slides, match 1 drops out): mean(2..7) = 4.5
    assert alfa.loc[8, "rolling6_goals_for_avg"] == approx(4.5)


def test_rolling_masked_without_full_history(alfa):
    # rolling6 needs 6 played matches: NaN through match 6, real from match 7
    assert alfa.loc[1:6, "rolling6_goals_for_avg"].isna().all()
    assert alfa.loc[7:8, "rolling6_goals_for_avg"].notna().all()
    # rolling8 never has 8 prior matches in an 8-match season: NaN everywhere
    assert alfa["rolling8_goals_for_avg"].isna().all()


def test_std_uses_population_ddof0(alfa):
    # match 4, goals_diff history 1, 2, 3: ddof=0 gives sqrt(2/3) ~ 0.8165
    # (ddof=1 would give 1.0 — this pins the convention)
    assert alfa.loc[4, "season_goals_diff_std"] == approx((2 / 3) ** 0.5)


def test_shots_ratio_is_sum_over_sum(alfa):
    # match 3: (1 + 5) / (10 + 5) = 0.4 — NOT the mean of per-match
    # ratios (0.1 + 1.0) / 2 = 0.55; the contract is sum/sum
    assert alfa.loc[3, "season_shots_on_target_for_ratio"] == approx(0.4)


def test_red_last_match_flag(alfa):
    assert pd.isna(alfa.loc[1, "red_last_match"])   # no previous match
    assert alfa.loc[2, "red_last_match"] == True    # red card in match 1
    assert alfa.loc[3, "red_last_match"] == False   # clean match 2
