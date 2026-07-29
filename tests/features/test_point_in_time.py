"""Leakage test: features may depend only on matches before the match date.

Perturbation pattern: build features from the mini-season, alter the result and
stats of one mid-season match, and build again. The features of that match and
of everything earlier must stay identical; later matches of both participants
must change (a control that the test measures anything at all).

Scope: the pure core of the builder (_build_long -> _add_summary_stubs ->
_add_stats). Categorical columns (cat2m/cat3q) come from population thresholds
computed over the whole dataset — a documented project decision, not a per-match
leak — and require files on disk, so they stay outside this test.
"""
import pandas as pd

from features.match_team_stats import _add_stats, _add_summary_stubs, _build_long


def _features(raw):
    long = _add_stats(_add_summary_stubs(_build_long(raw, "testliga", "2425")))
    return long.sort_values(["team", "date"]).reset_index(drop=True)


def _feature_cols(raw, built):
    """Columns computed by _add_stats: everything that was not in the input
    (helper columns with a "_" prefix excluded)."""
    inputs = set(_add_summary_stubs(_build_long(raw, "testliga", "2425")).columns)
    return [c for c in built.columns if c not in inputs and not c.startswith("_")]


def _perturb(raw, idx):
    raw = raw.copy()
    raw.loc[idx, ["FTHG", "FTAG", "FTR"]] = [7, 0, "H"]
    raw.loc[idx, ["HS", "HST", "HC", "HY", "HR"]] = [30, 15, 12, 4, 1]
    raw.loc[idx, ["AvgH", "AvgD", "AvgA"]] = [1.2, 6.0, 9.0]
    return raw


def test_features_see_only_the_past(mini_season_raw):
    mid = len(mini_season_raw) // 2
    match_date = pd.to_datetime(mini_season_raw.loc[mid, "Date"], dayfirst=True)
    home, away = mini_season_raw.loc[mid, ["HomeTeam", "AwayTeam"]]

    base = _features(mini_season_raw)
    pert = _features(_perturb(mini_season_raw, mid))
    cols = _feature_cols(mini_season_raw, base)

    # Both builds have identical (team, date) rows — the masks line up.
    assert base[["team", "date"]].equals(pert[["team", "date"]])

    # 1. The perturbed match and everything earlier: features identical.
    #    A match's features may depend neither on its own result nor on the future.
    upto = base["date"] <= match_date
    pd.testing.assert_frame_equal(base.loc[upto, cols], pert.loc[upto, cols])

    # 2. Teams not involved in the match: untouched to the end of the season.
    others = ~base["team"].isin([home, away])
    pd.testing.assert_frame_equal(base.loc[others, cols], pert.loc[others, cols])

    # 3. Control: later matches of both participants must differ — otherwise the
    #    test would compare two identical frames and pass emptily.
    real = ~base["_is_summary"].astype(bool)
    later_involved = real & (base["date"] > match_date) & base["team"].isin([home, away])
    assert later_involved.any()
    assert not base.loc[later_involved, cols].equals(pert.loc[later_involved, cols])
