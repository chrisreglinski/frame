"""Unit tests for the feature-space registry: column resolution, windowing, de-duplication."""
import pytest

from models import spaces

# Sybilla's feature set, which must equal goals_foragst + points + anchor.
SYBILLA = {
    "hmt_season_goals_for_avg", "hmt_season_goals_agst_avg",
    "awt_season_goals_for_avg", "awt_season_goals_agst_avg",
    "hmt_season_points_avg", "awt_season_points_avg",
    "hmt_season_mp_impl_points_avg", "awt_season_mp_impl_points_avg",
    "mrkt_draw_impl",
}


def test_goals_foragst_season_columns():
    assert spaces.columns("goals_foragst") == [
        "hmt_season_goals_for_avg", "hmt_season_goals_agst_avg",
        "awt_season_goals_for_avg", "awt_season_goals_agst_avg",
    ]


def test_rolling_suffix_swaps_the_window():
    assert spaces.columns("xg_r6") == [
        "hmt_rolling6_xg_for_avg", "hmt_rolling6_xg_agst_avg",
        "awt_rolling6_xg_for_avg", "awt_rolling6_xg_agst_avg",
    ]
    assert spaces.columns("points_r8") == [
        "hmt_rolling8_points_avg", "hmt_rolling8_mp_impl_points_avg",
        "awt_rolling8_points_avg", "awt_rolling8_mp_impl_points_avg",
    ]


def test_fixed_spaces():
    assert spaces.columns("anchor") == ["mrkt_draw_impl"]
    assert spaces.columns("elo") == ["hmt_elo", "awt_elo"]


def test_resolve_reproduces_sybilla():
    assert set(spaces.resolve(["goals_foragst", "points", "anchor"])) == SYBILLA


def test_resolve_dedups_and_keeps_order():
    cols = spaces.resolve(["anchor", "goals_foragst", "anchor"])
    assert cols[0] == "mrkt_draw_impl"
    assert cols.count("mrkt_draw_impl") == 1
    assert len(cols) == 5


def test_unknown_space_raises():
    with pytest.raises(KeyError):
        spaces.columns("nonsense")


def test_names_lists_windowed_variants():
    n = spaces.names()
    assert {"goals_foragst", "goals_foragst_r6", "goals_foragst_r8", "elo", "anchor"} <= set(n)
