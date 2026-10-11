"""Unit tests for portfolio_stats: proportional-staking summary and the p-value null.

Small hand-built frames with known answers. The staking numbers (staked, profit, roi) always
read the raw implied price, while the p-value null reads `prob` when given (the devigged
probability), so the two concerns are checked separately.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from evaluation.stats import portfolio_stats


@pytest.fixture
def frame():
    """Four bets: implied 0.25/0.25/0.5/0.5, wins on the first and third."""
    return pd.DataFrame({"implied": [0.25, 0.25, 0.5, 0.5], "y": [1, 0, 1, 0]})


def test_staking_summary(frame):
    s = portfolio_stats(frame)
    assert s["n_matches"] == 4
    assert s["staked"] == pytest.approx(1.5)          # sum of implied
    assert s["wins"] == pytest.approx(2.0)
    assert s["profit"] == pytest.approx(0.5)          # wins - staked
    assert s["roi"] == pytest.approx(0.5 / 1.5)       # profit / staked
    assert s["hit_rate"] == pytest.approx(0.5)
    assert s["breakeven"] == pytest.approx(0.375)     # mean implied


def test_pvalue_defaults_to_implied(frame):
    """With no prob column the null uses the implied price itself."""
    s = portfolio_stats(frame)
    var = (frame["implied"] * (1 - frame["implied"])).sum()
    z = (frame["y"].sum() - frame["implied"].sum()) / np.sqrt(var)
    assert s["p_value"] == pytest.approx(float(norm.sf(z)))


def test_pvalue_uses_prob_not_implied(frame):
    """The devigged prob drives the null; staking numbers stay on the raw implied."""
    frame = frame.assign(fair=[0.2, 0.2, 0.4, 0.4])
    s = portfolio_stats(frame, prob="fair")

    var = (frame["fair"] * (1 - frame["fair"])).sum()
    z = (frame["y"].sum() - frame["fair"].sum()) / np.sqrt(var)
    assert s["p_value"] == pytest.approx(float(norm.sf(z)))

    # staking side is untouched by prob
    assert s["staked"] == pytest.approx(1.5)
    assert s["profit"] == pytest.approx(0.5)
    # a lower null probability makes the same run look less likely
    assert s["p_value"] < portfolio_stats(frame)["p_value"]


def test_prob_equal_to_implied_matches_default(frame):
    frame = frame.assign(same=frame["implied"])
    assert portfolio_stats(frame, prob="same")["p_value"] == pytest.approx(
        portfolio_stats(frame)["p_value"])


def test_zero_variance_gives_nan_pvalue():
    """A null with every probability at 1 has no variance, so the z-score is undefined."""
    frame = pd.DataFrame({"implied": [0.5, 0.5], "y": [1, 1], "certain": [1.0, 1.0]})
    s = portfolio_stats(frame, prob="certain")
    assert np.isnan(s["p_value"])
    assert s["staked"] == pytest.approx(1.0)          # staking still computed
