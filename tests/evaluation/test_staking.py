"""Unit tests for kelly_bankroll: compounding trajectory, sizing, peak and drawdown.

A two-bet frame with implied 0.5 (decimal odds 2) keeps the arithmetic exact: with roi 0.2 the
stake fraction is f = (0.6 * 2 - 1) / (2 - 1) = 0.2, so each stake and bankroll step is checkable
by hand.
"""
import pandas as pd
import pytest

from evaluation.staking import kelly_bankroll


@pytest.fixture
def bets():
    """Two bets at even-money implied (odds 2): a win then a loss."""
    return pd.DataFrame({
        "implied": [0.5, 0.5],
        "y": [1, 0],
        "date": pd.to_datetime(["2024-08-10", "2024-08-17"]),
        "season": [2024, 2024],
    })


def test_leading_start_row(bets):
    curve = kelly_bankroll(bets, roi=0.2, start=100.0)
    assert len(curve) == len(bets) + 1                # start row then one per bet
    first = curve.iloc[0]
    assert first["step"] == 0
    assert first["bankroll"] == pytest.approx(100.0)
    assert first["stake"] == pytest.approx(0.0)
    assert first["fraction"] == pytest.approx(0.0)


def test_compounding_win_then_loss(bets):
    """f = 0.2. Win: 100 -> 120 (stake 20 at odds 2). Loss: 120 -> 96 (stake 24)."""
    curve = kelly_bankroll(bets, roi=0.2, start=100.0)
    win, loss = curve.iloc[1], curve.iloc[2]
    assert win["fraction"] == pytest.approx(0.2)
    assert win["stake"] == pytest.approx(20.0)
    assert win["bankroll"] == pytest.approx(120.0)
    assert loss["stake"] == pytest.approx(24.0)        # 0.2 of the grown bankroll
    assert loss["bankroll"] == pytest.approx(96.0)


def test_peak_and_drawdown(bets):
    curve = kelly_bankroll(bets, roi=0.2, start=100.0)
    assert list(curve["peak"]) == pytest.approx([100.0, 120.0, 120.0])
    assert list(curve["drawdown"]) == pytest.approx([0.0, 0.0, -0.2])   # 96 / 120 - 1


def test_kelly_fraction_scales_the_stake(bets):
    """Half-Kelly halves the fraction, so the first stake is 10, not 20."""
    curve = kelly_bankroll(bets, roi=0.2, start=100.0, kelly_fraction=0.5)
    win = curve.iloc[1]
    assert win["fraction"] == pytest.approx(0.1)
    assert win["stake"] == pytest.approx(10.0)
    assert win["bankroll"] == pytest.approx(110.0)


def test_negative_edge_clips_fraction_to_zero(bets):
    """A negative implied-lift makes Kelly want a short, which clips to no bet."""
    curve = kelly_bankroll(bets, roi=-0.5, start=100.0)
    assert (curve["fraction"] == 0.0).all()
    assert (curve["bankroll"] == 100.0).all()          # bankroll never moves


def test_max_fraction_caps_the_bet(bets):
    """A rich edge is capped by max_fraction before kelly_fraction scales it."""
    curve = kelly_bankroll(bets, roi=5.0, start=100.0, max_fraction=0.3)
    assert curve.iloc[1]["fraction"] == pytest.approx(0.3)
