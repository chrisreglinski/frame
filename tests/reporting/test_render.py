"""Unit tests for render.wrap_scale, the tight (non-zero-anchored) axis scale.

wrap_scale wraps [dmin, dmax] in exactly n ticks: the first tick is the largest multiple of
`base` at or below dmin, the step is the smallest multiple of `base` whose n-1 intervals reach
dmax. Every tick is a multiple of `base`, and the axis is tight (it does not over-reach).
"""
import pytest

from reporting.render import wrap_scale


@pytest.mark.parametrize("dmin, dmax, exp_vals", [
    (0.10, 0.30, [0.10, 0.20, 0.30, 0.40]),   # step grows to 0.10 to reach dmax
    (0.12, 0.22, [0.10, 0.15, 0.20, 0.25]),   # base step 0.05 already reaches dmax
    (-0.03, 0.05, [-0.05, 0.00, 0.05, 0.10]),  # first tick steps below a negative dmin
])
def test_tick_values(dmin, dmax, exp_vals):
    domain, ticks = wrap_scale(dmin, dmax, n=4, base=0.05)
    assert [t["v"] for t in ticks] == pytest.approx(exp_vals)
    assert domain == pytest.approx([exp_vals[0], exp_vals[-1]])


def test_wraps_tightly():
    """First tick sits at or below dmin, last at or above dmax, with no extra slack tick."""
    domain, ticks = wrap_scale(0.12, 0.22, n=4, base=0.05)
    vals = [t["v"] for t in ticks]
    assert vals[0] <= 0.12 and vals[-1] >= 0.22
    assert vals[0] + 0.05 > 0.12          # no room for a tick below dmin
    assert vals[-1] - 0.05 < 0.22         # no room for a tick above dmax


def test_ticks_are_evenly_spaced_multiples_of_base():
    _, ticks = wrap_scale(0.07, 0.41, n=4, base=0.05)
    vals = [t["v"] for t in ticks]
    steps = [vals[i + 1] - vals[i] for i in range(len(vals) - 1)]
    assert steps == pytest.approx([steps[0]] * len(steps))       # constant step
    assert all(round(v / 0.05) == pytest.approx(v / 0.05, abs=1e-9) for v in vals)


def test_tick_count_is_configurable():
    _, ticks = wrap_scale(0.0, 0.5, n=6, base=0.05)
    assert len(ticks) == 6


def test_labels_follow_format():
    _, ticks = wrap_scale(0.10, 0.30, n=4, base=0.05, fmt="pct0")
    assert [t["label"] for t in ticks] == ["10%", "20%", "30%", "40%"]
