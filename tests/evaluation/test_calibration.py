"""Unit tests for calibration_overlay: equal-count bucketing with a second predictor on the bins.

A frame of 30 rows with a strictly increasing `by` splits into predictable equal-count bins, so the
per-bin means and the bin count are checkable by hand.
"""
import numpy as np
import pandas as pd
import pytest

from evaluation.calibration import calibration_overlay


def frame(n=30):
    by = np.arange(n) / (n - 1)          # 0 .. 1, strictly increasing
    return pd.DataFrame({
        "b": by,
        "o": by * 0.5,                    # second predictor: half of `by`
        "y": [1] * 10 + [0] * 10 + [1] * 10,   # bin0 all win, bin1 all loss, bin2 all win
    })


def test_columns_and_bin_count():
    curve = calibration_overlay(frame(), "b", "o", min_per_bucket=10)   # 30 // 10 -> 3 bins
    assert list(curve.columns) == ["b", "observed", "o"]
    assert len(curve) == 3


def test_observed_is_outcome_mean_per_bin():
    curve = calibration_overlay(frame(), "b", "o", min_per_bucket=10)
    assert list(curve["observed"]) == pytest.approx([1.0, 0.0, 1.0])


def test_other_tracks_its_own_mean_on_the_shared_bins():
    curve = calibration_overlay(frame(), "b", "o", min_per_bucket=10)
    assert list(curve["o"]) == pytest.approx(list(curve["b"] * 0.5))
    assert list(curve["b"]) == pytest.approx(sorted(curve["b"]))    # bins ordered low to high


def test_bin_count_is_clamped():
    assert len(calibration_overlay(frame(), "b", "o", min_per_bucket=1)) == 7     # clamped up to 7
    assert len(calibration_overlay(frame(), "b", "o", min_per_bucket=100)) == 3   # clamped down to 3
