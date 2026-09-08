"""Calibration (reliability) of a probability predictor.

A predictor is well calibrated when, among the matches it prices at probability p, the event
actually happens about p of the time. reliability_curve bins a predictor by its own probability
and returns, per bin, the mean predicted probability against the observed frequency — the points
of a reliability diagram. Model and market are each bucketed by their own probability, so they
are two independent curves.
"""
import pandas as pd


def reliability_curve(prob, outcome, min_per_bucket=100):
    """Bin `prob` into equal-count (quantile) buckets and return per-bin mean predicted probability,
    observed frequency, and count. The number of buckets aims for at least `min_per_bucket`
    observations each, clamped to 3–7 — equal-count bins keep every point about as statistically
    reliable as the next, and the clamp keeps the diagram readable and not over-split.
    """
    df = pd.DataFrame({"prob": pd.Series(prob).to_numpy(), "outcome": pd.Series(outcome).to_numpy()})
    n_bins = max(3, min(7, len(df) // min_per_bucket))
    df["bin"] = pd.qcut(df["prob"], n_bins, duplicates="drop")

    grouped = df.groupby("bin", observed=True)
    return pd.DataFrame({
        "mean_prob": grouped["prob"].mean(),
        "mean_outcome": grouped["outcome"].mean(),
        "count": grouped["prob"].size(),
    }).reset_index(drop=True)
