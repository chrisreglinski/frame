"""Tearsheet panels — each takes a predictions frame and returns a figure and/or a table.

Panels are presentation only: they call the evaluation layer to compute, then draw. Nothing here
fits models or decides staking.
"""
import matplotlib.pyplot as plt
import pandas as pd

from evaluation.stats import buffer_curve, pick_buffers, portfolio_stats
from evaluation.calibration import reliability_curve

_MODEL_C = "#c0392b"
_MARKET_C = "#2980b9"

# Match the chart's font to the HTML (Consolas monospace) so the whole report reads as one.
plt.rcParams["font.family"] = "monospace"
plt.rcParams["font.monospace"] = ["Consolas", "DejaVu Sans Mono"]

_MARK = {"left": "#27ae60", "middle": "#8e44ad", "right": "#7f8c8d"}


def panel_buffer_profit(preds):
    """Profit vs bet threshold (buffer = model_p − implied): raw and smoothed profit curve, with the
    reference buffers from pick_buffers marked. Returns (fig, points).
    """
    curve = buffer_curve(preds)
    points = pick_buffers(curve)

    fig, ax = plt.subplots(figsize=(8.5, 3.0))
    ax.plot(curve.index, curve["profit"], color="#e6b0aa", lw=1)
    ax.plot(curve.index, curve["profit_smooth"], color="#c0392b", lw=2)
    ax.axhline(0, color="#999", lw=0.8)
    for name, buffer in points.items():
        ax.axvline(buffer, ls="--", lw=1, color=_MARK[name])
    for i, (name, buffer) in enumerate(reversed(points.items())):
        ax.text(0.985, 0.04 + i * 0.07, f"{name}  {buffer:.3f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color=_MARK[name])
    ax.set_xlabel("buffer  (model_p − implied)")
    ax.set_ylabel("profit  (implied staking)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, points


_SHOWN = ["n_matches", "breakeven", "hit_rate", "roi", "profit"]


def panel_edge_calibration(data, min_per_bucket=300):
    """Bucket matches by edge (model_p − implied); plot observed frequency, model probability and
    market (implied) probability against the mean edge per bucket. Bucketing by one quantity keeps
    all three lines on the same matches. (model = market + edge by construction.)
    """
    d = data.assign(edge=data["model_p"] - data["implied"])
    n_bins = max(3, min(10, len(d) // min_per_bucket))
    grouped = d.assign(_bin=pd.qcut(d["edge"], n_bins, duplicates="drop")).groupby("_bin", observed=True)
    x = grouped["edge"].mean()

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.axvline(0, color="#ccc", lw=1)
    ax.plot(x, grouped["y"].mean(), "o-", color="#333333", lw=1.6, ms=4, label="observed")
    ax.plot(x, grouped["model_p"].mean(), "s-", color=_MODEL_C, lw=1.6, ms=4, label="model")
    ax.plot(x, grouped["implied"].mean(), "^-", color=_MARKET_C, lw=1.6, ms=4, label="market")
    ax.set_title(f"probability vs edge   n={len(d)}, {n_bins} buckets", fontsize=9)
    ax.set_xlabel("edge  (model_p − implied)")
    ax.set_ylabel("probability / frequency")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def _reliability_ax(ax, data, title, min_per_bucket):
    model = reliability_curve(data["model_p"], data["y"], min_per_bucket)
    market = reliability_curve(data["implied"], data["y"], min_per_bucket)

    points = pd.concat([model[["mean_prob", "mean_outcome"]], market[["mean_prob", "mean_outcome"]]])
    lo, hi = points.min().min(), points.max().max()
    ax.plot([lo, hi], [lo, hi], ls=":", color="#999", lw=1)
    ax.plot(model["mean_prob"], model["mean_outcome"], "o-", color=_MODEL_C, lw=1.5, ms=4, label="model")
    ax.plot(market["mean_prob"], market["mean_outcome"], "s-", color=_MARKET_C, lw=1.5, ms=4, label="market")
    ax.set_title(f"{title}   n={len(data)}, {len(model)} buckets (~{len(data) // len(model)}/bucket)", fontsize=9)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def panel_reliability(preds, buffer, min_per_bucket=100):
    """Two reliability diagrams (model and market, each bucketed by its own probability): left over
    all matches, right over the bet matches (model_p > implied + buffer). Bucket count is chosen so
    each holds at least `min_per_bucket` matches.
    """
    bets = preds[preds["model_p"] > preds["implied"] + buffer]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    _reliability_ax(axes[0], preds, "all matches", min_per_bucket)
    _reliability_ax(axes[1], bets, "bet matches", min_per_bucket)
    fig.tight_layout()
    return fig


def _calibration_ax(ax, data, by, other, x_label, other_label, other_color, min_per_bucket, title):
    """Bucket `data` by column `by`; per bucket plot observed frequency and the mean of the `other`
    predictor against the mean of `by`. Bucketing by one predictor keeps every line on the same
    matches, so the points correspond. The diagonal is where `by` itself is perfectly calibrated.
    """
    n_bins = max(3, min(7, len(data) // min_per_bucket))
    binned = data.assign(_bin=pd.qcut(data[by], n_bins, duplicates="drop"))
    grouped = binned.groupby("_bin", observed=True)
    x = grouped[by].mean()
    observed = grouped["y"].mean()
    other_mean = grouped[other].mean()

    lo = min(x.min(), observed.min(), other_mean.min())
    hi = max(x.max(), observed.max(), other_mean.max())
    ax.plot([lo, hi], [lo, hi], ls=":", color="#999", lw=1)
    ax.plot(x, observed, "o-", color="#333333", lw=1.5, ms=4, label="observed")
    ax.plot(x, other_mean, "s-", color=other_color, lw=1.5, ms=4, label=other_label)
    ax.set_title(f"{title}   n={len(data)}, {n_bins} buckets", fontsize=9)
    ax.set_xlabel(x_label)
    ax.set_ylabel("frequency / probability")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def panel_calibration(all_matches, bet_matches, min_per_bucket=100):
    """A 2x2 calibration grid where each panel buckets by one predictor, so all lines share matches.
    Row 1 buckets by the market price (x = implied), overlaying observed frequency and the model's
    probability. Row 2 buckets by the model (x = model_p), overlaying observed and the market price.
    Columns are all matches (left) and bet matches (right).
    """
    fig, axes = plt.subplots(2, 2, figsize=(9, 8.4))
    _calibration_ax(axes[0, 0], all_matches, "implied", "model_p", "market probability", "model", _MODEL_C, min_per_bucket, "all matches")
    _calibration_ax(axes[0, 1], bet_matches, "implied", "model_p", "market probability", "model", _MODEL_C, min_per_bucket, "bet matches")
    _calibration_ax(axes[1, 0], all_matches, "model_p", "implied", "model probability", "market", _MARKET_C, min_per_bucket, "all matches")
    _calibration_ax(axes[1, 1], bet_matches, "model_p", "implied", "model probability", "market", _MARKET_C, min_per_bucket, "bet matches")
    fig.tight_layout()
    return fig


def panel_reliability_grid(rows, buffer, min_per_bucket=100):
    """A grid of reliability diagrams: one row per (label, predictions) in `rows`, each row showing
    all matches (left) and bet matches (right) — for comparing calibration across groups (e.g. seasons).
    """
    fig, axes = plt.subplots(len(rows), 2, figsize=(9, 4.2 * len(rows)))
    for (label, data), (ax_all, ax_bet) in zip(rows, axes):
        bets = data[data["model_p"] > data["implied"] + buffer]
        _reliability_ax(ax_all, data, f"{label} — all matches", min_per_bucket)
        _reliability_ax(ax_bet, bets, f"{label} — bet matches", min_per_bucket)
    fig.tight_layout()
    return fig


def group_stats(preds, group, buffer):
    """Per-group stats at one bet threshold, straight from portfolio_stats: n_matches, breakeven,
    hit_rate, roi. `group` is a column name like 'season' or 'league'.
    """
    bets = preds[preds["model_p"] > preds["implied"] + buffer]
    return bets.groupby(group).apply(portfolio_stats, include_groups=False)[_SHOWN]
