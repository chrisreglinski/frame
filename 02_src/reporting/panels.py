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
    """Profit vs bet threshold, with left/peak/right marked. Shows the model separating good bets
    from bad (the peak) and how the edge collapses when selection gets too liberal (negative buffer).
    Returns (fig, points) where points = {'left', 'peak', 'right'} buffers.
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
    ax.set_xlabel("bufor  (model_p − implied)")
    ax.set_ylabel("profit  (implied staking)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, points


_SHOWN = ["n_matches", "breakeven", "hit_rate", "roi", "profit"]


def _reliability_ax(ax, data, title):
    model = reliability_curve(data["model_p"], data["y"])
    market = reliability_curve(data["implied"], data["y"])

    points = pd.concat([model[["mean_prob", "mean_outcome"]], market[["mean_prob", "mean_outcome"]]])
    lo, hi = points.min().min(), points.max().max()
    ax.plot([lo, hi], [lo, hi], ls=":", color="#999", lw=1)
    ax.plot(model["mean_prob"], model["mean_outcome"], "o-", color=_MODEL_C, lw=1.5, ms=4, label="model")
    ax.plot(market["mean_prob"], market["mean_outcome"], "s-", color=_MARKET_C, lw=1.5, ms=4, label="market")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed frequency")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def panel_reliability(preds, buffer):
    """Two reliability diagrams (model vs market, each bucketed by its own probability). Left: all
    matches — the model's overall calibration. Right: only the bet matches (model_p > implied +
    buffer) — calibration where it actually stakes, which is what matters for Kelly sizing.
    """
    bets = preds[preds["model_p"] > preds["implied"] + buffer]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    _reliability_ax(axes[0], preds, "all matches")
    _reliability_ax(axes[1], bets, "bet matches")
    fig.tight_layout()
    return fig


def group_stats(preds, group, buffer):
    """Per-group stats at one bet threshold, straight from portfolio_stats: n_matches, breakeven,
    hit_rate, roi. `group` is a column name like 'season' or 'league'.
    """
    bets = preds[preds["model_p"] > preds["implied"] + buffer]
    return bets.groupby(group).apply(portfolio_stats, include_groups=False)[_SHOWN]
