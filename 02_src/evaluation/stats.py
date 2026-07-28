"""Summary statistics for a portfolio of matches under proportional (implied) staking.

portfolio_stats summarizes ANY subset of matches it is handed — the bets a model selected, a
whole season, an entire league. Selection and filtering happen upstream; this function only
scores what it receives, so the same call gives the model's edge on its bets or the "bet all"
baseline on a full set.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm


# Buffer grid swept by the profit-vs-buffer analysis (bet threshold = implied + buffer).
# Starts below zero so the curve shows the liberal side too — betting even where the model barely
# disagrees with, or sits under, the price — and how profit collapses there.
DEFAULT_BUFFER_GRID = np.round(np.arange(-0.04, 0.1201, 0.001), 4)


def portfolio_stats(matches, implied="implied", outcome="y"):
    """Proportional-staking stats for a subset of matches.

    Staking is implied-proportional: stake = implied, a win returns 1, so profit = wins - staked
    and ROI = profit / staked (scale-independent). The p-value is one-sided (upper tail): under the
    null that the market is right (true P = implied for each match), it is the probability of the
    outcome landing at least as often as observed — a normal approximation to the Poisson-binomial
    (mean Sum(p), variance Sum(p(1-p))). Small p = the subset outperformed its price.
    """
    imp = matches[implied].to_numpy(dtype=float)
    y = matches[outcome].to_numpy(dtype=float)

    staked = imp.sum()
    wins = y.sum()
    profit = wins - staked

    variance = (imp * (1 - imp)).sum()
    z = (wins - staked) / np.sqrt(variance) if variance > 0 else np.nan
    p_value = float(norm.sf(z)) if variance > 0 else np.nan

    return pd.Series({
        "n_matches": len(matches),
        "staked":    staked,
        "wins":      wins,
        "profit":    profit,
        "roi":       profit / staked if staked > 0 else np.nan,
        "hit_rate":  y.mean() if len(y) else np.nan,
        "breakeven": imp.mean() if len(imp) else np.nan,
        "p_value":   p_value,
    })


def buffer_curve(preds, grid=DEFAULT_BUFFER_GRID):
    """Profit and ROI at every candidate bet threshold (bet where model_p > implied + buffer).

    Sweeping the buffer never refits the model — it only re-thresholds the same predictions, so
    the whole curve is cheap. Returns a DataFrame indexed by buffer with profit, roi, n_matches.
    """
    rows = []
    for buffer in grid:
        bets = preds[preds["model_p"] > preds["implied"] + buffer]
        stats = portfolio_stats(bets)
        rows.append({"buffer": buffer, "profit": stats["profit"],
                     "roi": stats["roi"], "n_matches": stats["n_matches"]})
    return pd.DataFrame(rows).set_index("buffer")


def pick_buffers(curve, frac=0.5):
    """Three buffers off the profit curve: the peak, and the edges of the band where profit stays
    within `frac` of its maximum. Deterministic — no eyeballing. The report bets at `peak` and
    uses `left`/`right` to show the result holds across a plateau, not just at the cherry-picked max.
    """
    profit = curve["profit"]
    band = profit.index[profit >= frac * profit.max()]
    return {"left": float(band.min()), "peak": float(profit.idxmax()), "right": float(band.max())}
