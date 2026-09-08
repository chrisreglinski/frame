"""Summary statistics for a portfolio of matches under proportional (implied) staking.

portfolio_stats summarizes ANY subset of matches it is handed — the bets a model selected, a
whole season, an entire league. Selection and filtering happen upstream; this function only
scores what it receives, so the same call gives the model's edge on its bets or the "bet all"
baseline on a full set.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.signal import savgol_filter, find_peaks


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


def buffer_curve(preds, grid=DEFAULT_BUFFER_GRID, smooth_window=21, smooth_poly=2):
    """Profit and ROI at every candidate bet threshold (bet where model_p > implied + buffer).

    Sweeping the buffer never refits the model — it only re-thresholds the same predictions, so
    the whole curve is cheap. The raw profit is noisy (few bets remain at high buffers), so a
    Savitzky-Golay smoothing is added as `profit_smooth`; buffer choice reads that, not the raw
    spikes. Returns a DataFrame indexed by buffer with profit, profit_smooth, roi, n_matches.
    """
    rows = []
    for buffer in grid:
        bets = preds[preds["model_p"] > preds["implied"] + buffer]
        stats = portfolio_stats(bets)
        rows.append({"buffer": buffer, "profit": stats["profit"],
                     "roi": stats["roi"], "n_matches": stats["n_matches"]})
    curve = pd.DataFrame(rows).set_index("buffer")

    window = min(smooth_window, len(curve) - (1 - len(curve) % 2))  # odd, <= length
    if window > smooth_poly:
        curve["profit_smooth"] = savgol_filter(curve["profit"].to_numpy(), window, smooth_poly)
    else:
        curve["profit_smooth"] = curve["profit"]
    return curve


def pick_buffers(curve):
    """Three reference buffers from the local maxima of the smoothed profit curve:

    - `left`   — the first (lowest-buffer) local maximum.
    - `right`  — the last (highest-buffer) local maximum.
    - `middle` — the midpoint between `left` and `right`.

    Peaks are detected on `profit_smooth` with a prominence floor so minor wiggles are ignored.
    """
    profit = curve["profit_smooth"].to_numpy()
    buffers = curve.index.to_numpy()

    prominence = 0.03 * (profit.max() - profit.min())
    peaks, _ = find_peaks(profit, prominence=prominence)
    if len(peaks) == 0:
        peaks = [int(profit.argmax())]

    left, right = float(buffers[peaks[0]]), float(buffers[peaks[-1]])
    return {"left": left, "middle": (left + right) / 2, "right": right}
