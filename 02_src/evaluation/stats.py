"""Summary statistics for a portfolio of matches under proportional (implied) staking.

portfolio_stats summarizes ANY subset of matches it is handed — the bets a model selected, a
whole season, an entire league. Selection and filtering happen upstream; this function only
scores what it receives, so the same call gives the model's edge on its bets or the "bet all"
baseline on a full set.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm


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
