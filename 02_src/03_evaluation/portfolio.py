"""
Portfolio evaluation functions for betting model assessment.

portfolio_roi   — evaluates edge existence, independent of scale
portfolio_kelly — flat Kelly staking (bankroll held constant, no compounding)
"""
import pandas as pd


def portfolio_roi(
    implied_probs: pd.Series,
    outcomes: pd.Series,
    model_probs: pd.Series = None,
    buffer: float = 0.0,
) -> dict:
    """
    Implied-probability staking: stake_i = implied_prob_i, win return = 1.0.

    If model_probs not provided, bets all rows (bet_all mode).
    If model_probs provided, selects bets where model_prob > implied_prob + buffer.
    ROI is scale-independent — measures whether the edge exists.
    """
    if model_probs is None:
        # select bets with positive edge (model sees higher prob than market)
        mask = pd.Series(True, index=implied_probs.index).values
    else:
        mask = model_probs.values > implied_probs.values + buffer
    imp = implied_probs[mask]
    won = outcomes[mask]

    capital = float(imp.sum())   # total staked = sum of implied probs
    wins = int(won.sum())
    profit = wins - capital      # each win returns exactly 1.0

    return {
        "n_bets": int(mask.sum()),
        "capital": round(capital, 4),
        "wins": wins,
        "profit": round(profit, 4),
        "roi": round(profit / capital, 4) if capital > 0 else None,
    }


def portfolio_kelly(
    implied_probs: pd.Series,
    outcomes: pd.Series,
    model_probs: pd.Series = None,
    bankroll: float = 100.0,
    buffer: float = 0.0,
    max_fraction: float = 0.25,
) -> dict:
    """
    Flat Kelly staking: stake_i = f*_i * bankroll (bankroll never updated).

    f* = (model_prob * odds - 1) / (odds - 1), clipped to [0, max_fraction].
    If model_probs not provided, bets all rows (bet_all mode).
    If model_probs provided, selects bets where model_prob > implied_prob + buffer.
    """
    if model_probs is None:
        # select bets with positive edge (model sees higher prob than market)
        mask = pd.Series(True, index=implied_probs.index).values
        mp_all = implied_probs
    else:
        mask = model_probs.values > implied_probs.values + buffer
        mp_all = model_probs
    imp = implied_probs[mask]
    mp = mp_all[mask]
    won = outcomes[mask]

    odds = 1.0 / imp
    # Kelly fraction: how much of bankroll to stake given model edge
    f_star = ((mp * odds - 1) / (odds - 1)).clip(0, max_fraction)
    stakes = f_star * bankroll

    capital = float(stakes.sum())
    revenue = float((won * stakes * odds).sum())  # win returns stake * odds
    profit = revenue - capital

    return {
        "n_bets": int(mask.sum()),
        "capital": round(capital, 4),
        "wins": int(won.sum()),
        "profit": round(profit, 4),
        "roi": round(profit / capital, 4) if capital > 0 else None,
        # profit relative to full bankroll; below -1.0 means bankruptcy
        "bankroll_roi": round(profit / bankroll, 4),
    }
