"""
Portfolio evaluation functions for betting model assessment.

portfolio_roi   — evaluates edge existence, independent of scale
portfolio_kelly — flat Kelly staking (bankroll held constant, no compounding)
"""
import pandas as pd


def portfolio_roi(
    model_probs: pd.Series,
    implied_probs: pd.Series,
    outcomes: pd.Series,
    buffer: float = 0.0,
    bet_all: bool = False,
) -> dict:
    """
    Implied-probability staking: stake_i = implied_prob_i, win return = 1.0.

    Selects bets where model_prob > implied_prob + buffer.
    If bet_all=True, all rows are selected regardless of model_probs.
    ROI is scale-independent — measures whether the edge exists.
    """
    # select bets with positive edge (model sees higher prob than market)
    mask = pd.Series(True, index=model_probs.index).values if bet_all \
        else model_probs.values > implied_probs.values + buffer
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
    model_probs: pd.Series,
    implied_probs: pd.Series,
    outcomes: pd.Series,
    bankroll: float = 100.0,
    buffer: float = 0.0,
    max_fraction: float = 0.25,
    bet_all: bool = False,
) -> dict:
    """
    Flat Kelly staking: stake_i = f*_i * bankroll (bankroll never updated).

    f* = (model_prob * odds - 1) / (odds - 1), clipped to [0, max_fraction].
    Selects bets where model_prob > implied_prob + buffer.
    If bet_all=True, all rows are selected regardless of model_probs.
    """
    # select bets with positive edge (model sees higher prob than market)
    mask = pd.Series(True, index=model_probs.index).values if bet_all \
        else model_probs.values > implied_probs.values + buffer
    imp = implied_probs[mask]
    mp = model_probs[mask]
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
