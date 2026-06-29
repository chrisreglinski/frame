"""
Portfolio evaluation functions for betting model assessment.

portfolio_roi       — evaluates edge existence, independent of scale
portfolio_kelly     — flat Kelly staking with optional bankroll chart
portfolio_breakdown — ROI pivot table broken down by league × season
"""
import pandas as pd
import matplotlib.pyplot as plt


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
        mask = pd.Series(True, index=implied_probs.index).values
    else:
        mask = model_probs.values > implied_probs.values + buffer
    imp = implied_probs[mask]
    won = outcomes[mask]

    capital = float(imp.sum())
    wins = int(won.sum())
    profit = wins - capital

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
    kelly_fraction: float = 0.5,
    plot: bool = False,
    dates: pd.Series = None,
    title: str = "",
) -> dict:
    """
    Flat Kelly staking: stake_i = f*_i * bankroll (bankroll never updated).

    f* = (model_prob * odds - 1) / (odds - 1), clipped to [0, max_fraction].
    If model_probs not provided, bets all rows (bet_all mode).
    If model_probs provided, selects bets where model_prob > implied_prob + buffer.
    If plot=True, also draws the cumulative bankroll chart (requires dates).
    """
    df = pd.DataFrame({
        "imp": implied_probs,
        "won": outcomes,
        "mp":  model_probs if model_probs is not None else implied_probs,
    })
    if dates is not None:
        df["date"] = dates.values

    if model_probs is not None:
        df = df[df["mp"] > df["imp"] + buffer].copy()

    odds = 1.0 / df["imp"]
    f_star = ((df["mp"].values * odds.values - 1) / (odds.values - 1)).clip(0, max_fraction) * kelly_fraction
    stakes = f_star * bankroll

    capital = float(stakes.sum())
    revenue = float((df["won"].values * stakes * odds.values).sum())
    profit = revenue - capital

    result = {
        "n_bets":      len(df),
        "capital":     round(capital, 4),
        "wins":        int(df["won"].sum()),
        "profit":      round(profit, 4),
        "roi":         round(profit / capital, 4) if capital > 0 else None,
        "bankroll_roi": round(profit / bankroll, 4),
    }

    if plot:
        if dates is None:
            raise ValueError("dates required for plot=True")
        df_plot = df.sort_values("date").reset_index(drop=True)
        odds_plot = 1.0 / df_plot["imp"]
        f_plot = ((df_plot["mp"].values * odds_plot.values - 1) / (odds_plot.values - 1)).clip(0, max_fraction) * kelly_fraction
        stakes_plot = f_plot * bankroll
        profit_plot = df_plot["won"].values * stakes_plot * odds_plot.values - stakes_plot
        cumulative = bankroll + profit_plot.cumsum()

        plt.figure(figsize=(12, 4))
        plt.plot(df_plot["date"].values, cumulative, linewidth=1)
        plt.axhline(bankroll, color="gray", linewidth=0.8, linestyle="--")
        plt.xlabel("date")
        plt.ylabel("bankroll")
        if title:
            plt.title(title)
        plt.tight_layout()
        plt.show()

    return result


def portfolio_breakdown(
    implied_probs: pd.Series,
    outcomes: pd.Series,
    leagues: pd.Series,
    seasons: pd.Series,
    model_probs: pd.Series = None,
    buffer: float = 0.0,
) -> pd.DataFrame:
    """
    ROI pivot table broken down by league (rows) × season (cols).
    Accepts same implied_probs/outcomes/model_probs as portfolio_roi.
    """
    df = pd.DataFrame({
        "imp": implied_probs,
        "won": outcomes,
        "league": leagues,
        "season": seasons,
    })
    if model_probs is not None:
        df = df[model_probs.values > implied_probs.values + buffer]

    def _roi_scalar(sub):
        cap = float(sub["imp"].sum())
        return round((float(sub["won"].sum()) - cap) / cap, 4) if cap > 0 else None

    def _roi_series(by):
        agg = df.groupby(by)[["imp", "won"]].sum()
        cap = agg["imp"].replace(0, float("nan"))
        return ((agg["won"] - agg["imp"]) / cap).round(4)

    pivot = _roi_series(["league", "season"]).unstack("season")
    pivot["ALL"] = _roi_series("league")
    totals = _roi_series("season").rename("ALL")
    totals["ALL"] = _roi_scalar(df)
    pivot = pd.concat([pivot, totals.to_frame().T])

    return pivot
