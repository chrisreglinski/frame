"""Bankroll simulation under Kelly staking.

kelly_bankroll compounds a starting bankroll bet by bet, in the order it is handed the bets. The
probability that sizes each stake is not the model's: it is the market's implied probability lifted
by the portfolio's realized ROI (p = implied * (1 + roi)), a single roi applied to every bet. This
keeps the stake tied to the price and uses the model only through the aggregate edge it earned.
"""
import numpy as np
import pandas as pd


def kelly_bankroll(bets, roi, start=100.0, kelly_fraction=1.0, max_fraction=1.0,
                   implied="implied", outcome="y"):
    """Compounding Kelly bankroll over `bets`, staked in the given row order.

    Each bet pays decimal odds o = 1 / implied (staking `implied` returns 1). The probability driving
    the stake is p = implied * (1 + roi). The Kelly fraction is f = (p * o - 1) / (o - 1), clipped to
    [0, max_fraction] and scaled by kelly_fraction; the stake is f * bankroll. The bankroll compounds:
    a win adds stake * (o - 1), a loss subtracts the stake.

    `bets` must already be in the order to stake (chronological, say). Returns a DataFrame with a
    leading start row then one row per bet: step, date, season, the stake, the fraction, the bankroll
    after the bet, and the running peak and drawdown (bankroll / peak - 1).
    """
    imp = bets[implied].to_numpy(dtype=float)
    y = bets[outcome].to_numpy(dtype=float)
    dates = bets["date"].to_numpy()
    seasons = bets["season"].to_numpy()

    odds = 1.0 / imp
    p = imp * (1.0 + roi)
    f = np.clip((p * odds - 1.0) / (odds - 1.0), 0.0, max_fraction) * kelly_fraction

    n = len(imp)
    bankroll = float(start)
    rows = [{"step": 0, "date": dates[0] if n else None, "season": seasons[0] if n else None,
             "stake": 0.0, "fraction": 0.0, "bankroll": bankroll}]
    for i in range(n):
        stake = f[i] * bankroll
        bankroll += stake * (odds[i] - 1.0) if y[i] > 0 else -stake
        rows.append({"step": i + 1, "date": dates[i], "season": seasons[i],
                     "stake": stake, "fraction": f[i], "bankroll": bankroll})

    curve = pd.DataFrame(rows)
    curve["peak"] = curve["bankroll"].cummax()
    curve["drawdown"] = curve["bankroll"] / curve["peak"] - 1.0
    return curve
