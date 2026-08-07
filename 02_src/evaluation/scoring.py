"""Scoring a predicted quantity against the realized outcome, over a whole portfolio.

`score_diff` is the first, cheapest lens: it scores a predicted home-away margin (`h - a`,
in [-1, 1]) against the realized signed result `t_flg_diff` in {1, 0, -1}. Since
E[t_flg_diff] = P(home) - P(away) = h - a, the margin *is* the expected signed result, so
mean squared error is a proper score for it — minimized when the margin equals the true
h - a. It measures only the home-away axis (strength + home advantage), ignores the draw, and
needs no de-vigging (the margin cancels most of the vig).

Compare two margins by scoring each against the same outcomes: a model margin vs the market's
`mrkt_home_away_impl_diff`. Lower is better.
"""
import numpy as np
import pandas as pd


def score_diff(margin, outcome_diff):
    """Mean squared error of a predicted home-away margin against `t_flg_diff`.

    `margin` — predicted h - a per match (e.g. rr_impl_diff, mrkt_home_away_impl_diff).
    `outcome_diff` — realized `t_flg_diff` in {1, 0, -1}.
    Rows where either is NaN are dropped. Lower MSE = margin closer to the true h - a.
    """
    df = pd.DataFrame({"margin": np.asarray(margin, dtype=float),
                       "outcome": np.asarray(outcome_diff, dtype=float)}).dropna()
    return float(((df["margin"] - df["outcome"]) ** 2).mean())


def compare_diff(model_margin, market_margin, outcome_diff):
    """Score a model margin against the market margin on the same matches.

    Both margins are scored against `t_flg_diff` by `score_diff`, on the rows where model,
    market and outcome are all present (so the two MSEs are comparable). Returns a Series with:

    - `mse_model`  — MSE of the model margin
    - `mse_market` — MSE of the market margin
    - `skill`      — mse_market - mse_model; positive = the model is closer to the truth
    - `n`          — matches scored

    Maps cleanly through groupby for a per-league / per-season breakdown.
    """
    df = pd.DataFrame({"model": np.asarray(model_margin, dtype=float),
                       "market": np.asarray(market_margin, dtype=float),
                       "outcome": np.asarray(outcome_diff, dtype=float)}).dropna()
    mse_model = score_diff(df["model"], df["outcome"])
    mse_market = score_diff(df["market"], df["outcome"])
    return pd.Series({"mse_model": mse_model, "mse_market": mse_market,
                      "skill": mse_market - mse_model, "n": len(df)})
