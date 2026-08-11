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


def score_outcome(prob, flag):
    """Binary Brier score of a single-outcome probability against its realized flag.

    `prob` — predicted probability of one outcome (e.g. mrkt_draw_impl, or a draw model).
    `flag` — realized 1/0 indicator for that outcome (e.g. t_draw_flg). Rows with either NaN
    are dropped. Lower is better; the three atomic Briers (H/D/A) sum to the HDA Brier.
    """
    df = pd.DataFrame({"prob": np.asarray(prob, dtype=float),
                       "flag": np.asarray(flag, dtype=float)}).dropna()
    return float(((df["prob"] - df["flag"]) ** 2).mean())


def compare_outcome(model_prob, market_prob, flag):
    """Score a model's single-outcome probability against the market's on the same matches.

    Both are Brier-scored against `flag` by `score_outcome`, on rows where model, market and
    flag are all present. Returns a Series with:

    - `brier_model`  — Brier of the model probability
    - `brier_market` — Brier of the market probability
    - `skill`        — brier_market - brier_model; positive = the model is closer to the truth
    - `n`            — matches scored
    """
    df = pd.DataFrame({"model": np.asarray(model_prob, dtype=float),
                       "market": np.asarray(market_prob, dtype=float),
                       "flag": np.asarray(flag, dtype=float)}).dropna()
    brier_model = score_outcome(df["model"], df["flag"])
    brier_market = score_outcome(df["market"], df["flag"])
    return pd.Series({"brier_model": brier_model, "brier_market": brier_market,
                      "skill": brier_market - brier_model, "n": len(df)})


def _onehot_hda(outcome):
    """One-hot [H, D, A] from either `t_result` ('H'/'D'/'A') or `t_flg_diff` (1/0/-1)."""
    o = np.asarray(outcome)
    cols = [o == "H", o == "D", o == "A"] if o.dtype.kind in ("U", "S", "O") \
        else [o == 1, o == 0, o == -1]
    return np.stack(cols, axis=1).astype(float)


def score_hda(hda, outcome, ranked=False):
    """Mean quadratic score of an HDA probability triple against the realized result.

    `hda` columns are taken in home, draw, away order (e.g. impl_h/impl_d/impl_a from
    hda_from_margin, or devigged mrkt_home/draw/away_impl); each row is normalized to sum 1.
    `outcome` is the realized result as `t_result` ('H'/'D'/'A') or `t_flg_diff` (1/0/-1).
    Rows with any NaN are dropped.

    Default is the non-ranked Brier score, sum_k (p_k - o_k)^2 over the raw H/D/A probs — a
    wrong class is a wrong class, as for a settled bet. `ranked=True` gives RPS instead,
    scoring cumulative probabilities so a miss to an adjacent outcome costs less. Lower is better.
    """
    p = pd.DataFrame(hda).to_numpy(dtype=float)
    o = np.asarray(outcome)
    keep = ~(np.isnan(p).any(axis=1) | pd.isna(o))
    p, o = p[keep], o[keep]
    p = p / p.sum(axis=1, keepdims=True)                       # normalize to sum 1
    obs = _onehot_hda(o)                                       # one-hot H, D, A
    if ranked:
        per = 0.5 * ((np.cumsum(p[:, :2], axis=1) - np.cumsum(obs[:, :2], axis=1)) ** 2).sum(axis=1)
    else:
        per = ((p - obs) ** 2).sum(axis=1)
    return float(per.mean())


def compare_hda(model_hda, market_hda, outcome, ranked=False):
    """Score a model HDA against the market HDA on the same matches, same metric.

    Both are scored against the realized result by `score_hda`, on rows where model, market
    and outcome are all present. `outcome` is `t_result` ('H'/'D'/'A') or `t_flg_diff` (1/0/-1).
    Returns a Series with:

    - `score_model`  — score of the model HDA
    - `score_market` — score of the market HDA
    - `skill`        — score_market - score_model; positive = the model is closer to the truth
    - `n`            — matches scored

    `ranked` toggles Brier (default) vs RPS, as in `score_hda`.
    """
    m = pd.DataFrame(model_hda).to_numpy(dtype=float)
    k = pd.DataFrame(market_hda).to_numpy(dtype=float)
    o = np.asarray(outcome)
    keep = ~(np.isnan(m).any(axis=1) | np.isnan(k).any(axis=1) | pd.isna(o))
    m, k, o = m[keep], k[keep], o[keep]
    s_model = score_hda(m, o, ranked=ranked)
    s_market = score_hda(k, o, ranked=ranked)
    return pd.Series({"score_model": s_model, "score_market": s_market,
                      "skill": s_market - s_model, "n": int(keep.sum())})
