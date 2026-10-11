"""Scoring predicted quantities against realized outcomes, over a whole portfolio.

Every scorer reports a *skill score* — the R^2-style "fraction of variance explained"
against the trivial constant baseline (predict the mean / the base rate), not the raw error.
For a margin that is the ordinary R^2 = 1 - MSE/Var; for a probability forecast it is the
Brier (or ranked-probability) skill score = 1 - score/baseline_score. Higher is better, 0 =
no better than the constant baseline, negative = worse than it.

`score_*` scores one predictor; `compare_*` scores a model against a reference (e.g. the
market) on the same rows and returns both plus their gap. Outcomes: `score_diff` uses
`t_flg_diff` (the margin's target); `score_hda` uses `t_result` or `t_flg_diff`;
`score_outcome` uses a 0/1 flag (e.g. `t_draw_flg`). All map cleanly through groupby.
"""
import numpy as np
import pandas as pd


def score_diff(margin, outcome_diff):
    """R^2 of a predicted home-away margin against `t_flg_diff` (1 - MSE / Var(outcome)).

    `margin` — predicted h - a per match (e.g. rr_impl_diff, mrkt_home_away_impl_diff).
    `outcome_diff` — realized `t_flg_diff` in {1, 0, -1}. NaN rows dropped. Baseline =
    predicting the mean outcome; higher = margin explains more of the signed-result variance.
    """
    df = pd.DataFrame({"margin": np.asarray(margin, dtype=float),
                       "outcome": np.asarray(outcome_diff, dtype=float)}).dropna()
    mse = ((df["margin"] - df["outcome"]) ** 2).mean()
    var = ((df["outcome"] - df["outcome"].mean()) ** 2).mean()
    return float(1 - mse / var)


def compare_diff(model_margin, market_margin, outcome_diff):
    """R^2 of a model margin and a reference (market) margin against `t_flg_diff`, same rows.

    Scored on rows where model, market and outcome are all present. Returns a Series with:

    - `r2_model` / `r2_market` — R^2 of each margin vs the mean-of-outcome baseline
    - `skill`                  — r2_model - r2_market; positive = model explains more variance
    - `n`                      — matches scored

    Maps cleanly through groupby for a per-league / per-season breakdown.
    """
    df = pd.DataFrame({"model": np.asarray(model_margin, dtype=float),
                       "market": np.asarray(market_margin, dtype=float),
                       "outcome": np.asarray(outcome_diff, dtype=float)}).dropna()
    r2_model = score_diff(df["model"], df["outcome"])
    r2_market = score_diff(df["market"], df["outcome"])
    return pd.Series({"r2_model": r2_model, "r2_market": r2_market,
                      "skill": r2_model - r2_market, "n": len(df)})


def score_outcome(prob, flag):
    """Brier skill score (R^2 for a binary probability) of a single-outcome prob vs its flag.

    `prob` — predicted probability of one outcome (e.g. mrkt_draw_impl). `flag` — realized
    1/0 indicator (e.g. t_draw_flg). NaN rows dropped. 1 - Brier / Brier_baseline, baseline =
    the constant base-rate predictor (Brier = ybar*(1-ybar)). Higher = beats the base rate.
    """
    df = pd.DataFrame({"prob": np.asarray(prob, dtype=float),
                       "flag": np.asarray(flag, dtype=float)}).dropna()
    brier = ((df["prob"] - df["flag"]) ** 2).mean()
    base = df["flag"].mean()
    return float(1 - brier / (base * (1 - base)))


def compare_outcome(model_prob, market_prob, flag):
    """Brier skill score of a model's single-outcome prob and a reference (market) prob vs flag.

    Scored on rows where model, market and flag are all present. Returns a Series with:

    - `r2_model` / `r2_market` — Brier skill score of each probability
    - `skill`                  — r2_model - r2_market; positive = model beats the reference
    - `n`                      — matches scored
    """
    df = pd.DataFrame({"model": np.asarray(model_prob, dtype=float),
                       "market": np.asarray(market_prob, dtype=float),
                       "flag": np.asarray(flag, dtype=float)}).dropna()
    r2_model = score_outcome(df["model"], df["flag"])
    r2_market = score_outcome(df["market"], df["flag"])
    return pd.Series({"r2_model": r2_model, "r2_market": r2_market,
                      "skill": r2_model - r2_market, "n": len(df)})


def _onehot_hda(outcome):
    """One-hot [H, D, A] from either `t_result` ('H'/'D'/'A') or `t_flg_diff` (1/0/-1)."""
    o = np.asarray(outcome)
    cols = [o == "H", o == "D", o == "A"] if o.dtype.kind in ("U", "S", "O") \
        else [o == 1, o == 0, o == -1]
    return np.stack(cols, axis=1).astype(float)


def _hda_error(p, obs, ranked):
    """Per-match quadratic error: Brier (raw probs) or RPS (cumulative, respects H>D>A order)."""
    if ranked:
        return 0.5 * ((np.cumsum(p[:, :2], axis=1) - np.cumsum(obs[:, :2], axis=1)) ** 2).sum(axis=1)
    return ((p - obs) ** 2).sum(axis=1)


def score_hda(hda, outcome, ranked=False):
    """Skill score (R^2 for the H/D/A distribution) against the constant base-rate baseline.

    `hda` columns are taken in home, draw, away order (e.g. impl_h/impl_d/impl_a from
    hda_from_margin, or devigged mrkt_home/draw/away_impl); each row is normalized to sum 1.
    `outcome` is `t_result` ('H'/'D'/'A') or `t_flg_diff` (1/0/-1). NaN rows dropped.

    Default is the Brier skill score (non-ranked — a wrong class is a wrong class, as for a
    settled bet); `ranked=True` gives the ranked-probability skill score (RPSS), which credits
    a miss to an adjacent outcome. 1 - score / baseline_score; higher = beats the base rate.
    """
    p = pd.DataFrame(hda).to_numpy(dtype=float)
    o = np.asarray(outcome)
    keep = ~(np.isnan(p).any(axis=1) | pd.isna(o))
    p, o = p[keep], o[keep]
    p = p / p.sum(axis=1, keepdims=True)                       # normalize to sum 1
    obs = _onehot_hda(o)
    model = _hda_error(p, obs, ranked).mean()
    baseline = _hda_error(np.broadcast_to(obs.mean(axis=0), p.shape), obs, ranked).mean()
    return float(1 - model / baseline)


def compare_hda(model_hda, market_hda, outcome, ranked=False):
    """Skill score of a model HDA and a reference (market) HDA against the realized result.

    Scored on rows where model, market and outcome are all present. `outcome` is `t_result`
    ('H'/'D'/'A') or `t_flg_diff` (1/0/-1). Returns a Series with:

    - `r2_model` / `r2_market` — skill score of each HDA (Brier, or RPSS if ranked)
    - `skill`                  — r2_model - r2_market; positive = model beats the reference
    - `n`                      — matches scored

    `ranked` toggles Brier skill score (default) vs RPSS, as in `score_hda`.
    """
    m = pd.DataFrame(model_hda).to_numpy(dtype=float)
    k = pd.DataFrame(market_hda).to_numpy(dtype=float)
    o = np.asarray(outcome)
    keep = ~(np.isnan(m).any(axis=1) | np.isnan(k).any(axis=1) | pd.isna(o))
    m, k, o = m[keep], k[keep], o[keep]
    r2_model = score_hda(m, o, ranked=ranked)
    r2_market = score_hda(k, o, ranked=ranked)
    return pd.Series({"r2_model": r2_model, "r2_market": r2_market,
                      "skill": r2_model - r2_market, "n": int(keep.sum())})
