"""Run a model: apply it to data under a protocol, produce a Run.

A Run is one instantiation of a model on a specific data snapshot and evaluation protocol. It carries
the model, the out-of-fold predictions the report draws from, the buffer that was used and how it was
chosen, and the frozen headline metrics. The same model on a different set of seasons is a different
run. Training lives here, not in the report: the report consumes a finished Run.
"""
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from evaluation.folds import season_folds
from evaluation.predictions import collect_predictions
from evaluation.staking import kelly_bankroll
from evaluation.stats import buffer_curve, pick_buffers, portfolio_stats
from models import factory
from models._spec import Model, apply_domain

_ABT_ROOT = Path(__file__).parents[2] / "01_data" / "03_abt"


@dataclass
class Run:
    """One instantiation of a model. `predictions` are out-of-fold, `metrics` are frozen headline
    numbers, `buffer` is the bet threshold used and `buffer_name` how it was chosen."""
    model: Model
    predictions: pd.DataFrame
    buffer: float
    buffer_name: str
    holdout: object
    seasons: list
    bank_start: float
    kelly_fraction: float
    metrics: dict = field(default_factory=dict)


def _abt_path(leagues):
    return _ABT_ROOT / leagues / "abt.parquet"


def _devig(matches):
    """Fair (devigged) draw probability per match_id: the H/D/A implied prices normalized to sum to
    one, removing the bookmaker margin. Raw implied is a price (break-even), not a real probability."""
    total = matches["mrkt_home_impl"] + matches["mrkt_draw_impl"] + matches["mrkt_away_impl"]
    fair = (matches["mrkt_draw_impl"] / total).to_numpy()
    return pd.Series(fair, index=matches["match_id"].to_numpy())


def run(model, abt=None, buffer=None, holdout="last", bank_start=100.0, kelly_fraction=0.5):
    """Apply `model` to data and return a Run.

    `abt` defaults to the prebuilt ABT for the model's league set. `buffer` is the bet threshold
    (bet where model_p > implied + buffer): when None it is picked from the profit curve on the
    development seasons (every season but the holdout), `middle` for twin peaks and `peak` for a
    single one. `holdout` is the season kept out of buffer selection ("last" for the newest). Staking
    and ROI use the raw implied price, the p-value null uses the devigged probability, and the
    bankroll compounds at kelly_fraction from bank_start.
    """
    if abt is None:
        abt = pd.read_parquet(_abt_path(model.domain.leagues))

    feats = list(model.features)
    matches = apply_domain(abt, model.domain).dropna(
        subset=feats + [model.implied, model.target, "league", "season"]).copy()

    def make():
        return factory.build(model.estimator, model.hyperparams)

    preds = collect_predictions(matches, season_folds(matches), features=feats,
                                outcome=model.target, implied=model.implied, make_model=make)

    fair = _devig(matches)
    preds["fair"] = preds["match_id"].map(fair).fillna(preds["implied"])

    # Attach the model's feature columns, so the saved register is a thin self-contained ABT.
    feat_cols = [c for c in feats if c not in preds.columns]
    preds = preds.merge(matches[["match_id", *feat_cols]], on="match_id", how="left")

    seasons = sorted(preds["season"].unique())
    hold = seasons[-1] if holdout == "last" else holdout

    if buffer is None:
        points = pick_buffers(buffer_curve(preds[preds["season"] != hold]))
        buffer = points.get("middle", points.get("peak"))
        buffer_name = "middle" if "middle" in points else "peak"
    else:
        buffer = float(buffer)
        buffer_name = "operational"

    bets = preds[preds["model_p"] > preds["implied"] + buffer]
    ps = portfolio_stats(bets, prob="fair")

    bank = kelly_bankroll(bets.sort_values("date"), ps["roi"], start=bank_start,
                          kelly_fraction=kelly_fraction)
    bank_final = float(bank["bankroll"].iloc[-1])
    bank_maxdd = float(bank["drawdown"].min())
    d0, d1 = pd.Timestamp(bank["date"].iloc[0]), pd.Timestamp(bank["date"].iloc[-1])
    bank_cagr = (bank_final / bank_start) ** (365.25 / max((d1 - d0).days, 1)) - 1.0

    metrics = {
        "n_bets": int(ps["n_matches"]), "staked": float(ps["staked"]),
        "wins": float(ps["wins"]), "profit": float(ps["profit"]), "roi": float(ps["roi"]),
        "hit_rate": float(ps["hit_rate"]), "breakeven": float(ps["breakeven"]),
        "p_value": float(ps["p_value"]),
        "bank_final": bank_final, "bank_maxdd": bank_maxdd, "bank_cagr": bank_cagr,
    }
    return Run(model=model, predictions=preds, buffer=buffer, buffer_name=buffer_name,
               holdout=hold, seasons=seasons, bank_start=bank_start,
               kelly_fraction=kelly_fraction, metrics=metrics)
