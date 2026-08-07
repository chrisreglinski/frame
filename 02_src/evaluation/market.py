"""Market/odds utilities for analysis (not part of the data contract).

Implied probabilities in the ABT (`*_impl`) are `1/odds` and carry the bookmaker margin,
so per match they sum to more than 1. Scoring a market quote as a probability forecast
(log-loss, RPS, ...) needs them normalized to sum 1 first — that is what `devig` does.
The home-away margin `h - a` barely needs it (the vig roughly cancels in the difference),
but full HDA quotes do.
"""
import pandas as pd


def devig(probs):
    """Remove the bookmaker margin by normalizing each match's implied probs to sum 1.

    `probs` is a DataFrame with one column per outcome (e.g. mrkt_home/draw/away_impl),
    one row per match. Returns a DataFrame of the same shape whose rows sum to 1.

    This is basic (proportional) de-vigging: the margin is split across outcomes in
    proportion to their implied probability. Shin / other schemes can live here later.
    """
    probs = pd.DataFrame(probs)
    return probs.div(probs.sum(axis=1), axis=0)
