"""Market/odds utilities for analysis (not part of the data contract).

Implied probabilities in the ABT (`*_impl`) are `1/odds` and carry the bookmaker margin,
so per match they sum to more than 1. Scoring a market quote as a probability forecast
(log-loss, RPS, ...) needs them normalized to sum 1 first — that is what `devig` does.
The home-away margin `h - a` barely needs it (the vig roughly cancels in the difference),
but full HDA quotes do.
"""
import numpy as np
import pandas as pd


# Draw-probability curve: P(D) = a - b*margin**2, margin = impl_home - impl_away.
# Deg-2 parabola fitted on the major group (peak `a` at margin 0, curvature `b`).
# Presets by line (mp pre-closing / mc closing) x devig; devig shaves ~1pp off the peak,
# closing ~ pre-closing. Use a devigged preset when the H/D/A triple must sum to 1.
DRAW_CURVE = {
    "mc_raw":      (0.30, 0.30),   # closing, with vig (default)
    "mp_raw":      (0.30, 0.29),   # pre-closing, with vig
    "mc_devigged": (0.29, 0.31),   # closing, devigged
    "mp_devigged": (0.29, 0.30),   # pre-closing, devigged
}


def impl_draw_from_margin(margin, preset="mc_raw", a=None, b=None):
    """Estimate the draw implied probability from the HDA margin (impl_home - impl_away).

    P(D) = a - b*margin**2, clipped to [0, 1]. Defaults to the `mc_raw` line; pass
    `preset` for another line/devig (see DRAW_CURVE) or `a`/`b` to override coefficients.
    Reconstruct a full triple with a devigged preset: it sums to 1 by construction —
    P(H) = (1 - P(D) + margin)/2, P(A) = (1 - P(D) - margin)/2.
    """
    pa, pb = DRAW_CURVE[preset]
    a = pa if a is None else a
    b = pb if b is None else b
    return np.clip(a - b * margin**2, 0.0, 1.0)


def hda_from_margin(margin, impl_d=None, *, preset="mc_raw", clip=False):
    """Home/draw/away probability triple from a margin (impl_home - impl_away) and a draw
    probability. Returns a DataFrame [impl_h, impl_d, impl_a] summing to 1:
        impl_h = (1 - impl_d + margin) / 2,  impl_a = (1 - impl_d - margin) / 2.
    If impl_d is None it is derived from margin via impl_draw_from_margin(preset). A tail
    can go below 0 when inputs are inconsistent; clip=True clamps to [0, 1] and renormalizes.
    """
    if impl_d is None:
        impl_d = impl_draw_from_margin(margin, preset=preset)
    home = (1 - impl_d + margin) / 2
    away = (1 - impl_d - margin) / 2
    if np.ndim(margin) == 0 and np.ndim(impl_d) == 0:
        out = pd.DataFrame([[home, impl_d, away]], columns=["impl_h", "impl_d", "impl_a"])
    else:
        out = pd.DataFrame({"impl_h": home, "impl_d": impl_d, "impl_a": away})
    if clip:
        out = out.clip(lower=0.0)
        out = out.div(out.sum(axis=1), axis=0)
    return out


def devig(probs):
    """Remove the bookmaker margin by normalizing each match's implied probs to sum 1.

    `probs` is a DataFrame with one column per outcome (e.g. mrkt_home/draw/away_impl),
    one row per match. Returns a DataFrame of the same shape whose rows sum to 1.

    This is basic (proportional) de-vigging: the margin is split across outcomes in
    proportion to their implied probability. Shin / other schemes can live here later.
    """
    probs = pd.DataFrame(probs)
    return probs.div(probs.sum(axis=1), axis=0)
