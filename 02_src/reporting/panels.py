"""Tearsheet panels — each takes a predictions frame and returns a chart spec (data + axes).

Panels are presentation only: they call the evaluation layer to compute, then shape the result
into a spec the render layer draws as browser-side SVG. Nothing here fits models or decides
staking, and nothing draws pixels — the spec is theme-agnostic (colours are CSS token names).
"""
import pandas as pd

from evaluation.calibration import calibration_overlay
from evaluation.stats import buffer_curve, pick_buffers, portfolio_stats
from evaluation.staking import kelly_bankroll
from reporting import render


# Colour tokens (resolved against the theme at draw time) for the buffer reference lines.
_MARK = {"left": "--mark-left", "middle": "--mark-mid", "right": "--mark-right",
         "peak": "--mark-mid", "operational": "--mark-op"}

# Short legend descriptions for each reference buffer.
_MARK_DESC = {
    "left": "local profit maximum from the left",
    "right": "local profit maximum from the right",
    "middle": "midpoint of left and right",
    "peak": "the local profit maximum",
    "operational": "chosen as a report parameter",
}

_SHOWN = ["n_matches", "breakeven", "hit_rate", "roi", "profit"]


def _pts(xs, ys):
    return [[float(x), float(y)] for x, y in zip(xs, ys)]


def _buffer_marks(points, operational):
    """Which reference buffers to draw, capped so the chart never carries four lines at once.

    Twin peaks (left/right): draw `left`, `right`, and the `operational` buffer if one is given,
    else the `middle`. Single peak: draw `peak`, plus `operational` if given. So the count is three
    (left/right/operational-or-middle), two (peak/operational) or one (peak).
    """
    if "peak" in points:
        marks = {"peak": points["peak"]}
        if operational is not None:
            marks["operational"] = float(operational)
        return marks
    marks = {"left": points["left"], "right": points["right"]}
    if operational is not None:
        marks["operational"] = float(operational)
    else:
        marks["middle"] = points["middle"]
    return marks


def operative_buffer(points, operational=None):
    """The buffer to score bets at: the caller's `operational` if given, otherwise the reference
    point from pick_buffers (`middle` for twin peaks, `peak` for a single peak)."""
    if operational is not None:
        return float(operational)
    return points.get("middle", points.get("peak"))


def panel_buffer_profit(preds, operational=None):
    """Spec for profit vs bet threshold (buffer = model_p − implied): the raw and smoothed profit
    curves, with reference buffers marked as vertical lines. `operational` is the caller's chosen
    buffer; when given it is drawn (and replaces `middle` in the twin-peak case). Returns
    (spec, points).
    """
    curve = buffer_curve(preds)
    points = pick_buffers(curve)
    marks = _buffer_marks(points, operational)
    x = curve.index.to_numpy()
    raw = curve["profit"].to_numpy()
    smooth = curve["profit_smooth"].to_numpy()

    ylo = min(raw.min(), smooth.min(), 0.0)
    yhi = max(raw.max(), smooth.max(), 0.0)
    # Hover tooltip per raw point: bets, staked and profit at that buffer (values pre-formatted).
    tip_data = [[["bets", f"{int(n)}"], ["staked", f"{s:.1f}"], ["profit", f"{p:+.1f}"]]
                for n, s, p in zip(curve["n_matches"], curve["staked"], raw)]
    spec = {
        "w": 780, "h": 320, "margins": {"l": 54, "r": 96, "t": 22, "b": 44},
        "xLabel": "buffer  (model_p − implied)", "yLabel": "profit",
        "tip": True, "tipXLabel": "buffer", "tipData": tip_data,
        "xDomain": [float(x.min()), float(x.max())],
        "yDomain": render.pad_domain(ylo, yhi),
        "xTicks": render.ticks(-0.04, 0.12, 0.04, "plus2"),
        "yTicks": render.auto_ticks(ylo, yhi, 5, "int"),
        "refLines": [{"o": "h", "v": 0.0}] + [
            {"o": "v", "v": b, "color": _MARK[name], "name": name,
             "label": f"{name} {b:.3f}", "desc": _MARK_DESC[name]}
            for name, b in marks.items()
        ],
        "series": [
            {"name": "raw", "color": "--profit-faint", "points": _pts(x, raw),
             "width": 1, "r": 0, "desc": "profit at each threshold"},
            {"name": "smoothed", "color": "--profit", "points": _pts(x, smooth),
             "width": 2.4, "r": 0, "desc": "savgol on raw profit"},
        ],
        "aria": "Profit versus bet-threshold buffer, raw and smoothed, with reference buffers marked.",
    }
    return spec, points


def panel_edge_ranking(preds, window=300, mark_buffer=None):
    """Spec for the edge-ranking curve: matches sorted by edge (model_p − implied), high to low,
    with a rolling mean (width `window`) of the observed outcome rate, the model probability and the
    market (implied) probability against match rank.

    This is the local (marginal) view of the signal the profit-vs-buffer curve integrates: where the
    observed line runs above the market, the high-edge matches carry genuine excess events; the model
    line sitting above observed shows it ranks well but overshoots the level. If `mark_buffer` is
    given, a vertical line marks the rank where edge crosses it — the bets sit to its left.
    """
    d = (preds.assign(edge=preds["model_p"] - preds["implied"])
         .sort_values("edge", ascending=False).reset_index(drop=True))
    roll = lambda s: s.rolling(window, center=True, min_periods=window // 2).mean()
    rank = d.index.to_numpy()
    edge = roll(d["edge"]).to_numpy()
    obs = roll(d["y"].astype(float)).to_numpy()
    model = roll(d["model_p"]).to_numpy()
    market = roll(d["implied"]).to_numpy()

    stride = max(1, len(d) // 1200)                       # lean SVG: the curve is already smooth
    sl = slice(None, None, stride)
    n = len(d)

    # Hover tooltip per point: the local edge and the three rolling rates at that rank.
    def _row(e, o, m, k):
        return [["edge", f"{e:+.3f}"], ["observed", f"{o:.3f}"],
                ["model", f"{m:.3f}"], ["market", f"{k:.3f}"]]
    tip_data = [_row(e, o, m, k) for e, o, m, k in zip(edge[sl], obs[sl], model[sl], market[sl])]
    ylo = float(min(obs.min(), model.min(), market.min()))
    yhi = float(max(obs.max(), model.max(), market.max()))

    ref = []
    if mark_buffer is not None:
        n_bets = int((d["edge"] > mark_buffer).sum())
        ref = [{"o": "v", "v": float(n_bets), "color": "--ink", "label": f"buffer {mark_buffer:.2f}"}]

    return {
        "w": 780, "h": 320, "margins": {"l": 54, "r": 96, "t": 22, "b": 44},
        "xLabel": "matches ranked by edge  (high → low)", "yLabel": "rate / probability",
        "xDomain": [0.0, float(n)], "yDomain": render.pad_domain(ylo, yhi),
        "xTicks": render.auto_ticks(0, n, 6, "int"),
        "yTicks": render.auto_ticks(ylo, yhi, 5, "f1"),
        "tip": True, "tipXLabel": "rank", "tipXFmt": "int", "tipData": tip_data,
        "refLines": ref,
        "series": [
            {"name": "market", "color": "--market", "points": _pts(rank[sl], market[sl]),
             "width": 1.6, "r": 0},
            {"name": "model", "color": "--model", "points": _pts(rank[sl], model[sl]),
             "width": 1.6, "r": 0},
            {"name": "observed", "color": "--observed", "points": _pts(rank[sl], obs[sl]),
             "width": 2.4, "r": 0},
        ],
        "aria": "Observed rate, model and market probability by match rank, sorted by edge.",
    }


def _calibration_spec(data, by, other, other_name, other_color, x_label, title, min_per_bucket):
    """Bucket `data` by column `by`; per bucket the mean of `by`, the observed frequency and the
    mean of the `other` predictor. Bucketing by one predictor keeps every line on the same matches,
    so the points correspond; the dashed diagonal is where `by` is perfectly calibrated.
    """
    curve = calibration_overlay(data, by, other, min_per_bucket=min_per_bucket)
    x = curve[by].to_numpy()
    observed = curve["observed"].to_numpy()
    other_mean = curve[other].to_numpy()
    n_bins = len(curve)

    lo = min(x.min(), observed.min(), other_mean.min())
    hi = max(x.max(), observed.max(), other_mean.max())
    # Square domain wrapping this panel's own data tightly (four ticks), so the diagonal stays at 45
    # degrees. This is the default zoomed view; unify_calibration_scale adds the shared scale as the
    # unzoom alternate.
    dom, tk = render.wrap_scale(lo, hi)
    return {
        "w": 470, "h": 400, "margins": {"l": 60, "r": 54, "t": 16, "b": 44},
        "title": f"{title}   n={len(data)}, {n_bins} buckets",
        "xLabel": x_label, "yLabel": "frequency / probability",
        "xDomain": dom, "yDomain": dom,
        "xTicks": tk,
        "yTicks": tk,
        "refLines": [{"o": "diag"}],
        "series": [
            {"name": "observed", "color": "--observed", "points": _pts(x, observed),
             "width": 1.8, "r": 3.4},
            {"name": other_name, "color": other_color, "points": _pts(x, other_mean),
             "width": 1.8, "r": 3.4},
        ],
        "tip": True, "tipXLabel": x_label.split()[0],
        "aria": f"Calibration bucketed by {x_label}: observed frequency and {other_name} per bucket.",
    }


def panel_calibration_by_market(data, title, min_per_bucket=100):
    """Calibration spec bucketed by the market price (x = implied), overlaying observed frequency
    and the model's probability."""
    return _calibration_spec(data, "implied", "model_p", "model", "--model",
                             "market probability", title, min_per_bucket)


def panel_calibration_by_model(data, title, min_per_bucket=100):
    """Calibration spec bucketed by the model (x = model_p), overlaying observed frequency and the
    market (implied) probability."""
    return _calibration_spec(data, "model_p", "implied", "market", "--market",
                             "model probability", title, min_per_bucket)


def unify_calibration_scale(specs):
    """Add one shared square scale across a group of calibration specs, wrapping the combined data, as
    the unzoom alternate. Each panel keeps its own tight scale as the default (zoomed) view; toggling
    unzoom switches every panel to this shared scale so they become directly comparable. Mutates specs.
    """
    coords = [c for spec in specs for s in spec["series"] for p in s["points"] for c in p]
    dom, tk = render.wrap_scale(min(coords), max(coords))
    for spec in specs:
        spec["zoom"] = {"xDomain": dom, "yDomain": dom, "xTicks": tk, "yTicks": tk}
    return specs


def panel_bankroll(bets, roi, start=100.0, kelly_fraction=1.0, season_label=str):
    """Spec for the compounding Kelly bankroll over `bets` (staked in the given row order).

    Draws the bankroll against bet number from a zero baseline, a solid line at zero, a dashed line at
    the starting bankroll, and each season change as a labelled vertical line. `roi` is the portfolio
    ROI that lifts the implied probability into the stake (see evaluation.staking.kelly_bankroll).
    `season_label` formats a raw season value for its vertical-line label.
    """
    curve = kelly_bankroll(bets, roi, start=start, kelly_fraction=kelly_fraction)
    x = curve["step"].to_numpy()
    b = curve["bankroll"].to_numpy()
    yhi = max(float(b.max()), float(start))

    # A solid baseline at zero, a dashed line at the starting bankroll, then a vertical line wherever
    # the season changes.
    ref = [{"o": "h", "v": 0.0},
           {"o": "h", "v": float(start), "dash": True, "color": "--muted", "label": f"{start:.0f}",
            "desc": "starting bankroll"}]
    prev = None
    for step, season in zip(x, curve["season"].to_numpy()):
        if step != 0 and season != prev:
            ref.append({"o": "v", "v": float(step), "color": "--muted", "label": season_label(season)})
        prev = season

    def _date(v):
        try:
            return pd.Timestamp(v).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            return ""
    tip_data = [[["date", _date(d)], ["bankroll", f"{bk:.1f}"]]
                for d, bk in zip(curve["date"], b)]

    return {
        "w": 780, "h": 320, "margins": {"l": 54, "r": 96, "t": 22, "b": 44},
        "xLabel": "bet number", "yLabel": "bankroll",
        "xDomain": [0.0, float(x.max())], "yDomain": [0.0, yhi * 1.2],
        "xTicks": render.auto_ticks(0, float(x.max()), 6, "int"),
        "yTicks": render.auto_ticks(0, yhi, 5, "int"),
        "tip": True, "tipXLabel": "bet", "tipXFmt": "int", "tipData": tip_data,
        "refLines": ref,
        "series": [{"name": "bankroll", "color": "--profit", "points": _pts(x, b),
                    "width": 2.0, "r": 0}],
        "aria": "Compounding Kelly bankroll over the sequence of bets, by bet number.",
    }


def group_stats(preds, group, buffer):
    """Per-group stats at one bet threshold, straight from portfolio_stats: n_matches, breakeven,
    hit_rate, roi, profit. `group` is a column name like 'season' or 'league'.
    """
    bets = preds[preds["model_p"] > preds["implied"] + buffer]
    return bets.groupby(group).apply(portfolio_stats, include_groups=False)[_SHOWN]
