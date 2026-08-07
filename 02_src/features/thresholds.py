import json
from pathlib import Path

import pandas as pd


_ROOT = Path(__file__).parents[2]
_DATA = _ROOT / "01_data"


def _features_dir(group: str) -> Path:
    return _DATA / "02_features" / group


# Fixed offline population fit (on gw>8) of the elo -> home-minus-away margin, per group. NOT refit
# per build — population constants (like the cat3q tertiles). Consumed by match_matchup_stats to build
# teams_elo_diff_{crr,cmr,lrr,lmr}_impl_{diff,stgh,lhfa}. 2x2 grid: FORM (c=cubic / l=logistic) x
# FIT (rr=results / mr=market).
#   CUBIC  g(d) = a0[league] + a1*d + a2*d^2 + a3*d^3;  a1/a3 = strength (odd), a0/a2 = home field (even).
#   LOGISTIC margin = 2/(1 + 10^(-(d + hfa[league])/scale)) - 1 (ClubElo Elo equation); global scale +
#            per-league hfa (elo points). Fixed shape, saturates -> no clamping; strength = same formula
#            with hfa=0, home field = the rest.
#   rr = fit to realized outcomes (t_flg_diff): the TRUE margin;  mr = fit to mrkt_home_away_impl_diff.
# The logistic scale is near-identical for rr and mr (market reads elo strength at the same steepness);
# its whole rr/mr gap lives in hfa — the market's per-league home-advantage mispricing.
_ELO_FIT = {
    "major": {
        "crr": {"a1": 0.0024872069, "a2": -3.266003e-07, "a3": -4.2390319e-09, "a0": {
            "england": 0.138645, "france": 0.117272, "germany": 0.127646,
            "italy": 0.099054, "spain": 0.188116,
        }},
        "cmr": {"a1": 0.002333523, "a2": -4.2150299e-07, "a3": -3.0138405e-09, "a0": {
            "england": 0.136515, "france": 0.136017, "germany": 0.150326,
            "italy": 0.129036, "spain": 0.16567,
        }},
        "lrr": {"scale": 474.5185, "hfa": {
            "england": 59.713, "france": 47.2109, "germany": 64.5713,
            "italy": 37.68, "spain": 81.5589,
        }},
        "lmr": {"scale": 478.4904, "hfa": {
            "england": 58.5409, "france": 59.2285, "germany": 66.8138,
            "italy": 56.0349, "spain": 73.7742,
        }},
    },
    "minor": {
        "crr": {"a1": 0.0020265984, "a2": 5.7779296e-08, "a3": -2.1137505e-09, "a0": {
            "england2": 0.127593, "france2": 0.128441, "germany2": 0.142379,
            "italy2": 0.134621, "spain2": 0.199998,
        }},
        "cmr": {"a1": 0.0020258545, "a2": -4.638334e-07, "a3": -1.577909e-10, "a0": {
            "england2": 0.137282, "france2": 0.125404, "germany2": 0.126702,
            "italy2": 0.135681, "spain2": 0.173019,
        }},
        "lrr": {"scale": 508.2327, "hfa": {
            "england2": 56.7789, "france2": 63.398, "germany2": 68.4808,
            "italy2": 70.6654, "spain2": 99.1074,
        }},
        "lmr": {"scale": 516.2429, "hfa": {
            "england2": 62.2219, "france2": 56.6338, "germany2": 57.8411,
            "italy2": 62.7229, "spain2": 79.8481,
        }},
    },
    "other": {
        "crr": {"a1": 0.0024862658, "a2": -4.430317e-07, "a3": -3.4416436e-09, "a0": {
            "netherlands": 0.156868, "portugal": 0.143617,
        }},
        "cmr": {"a1": 0.0022225849, "a2": -4.6697663e-07, "a3": -2.0880864e-09, "a0": {
            "netherlands": 0.159148, "portugal": 0.133802,
        }},
        "lrr": {"scale": 452.6942, "hfa": {
            "netherlands": 61.6623, "portugal": 60.7149,
        }},
        "lmr": {"scale": 483.4978, "hfa": {
            "netherlands": 70.6544, "portugal": 57.2638,
        }},
    },
}


def build_thresholds(group: str = "major") -> dict:
    features_dir = _features_dir(group)
    raw = pd.read_parquet(features_dir / "match_raw_stats.parquet")
    raw["goals_total"] = raw["FTHG"] + raw["FTAG"]
    raw["sot_total"]   = raw["HST"]  + raw["AST"]

    goals_total_mean = float(raw["goals_total"].mean())
    sot_total_mean   = float(raw["sot_total"].mean())
    thresholds = {
        "goals_total_mean":             round(goals_total_mean, 3),
        "goals_foragst_mean":           round(goals_total_mean / 2, 3),
        "shots_on_target_total_mean":   round(sot_total_mean, 3),
        "shots_on_target_foragst_mean": round(sot_total_mean / 2, 3),
    }
    if group in _ELO_FIT:
        for tag in ("crr", "cmr", "lrr", "lmr"):
            thresholds[f"elo_{tag}"] = _ELO_FIT[group][tag]

    features_dir.mkdir(parents=True, exist_ok=True)
    (features_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2))

    return thresholds
