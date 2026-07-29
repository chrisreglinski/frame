import json
from pathlib import Path

import pandas as pd


_ROOT = Path(__file__).parents[2]
_DATA = _ROOT / "01_data"


def _features_dir(group: str) -> Path:
    return _DATA / "02_features" / group


# Fixed offline population fit of g(d) = a0[league] + a1*d + a2*d^2 + a3*d^3 (d = teams_elo_diff),
# fit to realized outcomes (t_home_flg - t_away_flg) per group. NOT refit per build — these are
# treated as population constants (like the cat3q tertiles). Consumed by match_matchup_stats to
# build teams_elo_diff_impl_{diff,tilt,lhfa}. a1/a3 = strength (odd), a0/a2 = home field (even).
_ELO_IMPL_FIT = {
    "major": {"a1": 0.0024872069, "a2": -3.266003e-07, "a3": -4.2390319e-09, "a0": {
        "england": 0.138645, "france": 0.117272, "germany": 0.127646,
        "italy": 0.099054, "spain": 0.188116,
    }},
    "minor": {"a1": 0.0020265984, "a2": 5.7779296e-08, "a3": -2.1137505e-09, "a0": {
        "england2": 0.127593, "france2": 0.128441, "germany2": 0.142379,
        "italy2": 0.134621, "spain2": 0.199998,
    }},
    "other": {"a1": 0.0024862658, "a2": -4.430317e-07, "a3": -3.4416436e-09, "a0": {
        "netherlands": 0.156868, "portugal": 0.143617,
    }},
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
    if group in _ELO_IMPL_FIT:
        thresholds["elo_impl"] = _ELO_IMPL_FIT[group]

    features_dir.mkdir(parents=True, exist_ok=True)
    (features_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2))

    return thresholds
