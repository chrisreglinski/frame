import json
from pathlib import Path

import pandas as pd


_ROOT = Path(__file__).parents[2]
_DATA = _ROOT / "01_data"


def _features_dir(group: str) -> Path:
    return _DATA / "02_features" / group


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

    features_dir.mkdir(parents=True, exist_ok=True)
    (features_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2))

    return thresholds
