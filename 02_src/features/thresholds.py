import json
from pathlib import Path

import pandas as pd


_ROOT = Path(__file__).parents[2]
_RAW_STATS_PATH = _ROOT / "01_data" / "01_raw" / "match_raw_stats.parquet"
_FEATURES_DIR = _ROOT / "01_data" / "02_features"
_OUT_PATH = _FEATURES_DIR / "thresholds.json"


def build_thresholds() -> dict:
    raw = pd.read_parquet(_RAW_STATS_PATH)
    raw["goals_total"] = raw["FTHG"] + raw["FTAG"]

    goals_total_mean = float(raw["goals_total"].mean())
    thresholds = {
        "goals_total_mean": round(goals_total_mean, 3),
        # from team perspective each match appears as home and away,
        # so goals_for_mean == goals_agst_mean == goals_total_mean / 2
        "goals_for_mean":  round(goals_total_mean / 2, 3),
        "goals_agst_mean": round(goals_total_mean / 2, 3),
    }

    _FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    _OUT_PATH.write_text(json.dumps(thresholds, indent=2))

    return thresholds
