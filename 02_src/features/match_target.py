import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from features.utils import round_floats
import yaml


_ROOT = Path(__file__).parents[2]
_RAW_DIR = _ROOT / "01_data" / "01_raw" / "01_matches"
_FEATURES_DIR = _ROOT / "01_data" / "02_features"
_YAML_PATH = _ROOT / "06_docs" / "data.yaml"

_LEAGUES = ["england", "spain", "italy", "germany", "france"]
_SEASONS = ["2223", "2324", "2425", "2526"]


def _columns() -> list[str]:
    with open(_YAML_PATH) as f:
        schema = yaml.safe_load(f)
    return [
        name for name, meta in schema["target_columns"].items()
        if meta and meta.get("table") == "match_target"
    ]


def _match_id(league: str, season: str, home: str, away: str) -> str:
    return hashlib.md5(f"{league}|{season}|{home}|{away}".encode()).hexdigest()


def _load_file(path: Path) -> pd.DataFrame:
    league, season = path.stem.split("_", 1)
    raw = pd.read_csv(path)

    return pd.concat([
        pd.Series(
            [_match_id(league, season, h, a) for h, a in zip(raw["HomeTeam"], raw["AwayTeam"])],
            name="match_id",
        ),
        raw["FTHG"].rename("t_home_goals"),
        raw["FTAG"].rename("t_away_goals"),
        raw["FTR"].rename("t_result"),
        (raw["FTR"] == "H").rename("t_home_flg"),
        (raw["FTR"] == "D").rename("t_draw_flg"),
        (raw["FTR"] == "A").rename("t_away_flg"),
        (raw["FTHG"] - raw["FTAG"]).rename("t_goals_diff"),
        (raw["FTHG"] + raw["FTAG"]).rename("t_goals_total"),
        pd.Series(
            np.where(raw["FTR"] == "H", 1 - 1/raw["AvgH"], -1/raw["AvgH"]),
            name="t_home_profit",
        ),
        pd.Series(
            np.where(raw["FTR"] == "D", 1 - 1/raw["AvgD"], -1/raw["AvgD"]),
            name="t_draw_profit",
        ),
        pd.Series(
            np.where(raw["FTR"] == "A", 1 - 1/raw["AvgA"], -1/raw["AvgA"]),
            name="t_away_profit",
        ),
    ], axis=1)


def build_match_target() -> pd.DataFrame:
    cols = ["match_id"] + _columns()
    frames = [
        _load_file(path)
        for league in _LEAGUES
        for season in _SEASONS
        if (path := _RAW_DIR / f"{league}_{season}.csv").exists()
    ]
    df = round_floats(pd.concat(frames, ignore_index=True)[cols])

    _FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(_FEATURES_DIR / "match_target.csv", index=False)
    df.to_parquet(_FEATURES_DIR / "match_target.parquet", index=False)

    return df
