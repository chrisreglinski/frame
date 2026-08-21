import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from features.utils import match_files, round_floats
import yaml


_ROOT = Path(__file__).parents[2]
_DATA = _ROOT / "01_data"
_YAML_PATH = _ROOT / "06_docs" / "data.yaml"


def _features_dir(group: str) -> Path:
    return _DATA / "02_features" / group


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
    league, season = path.stem.removesuffix("_matches").rsplit("_", 1)
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
        raw["FTR"].map({"H": 1, "D": 0, "A": -1}).rename("t_flg_diff"),
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


def build_match_target(group: str = "major") -> pd.DataFrame:
    cols = ["match_id"] + _columns()
    features_dir = _features_dir(group)
    frames = [_load_file(path) for path in match_files(group)]
    df = round_floats(pd.concat(frames, ignore_index=True)[cols])

    features_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(features_dir / "match_target.csv", index=False)
    df.to_parquet(features_dir / "match_target.parquet", index=False)

    return df
