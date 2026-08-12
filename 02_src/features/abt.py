from pathlib import Path

import pandas as pd

from features.utils import round_floats


_ROOT = Path(__file__).parents[2]
_DATA = _ROOT / "01_data"


def _features_dir(group: str) -> Path:
    return _DATA / "02_features" / group


def _abt_dir(group: str) -> Path:
    return _DATA / "03_abt" / group


def build_abt(group: str = "major") -> pd.DataFrame:
    features_dir = _features_dir(group)
    match_info = pd.read_parquet(features_dir / "match_info.parquet")
    match_team_stats = pd.read_parquet(features_dir / "match_team_stats.parquet")
    match_matchup_stats = pd.read_parquet(features_dir / "match_matchup_stats.parquet")
    match_team_fatigue = pd.read_parquet(features_dir / "match_team_fatigue.parquet")
    match_target = pd.read_parquet(features_dir / "match_target.parquet")

    abt = round_floats(
        match_info
        .merge(match_team_stats, on="match_id")
        .merge(match_matchup_stats, on="match_id")
        .merge(match_team_fatigue, on="match_id")
        .merge(match_target, on="match_id")
    )

    abt_dir = _abt_dir(group)
    abt_dir.mkdir(parents=True, exist_ok=True)
    abt.to_csv(abt_dir / "abt.csv", index=False)
    abt.to_parquet(abt_dir / "abt.parquet", index=False)

    return abt
