from pathlib import Path

import pandas as pd


_ROOT = Path(__file__).parents[2]
_FEATURES_DIR = _ROOT / "01_data" / "02_features"
_ABT_DIR = _ROOT / "01_data" / "03_abt"


def build_abt() -> pd.DataFrame:
    match_info = pd.read_parquet(_FEATURES_DIR / "match_info.parquet")
    match_team_stats = pd.read_parquet(_FEATURES_DIR / "match_team_stats.parquet")
    match_matchup_stats = pd.read_parquet(_FEATURES_DIR / "match_matchup_stats.parquet")
    match_target = pd.read_parquet(_FEATURES_DIR / "match_target.parquet")

    abt = (
        match_info
        .merge(match_team_stats, on="match_id")
        .merge(match_matchup_stats, on="match_id")
        .merge(match_target, on="match_id")
    )

    _ABT_DIR.mkdir(parents=True, exist_ok=True)
    abt.to_csv(_ABT_DIR / "abt.csv", index=False)
    abt.to_parquet(_ABT_DIR / "abt.parquet", index=False)

    return abt
