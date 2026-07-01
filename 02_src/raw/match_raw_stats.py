import hashlib
from pathlib import Path

import pandas as pd


_ROOT = Path(__file__).parents[2]
_DATA = _ROOT / "01_data"

_RAW_COLS = [
    "Date", "Time",
    "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR",
    "HS", "AS", "HST", "AST",
    "HC", "AC",
    "HY", "AY",
    "HR", "AR",
    "B365H", "B365D", "B365A",
    "AvgH", "AvgD", "AvgA",
]


def _matches_dir(group: str) -> Path:
    return _DATA / "01_raw" / "01_matches" / group


def _features_dir(group: str) -> Path:
    return _DATA / "02_features" / group


def _match_id(league: str, season: str, home: str, away: str) -> str:
    return hashlib.md5(f"{league}|{season}|{home}|{away}".encode()).hexdigest()


def build_match_raw_stats(group: str = "major") -> pd.DataFrame:
    frames = []
    for path in sorted(_matches_dir(group).glob("*.csv")):
        league, season = path.stem.rsplit("_", 1)
        raw = pd.read_csv(path, usecols=lambda c: c in _RAW_COLS)
        raw.insert(0, "match_id", [
            _match_id(league, season, h, a)
            for h, a in zip(raw["HomeTeam"], raw["AwayTeam"])
        ])
        raw.insert(1, "league", league)
        raw.insert(2, "season", season)
        frames.append(raw)

    df = pd.concat(frames, ignore_index=True)

    features_dir = _features_dir(group)
    features_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(features_dir / "match_raw_stats.csv", index=False)
    df.to_parquet(features_dir / "match_raw_stats.parquet", index=False)

    return df


if __name__ == "__main__":
    df = build_match_raw_stats()
    print(f"done: {df.shape}")
