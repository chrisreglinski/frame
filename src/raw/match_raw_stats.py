import hashlib
from pathlib import Path

import pandas as pd


_ROOT = Path(__file__).parents[2]
_RAW_DIR = _ROOT / "01_data" / "01_raw" / "01_matches"
_OUT_DIR = _ROOT / "01_data" / "01_raw"

_LEAGUES = ["england", "spain", "italy", "germany", "france"]
_SEASONS = ["2223", "2324", "2425", "2526"]

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


def _match_id(league: str, season: str, home: str, away: str) -> str:
    return hashlib.md5(f"{league}|{season}|{home}|{away}".encode()).hexdigest()


def build_match_raw_stats() -> pd.DataFrame:
    frames = []
    for league in _LEAGUES:
        for season in _SEASONS:
            path = _RAW_DIR / f"{league}_{season}.csv"
            if not path.exists():
                continue
            raw = pd.read_csv(path, usecols=lambda c: c in _RAW_COLS)
            raw.insert(0, "match_id", [
                _match_id(league, season, h, a)
                for h, a in zip(raw["HomeTeam"], raw["AwayTeam"])
            ])
            raw.insert(1, "league", league)
            raw.insert(2, "season", season)
            frames.append(raw)

    df = pd.concat(frames, ignore_index=True)

    df.to_csv(_OUT_DIR / "match_raw_stats.csv", index=False)
    df.to_parquet(_OUT_DIR / "match_raw_stats.parquet", index=False)

    return df


if __name__ == "__main__":
    df = build_match_raw_stats()
    print(f"done: {df.shape}")
