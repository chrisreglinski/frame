import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from features.utils import league_set, match_files


_ROOT = Path(__file__).parents[2]
_DATA = _ROOT / "01_data"
_XG_DIR = _DATA / "01_raw" / "05_xg"

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


def _features_dir(group: str) -> Path:
    return _DATA / "02_features" / group


def _match_id(league: str, season: str, home: str, away: str) -> str:
    return hashlib.md5(f"{league}|{season}|{home}|{away}".encode()).hexdigest()


def _load_xg(group: str) -> pd.DataFrame | None:
    """Understat home_xg/away_xg keyed by match_id (Understat names translated via the
    league's team_map). Returns None if no league in the set has xg coverage."""
    frames = []
    for league_name in league_set(group):
        xg_dir = _XG_DIR / league_name
        tmap = xg_dir / "team_map.csv"
        if not tmap.exists():
            continue
        u2o = pd.read_csv(tmap).set_index("understat")["team"].to_dict()
        for path in sorted(xg_dir.glob("*_xg.csv")):
            league, season = path.stem.removesuffix("_xg").rsplit("_", 1)
            d = pd.read_csv(path, usecols=["home_team", "away_team", "home_xg", "away_xg"])
            mid = [_match_id(league, season, u2o.get(h, h), u2o.get(a, a))
                   for h, a in zip(d["home_team"], d["away_team"])]
            frames.append(pd.DataFrame({"match_id": mid,
                                        "home_xg": d["home_xg"].values,
                                        "away_xg": d["away_xg"].values}))
    return pd.concat(frames, ignore_index=True) if frames else None


def build_match_raw_stats(group: str = "major") -> pd.DataFrame:
    frames = []
    for path in match_files(group):
        league, season = path.stem.removesuffix("_matches").rsplit("_", 1)
        raw = pd.read_csv(path, usecols=lambda c: c in _RAW_COLS)
        raw.insert(0, "match_id", [
            _match_id(league, season, h, a)
            for h, a in zip(raw["HomeTeam"], raw["AwayTeam"])
        ])
        raw.insert(1, "league", league)
        raw.insert(2, "season", season)
        frames.append(raw)

    df = pd.concat(frames, ignore_index=True)

    xg = _load_xg(group)
    if xg is not None:
        df = df.merge(xg, on="match_id", how="left")
    else:
        df["home_xg"] = np.nan
        df["away_xg"] = np.nan

    features_dir = _features_dir(group)
    features_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(features_dir / "match_raw_stats.csv", index=False)
    df.to_parquet(features_dir / "match_raw_stats.parquet", index=False)

    return df


if __name__ == "__main__":
    df = build_match_raw_stats()
    print(f"done: {df.shape}")
