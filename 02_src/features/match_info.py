import hashlib
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


_ROOT = Path(__file__).parents[2]
_RAW_DIR = _ROOT / "01_data" / "01_raw" / "01_matches"
_STADIUMS_DIR = _ROOT / "01_data" / "01_raw" / "02_stadiums"
_FEATURES_DIR = _ROOT / "01_data" / "02_features"
_YAML_PATH = _ROOT / "06_docs" / "data.yaml"
_DATES_PATH = _ROOT / "01_data" / "01_raw" / "03_dates" / "season_limit_dates.csv"

_LEAGUES = ["england", "spain", "italy", "germany", "france"]
_SEASONS = ["2223", "2324", "2425", "2526"]

def _load_phase_limits() -> dict[str, dict[str, pd.Timestamp]]:
    df = pd.read_csv(_DATES_PATH, dtype={"season": str}, parse_dates=["summer_fall", "fall_winter", "winter_spring"])
    return {
        row["season"]: {
            "summer_fall":  row["summer_fall"],
            "fall_winter":  row["fall_winter"],
            "winter_spring": row["winter_spring"],
        }
        for _, row in df.iterrows()
    }


def _season_4phase(date: pd.Series, season: str, limits: dict) -> pd.Series:
    lim = limits[season]
    return pd.cut(
        date,
        bins=[pd.Timestamp.min, lim["summer_fall"], lim["fall_winter"], lim["winter_spring"], pd.Timestamp.max],
        labels=["summer", "fall", "winter", "spring"],
        right=False,
    ).astype(str)


_3PHASE_MAP = {"summer": "start", "fall": "mid", "winter": "mid", "spring": "end"}


def _season_3phase(phase4: pd.Series) -> pd.Series:
    return phase4.map(_3PHASE_MAP)


_RAW_ODDS = {
    "b365": {"home": "B365H", "draw": "B365D", "away": "B365A"},
    "mrkt": {"home": "AvgH",  "draw": "AvgD",  "away": "AvgA"},
}


def _columns() -> list[str]:
    with open(_YAML_PATH) as f:
        schema = yaml.safe_load(f)
    result = []
    for name, meta in schema["columns"].items():
        if not meta:
            continue
        table = meta.get("table")
        if "match_info" not in (table if isinstance(table, list) else [table]):
            continue
        if "dims" in meta:
            keys = list(meta["dims"].keys())
            values = [meta["dims"][k] for k in keys]
            for combo in itertools.product(*values):
                result.append(name.format(**dict(zip(keys, combo))))
        else:
            result.append(name)
    return result


def _match_id(league: str, season: str, home: str, away: str) -> str:
    return hashlib.md5(f"{league}|{season}|{home}|{away}".encode()).hexdigest()


def _parse_dates(series: pd.Series) -> pd.Series:
    for fmt in ("%d/%m/%y", "%d/%m/%Y"):
        try:
            return pd.to_datetime(series, format=fmt)
        except ValueError:
            continue
    return pd.to_datetime(series, dayfirst=True)


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _load_stadiums(league: str, season: str) -> dict[str, dict]:
    path = _STADIUMS_DIR / f"{league}_{season}_stadiums.csv"
    df = pd.read_csv(path)
    return {
        row["TeamName"]: {
            "lat": row["Latitude"],
            "lon": row["Longitude"],
            "promoted": bool(row["Promotee"]),
        }
        for _, row in df.iterrows()
    }


def _check_missing(teams: set[str], stadiums: dict, league: str, season: str) -> None:
    missing = sorted(teams - stadiums.keys())
    if missing:
        lines = [f"  {league} {season}: {t}" for t in missing]
        raise ValueError("missing stadium entries:\n" + "\n".join(lines))


def _load_file(path: Path, stadiums: dict[str, dict], limits: dict) -> pd.DataFrame:
    league, season = path.stem.split("_", 1)
    raw = pd.read_csv(path)

    all_teams = set(raw["HomeTeam"]) | set(raw["AwayTeam"])
    _check_missing(all_teams, stadiums, league, season)

    matches_per_gw = len(all_teams) // 2
    season_game_number = pd.Series(range(1, len(raw) + 1))
    gameweek = ((season_game_number - 1) // matches_per_gw + 1).rename("gameweek")

    date = _parse_dates(raw["Date"])
    odds, impl, margins = {}, {}, {}

    for bk, mapping in _RAW_ODDS.items():
        for outcome, raw_col in mapping.items():
            odds[f"{bk}_{outcome}_odds"] = raw[raw_col]
            impl[f"{bk}_{outcome}_impl"] = 1.0 / raw[raw_col]
        margins[f"{bk}_margin"] = sum(1.0 / raw[c] for c in mapping.values()) - 1

    entropies = {}
    for bk, mapping in _RAW_ODDS.items():
        p = pd.DataFrame({o: impl[f"{bk}_{o}_impl"] for o in mapping})
        p_norm = p.div(p.sum(axis=1), axis=0)
        entropies[f"{bk}_entropy"] = -(p_norm * np.log2(p_norm)).sum(axis=1)

    mrkt_p = pd.DataFrame({o: impl[f"mrkt_{o}_impl"] for o in ["home", "draw", "away"]})
    mrkt_p_norm = mrkt_p.div(mrkt_p.sum(axis=1), axis=0)
    mrkt_favourite = pd.Series(np.where(
        mrkt_p_norm["home"] > 0.5, "home",
        np.where(mrkt_p_norm["away"] > 0.5, "away", "none")
    ), name="mrkt_favourite")
    mrkt_impl_order = pd.Series([
        "".join(k[0] for k in sorted(
            [("h", h), ("d", d), ("a", a)], key=lambda x: -x[1]
        ))
        for h, d, a in zip(
            impl["mrkt_home_impl"], impl["mrkt_draw_impl"], impl["mrkt_away_impl"]
        )
    ], name="mrkt_impl_order")

    home_is_promoted = raw["HomeTeam"].map(lambda t: stadiums[t]["promoted"])
    away_is_promoted = raw["AwayTeam"].map(lambda t: stadiums[t]["promoted"])
    travel = pd.Series([
        _haversine(
            stadiums[h]["lat"], stadiums[h]["lon"],
            stadiums[a]["lat"], stadiums[a]["lon"],
        )
        for h, a in zip(raw["HomeTeam"], raw["AwayTeam"])
    ], name="travel_distance_km")

    return pd.concat([
        pd.Series(
            [_match_id(league, season, h, a) for h, a in zip(raw["HomeTeam"], raw["AwayTeam"])],
            name="match_id",
        ),
        pd.Series([league] * len(raw), name="league"),
        pd.Series([season] * len(raw), name="season"),
        date.rename("date"),
        raw["Time"].rename("time"),
        date.dt.day_name().rename("day_of_week"),
        season_game_number.rename("season_game_number"),
        gameweek,
        _season_4phase(date, season, limits).rename("season_4phase"),
        _season_3phase(_season_4phase(date, season, limits)).rename("season_3phase"),
        raw["HomeTeam"].rename("home_team"),
        raw["AwayTeam"].rename("away_team"),
        home_is_promoted.rename("home_is_promoted"),
        away_is_promoted.rename("away_is_promoted"),
        travel,
        pd.DataFrame(odds),
        pd.DataFrame(impl),
        pd.DataFrame(margins),
        pd.DataFrame(entropies),
        mrkt_favourite,
        mrkt_impl_order,
    ], axis=1)


def build_match_info() -> pd.DataFrame:
    cols = ["match_id"] + _columns()
    limits = _load_phase_limits()
    frames = []
    for league in _LEAGUES:
        for season in _SEASONS:
            path = _RAW_DIR / f"{league}_{season}.csv"
            if not path.exists():
                continue
            stadiums = _load_stadiums(league, season)
            frames.append(_load_file(path, stadiums, limits))

    df = pd.concat(frames, ignore_index=True)[cols]

    _FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(_FEATURES_DIR / "match_info.csv", index=False)
    df.to_parquet(_FEATURES_DIR / "match_info.parquet", index=False)

    return df
