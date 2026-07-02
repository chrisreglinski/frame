import hashlib
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from features.utils import round_floats


_ROOT = Path(__file__).parents[2]
_DATA = _ROOT / "01_data"
_STADIUMS_DIR = _DATA / "01_raw" / "02_stadiums"
_YAML_PATH = _ROOT / "06_docs" / "data.yaml"
_DATES_PATH = _DATA / "01_raw" / "03_dates" / "season_limit_dates.csv"
_ELO_DIR = _DATA / "01_raw" / "04_elo"


def _matches_dir(group: str) -> Path:
    return _DATA / "01_raw" / "01_matches" / group


def _features_dir(group: str) -> Path:
    return _DATA / "02_features" / group

def _load_phase_limits() -> dict[str, dict[str, pd.Timestamp]]:
    if not _DATES_PATH.exists():
        return {}
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
    if season not in limits:
        return pd.Series(np.nan, index=date.index)
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


def _load_stadiums(league: str, season: str) -> dict[str, dict] | None:
    path = _STADIUMS_DIR / f"{league}_{season}_stadiums.csv"
    if not path.exists():
        return None
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
    league, season = path.stem.rsplit("_", 1)
    raw = pd.read_csv(path)

    all_teams = set(raw["HomeTeam"]) | set(raw["AwayTeam"])
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
        np.where(mrkt_p_norm["away"] > 0.5, "away", "balanced")
    ), name="mrkt_favourite")
    mrkt_impl_order = pd.Series([
        "".join(k[0] for k in sorted(
            [("h", h), ("d", d), ("a", a)], key=lambda x: -x[1]
        ))
        for h, d, a in zip(
            impl["mrkt_home_impl"], impl["mrkt_draw_impl"], impl["mrkt_away_impl"]
        )
    ], name="mrkt_impl_order")
    mrkt_favrt_impl = pd.Series(
        np.maximum(impl["mrkt_home_impl"], impl["mrkt_away_impl"]),
        name="mrkt_favrt_impl",
    )
    mrkt_undrd_impl = pd.Series(
        np.minimum(impl["mrkt_home_impl"], impl["mrkt_away_impl"]),
        name="mrkt_undrd_impl",
    )
    mrkt_home_away_impl_diff = pd.Series(
        impl["mrkt_home_impl"] - impl["mrkt_away_impl"],
        name="mrkt_home_away_impl_diff",
    )

    phase4 = _season_4phase(date, season, limits)
    phase3 = _season_3phase(phase4)

    if stadiums is None:
        blank = pd.Series(np.nan, index=raw.index)
        home_is_promoted = away_is_promoted = blank
        travel = blank.rename("travel_distance_km")
    else:
        _check_missing(all_teams, stadiums, league, season)
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
        phase4.rename("season_4phase"),
        phase3.rename("season_3phase"),
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
        mrkt_favrt_impl,
        mrkt_undrd_impl,
        mrkt_home_away_impl_diff,
    ], axis=1)


def _attach_elo(df: pd.DataFrame, group: str) -> pd.DataFrame:
    """Add home_elo / away_elo: the Club Elo rating of each team as of the match date.

    For a match on date D, the rating whose window contains D (From <= D <= To) is the
    PRE-match rating (Club Elo dates the post-match update to D+1). We resolve it with a
    point-in-time merge_asof (backward on From), so no future information leaks in.
    Falls back to NaN if the Elo data or the group's team map is absent, or a team has no
    rating for that date."""
    df = df.copy()
    hist_path = _ELO_DIR / "clubelo_history.csv"
    tmap_path = _ELO_DIR / group / "team_map.csv"
    if not hist_path.exists() or not tmap_path.exists():
        df["home_elo"] = np.nan
        df["away_elo"] = np.nan
        return df

    name_to_clubelo = pd.read_csv(tmap_path).set_index("team")["clubelo"].to_dict()
    hist = pd.read_csv(hist_path, usecols=["clubelo", "Elo", "From"])
    hist["Elo"] = pd.to_numeric(hist["Elo"], errors="coerce")
    hist["From"] = pd.to_datetime(hist["From"])
    hist = hist.dropna(subset=["From"]).sort_values("From").reset_index(drop=True)

    match_date = pd.to_datetime(df["date"])
    for side in ("home", "away"):
        left = pd.DataFrame({
            "_row": range(len(df)),
            "date": match_date.values,
            "clubelo": df[f"{side}_team"].map(name_to_clubelo).values,
        }).sort_values("date")
        merged = pd.merge_asof(left, hist, left_on="date", right_on="From",
                               by="clubelo", direction="backward")
        df[f"{side}_elo"] = merged.sort_values("_row")["Elo"].values
    return df


def build_match_info(group: str = "major") -> pd.DataFrame:
    cols = ["match_id"] + _columns()
    limits = _load_phase_limits()
    features_dir = _features_dir(group)
    frames = []
    for path in sorted(_matches_dir(group).glob("*.csv")):
        league, season = path.stem.rsplit("_", 1)
        stadiums = _load_stadiums(league, season)
        frames.append(_load_file(path, stadiums, limits))

    base = _attach_elo(pd.concat(frames, ignore_index=True), group)
    df = round_floats(base[cols])

    features_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(features_dir / "match_info.csv", index=False)
    df.to_parquet(features_dir / "match_info.parquet", index=False)

    return df
