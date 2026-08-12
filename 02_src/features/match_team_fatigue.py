"""match_team_fatigue — schedule-density features on each team's full fixture calendar.

The calendar unions four streams: the domestic league (01_matches), domestic cups and
domestic super cups (07_domestic), european cups (06_europe) and the fifa club
competitions (08_intl). Non-league fixtures exist only as history — the windows count
them, but they never become rows of the output table, which stays on match_id grain.

The layout mirrors match_team_stats: a long frame with one row per (team, fixture),
statistics computed within (league, season, team) ordered by kick-off, then a pivot back
to match grain via is_home.
"""

import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from features.utils import round_floats


_ROOT = Path(__file__).parents[2]
_DATA = _ROOT / "01_data"
_RAW = _DATA / "01_raw"
_YAML_PATH = _ROOT / "06_docs" / "data.yaml"

# rest_hours is capped: past a week the gap is a winter/international break or a
# postponement, and the number stops describing recovery.
_REST_CAP_HOURS = 200

# competition stream per file-name prefix. domestic and european super cups sit together
# in `other` with the fifa competitions; european qualifying counts as `europe`.
_STREAMS = {
    "fa": "cup",  "efl": "cup",  "cdr": "cup", "ci": "cup",  "dfb": "cup", "cdf": "cup",
    "cs": "other", "sce": "other", "sci": "other", "dfl": "other", "tdc": "other",
    "usc": "other", "cwc": "other", "ic": "other",
    "cl": "europe", "clq": "europe", "el": "europe", "elq": "europe",
    "cfl": "europe", "cflq": "europe",
}

_FBREF_DIRS = ["06_europe", "07_domestic", "08_intl"]
_STREAM_ORDER = ["league", "cup", "europe", "other"]

# indicator columns summed over every window; the names double as the column stems
_IND_COLS = (
    ["games"]
    + [f"{s}_games" for s in _STREAM_ORDER]
    + ["away_games"]
    + [f"away_{s}_games" for s in _STREAM_ORDER]
    + ["awayn_games"]
)


def _matches_dir(group: str) -> Path:
    return _RAW / "01_matches" / group


def _features_dir(group: str) -> Path:
    return _DATA / "02_features" / group


def _schema() -> dict:
    with open(_YAML_PATH) as f:
        return yaml.safe_load(f)


def _windows(schema: dict) -> list[int]:
    """Window widths come from the contract, so pruning them there prunes the builder."""
    return schema["columns"]["{side}_games_in_{x}d"]["dims"]["x"]


def _columns(schema: dict) -> list[str]:
    result = []
    for name, meta in schema["columns"].items():
        if not meta or meta.get("table") != "match_team_fatigue":
            continue
        if "dims" in meta:
            keys = list(meta["dims"])
            for combo in itertools.product(*(meta["dims"][k] for k in keys)):
                result.append(name.format(**dict(zip(keys, combo))))
        else:
            result.append(name)
    return result


def _parse_dates(series: pd.Series) -> pd.Series:
    for fmt in ("%d/%m/%y", "%d/%m/%Y"):
        try:
            return pd.to_datetime(series, format=fmt)
        except ValueError:
            continue
    return pd.to_datetime(series, dayfirst=True)


_PAREN_TIME = re.compile(r"\((\d{1,2}:\d{2})\)")
_COUNTRY_PREFIX = re.compile(r"^[a-z]{2,3} ")
_COUNTRY_SUFFIX = re.compile(r" [a-z]{2,3}$")


def _strip_country(name: str) -> str:
    """FBref tags international fixtures with a country code — suffixed on the home team
    (`PSV nl`), prefixed on the away team (`be Union SG`). Codes are two or three letters
    (`eng`, `wls`, `nir`)."""
    s = str(name).strip()
    return _COUNTRY_SUFFIX.sub("", _COUNTRY_PREFIX.sub("", s))


def _fbref_timestamp(date: pd.Series, time: pd.Series) -> pd.Series:
    """FBref prints the venue-local kick-off with central european time in brackets when
    the two differ. Take the bracketed value where present; otherwise the venue is already
    on central european time."""
    time = time.astype("string").fillna("")
    cet = time.str.extract(_PAREN_TIME)[0]
    local = time.str.split(" ").str[0]
    return pd.to_datetime(date) + pd.to_timedelta(cet.fillna(local) + ":00")


def _league_membership(group: str) -> dict[tuple[str, str], str]:
    """(season, team) -> league. A cup or european fixture is filed under the league its
    team plays in that season, which is what the calendar groups by."""
    membership = {}
    for path in sorted(_matches_dir(group).glob("*.csv")):
        league, season = path.stem.removesuffix("_matches").rsplit("_", 1)
        raw = pd.read_csv(path, usecols=["HomeTeam", "AwayTeam"])
        for team in set(raw["HomeTeam"]) | set(raw["AwayTeam"]):
            membership[(season, team)] = league
    return membership


def _name_maps() -> tuple[dict, dict]:
    """FBref names -> football-data names. The european map is a full membership list;
    the domestic map holds only the names that differ, so lookups fall back to identity."""
    europe = pd.read_csv(_RAW / "06_europe" / "team_map.csv")
    domestic = pd.read_csv(_RAW / "07_domestic" / "team_map.csv")
    return (dict(zip(europe["europe"], europe["team"])),
            dict(zip(domestic["domestic"], domestic["team"])))


def _league_rows(group: str) -> pd.DataFrame:
    """League fixtures, two rows per match, carrying the match_id the output joins on.
    football-data prints uk kick-offs; central european time is a constant hour ahead
    (both zones switch on the same dates)."""
    from features.match_info import _match_id

    frames = []
    for path in sorted(_matches_dir(group).glob("*.csv")):
        league, season = path.stem.removesuffix("_matches").rsplit("_", 1)
        raw = pd.read_csv(path)
        ts = (_parse_dates(raw["Date"])
              + pd.to_timedelta(raw["Time"].astype(str) + ":00")
              + pd.Timedelta(hours=1))
        ids = [_match_id(league, season, h, a) for h, a in zip(raw["HomeTeam"], raw["AwayTeam"])]
        base = dict(match_id=ids, league=league, season=season, ts=ts.values,
                    stream="league", is_neutral=False)
        frames.append(pd.DataFrame({**base, "team": raw["HomeTeam"].values, "is_home": True}))
        frames.append(pd.DataFrame({**base, "team": raw["AwayTeam"].values, "is_home": False}))
    return pd.concat(frames, ignore_index=True)


def _fbref_rows(membership: dict) -> pd.DataFrame:
    """Cup, european and intercontinental fixtures. Rows carry no match_id — they are
    history for the windows only. Teams outside the group's leagues drop out here."""
    europe_map, domestic_map = _name_maps()
    frames = []

    for folder in _FBREF_DIRS:
        for path in sorted((_RAW / folder).glob("*.csv")):
            prefix, season = path.stem.rsplit("_", 1)
            if prefix not in _STREAMS:
                continue
            raw = pd.read_csv(path, encoding="utf-8")
            raw = raw[raw["Home"].notna() & raw["Score"].notna()]
            if raw.empty:
                continue

            if folder == "07_domestic":
                home = raw["Home"].map(lambda t: domestic_map.get(t, t))
                away = raw["Away"].map(lambda t: domestic_map.get(t, t))
            else:
                home = raw["Home"].map(lambda t: europe_map.get(_strip_country(t)))
                away = raw["Away"].map(lambda t: europe_map.get(_strip_country(t)))

            ts = _fbref_timestamp(raw["Date"], raw["Time"])
            neutral = raw["Venue"].astype(str).str.contains("Neutral Site")
            base = dict(match_id=None, season=season, stream=_STREAMS[prefix],
                        ts=ts.values, is_neutral=neutral.values)
            frames.append(pd.DataFrame({**base, "team": home.values, "is_home": True}))
            frames.append(pd.DataFrame({**base, "team": away.values, "is_home": False}))

    rows = pd.concat(frames, ignore_index=True)
    rows["league"] = [membership.get((s, t)) for s, t in zip(rows["season"], rows["team"])]
    return rows[rows["league"].notna()].reset_index(drop=True)


def _derive_flags(calendar: pd.DataFrame) -> pd.DataFrame:
    """A neutral-venue fixture is nobody's home match: it counts as away for `awayn`
    but not for `away`."""
    calendar = calendar.copy()
    calendar["ts"] = pd.to_datetime(calendar["ts"])
    calendar.loc[calendar["is_neutral"], "is_home"] = False
    calendar["is_away"] = ~calendar["is_home"] & ~calendar["is_neutral"]
    calendar["is_europe"] = calendar["stream"] == "europe"
    return calendar


def _build_calendar(group: str) -> pd.DataFrame:
    membership = _league_membership(group)
    calendar = _derive_flags(pd.concat(
        [_league_rows(group), _fbref_rows(membership)], ignore_index=True
    ))
    return calendar.sort_values(["league", "season", "team", "ts"]).reset_index(drop=True)


def _add_indicators(calendar: pd.DataFrame) -> pd.DataFrame:
    calendar = calendar.copy()
    calendar["games"] = 1.0
    calendar["away_games"] = calendar["is_away"].astype(float)
    calendar["awayn_games"] = (~calendar["is_home"]).astype(float)
    for stream in _STREAM_ORDER:
        in_stream = calendar["stream"] == stream
        calendar[f"{stream}_games"] = in_stream.astype(float)
        calendar[f"away_{stream}_games"] = (in_stream & calendar["is_away"]).astype(float)
    return calendar


def _window_sums(calendar: pd.DataFrame, groups: dict, x: int) -> np.ndarray:
    """Sum every indicator over the half-open window [ts - x days, ts), per team.

    Done with prefix sums and searchsorted rather than groupby().rolling(): the latter
    returns its rows grouped, not in frame order, which silently misaligns the result.
    The window is half-open so the current fixture is excluded, and an empty window
    sums to 0.
    """
    values = calendar[_IND_COLS].to_numpy(dtype=float)
    ts = calendar["ts"].to_numpy()
    span = np.timedelta64(x, "D")
    out = np.zeros_like(values)

    for idx in groups.values():
        idx = np.sort(idx)
        prefix = np.vstack([np.zeros(len(_IND_COLS)), values[idx].cumsum(axis=0)])
        lo = np.searchsorted(ts[idx], ts[idx] - span, side="left")
        out[idx] = prefix[np.arange(len(idx))] - prefix[lo]
    return out


def _add_stats(calendar: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    calendar = _add_indicators(calendar)
    keys = ["league", "season", "team"]
    g = calendar.groupby(keys, sort=False)

    gap = g["ts"].diff().dt.total_seconds() / 3600.0
    calendar["rest_hours"] = gap.clip(upper=_REST_CAP_HOURS)
    calendar["last_match_is_away"] = g["is_away"].shift(1)
    calendar["last_match_is_europe"] = g["is_europe"].shift(1)

    groups = g.indices
    for x in windows:
        sums = _window_sums(calendar, groups, x)
        for i, col in enumerate(_IND_COLS):
            calendar[f"{col}_in_{x}d"] = sums[:, i]

    return calendar


def build_match_team_fatigue(group: str = "major") -> pd.DataFrame:
    schema = _schema()
    windows = _windows(schema)
    calendar = _add_stats(_build_calendar(group), windows)

    stat_cols = (["rest_hours", "last_match_is_away", "last_match_is_europe"]
                 + [f"{c}_in_{x}d" for x in windows for c in _IND_COLS])
    played = calendar[calendar["match_id"].notna()]

    home = (played[played["is_home"]][["match_id"] + stat_cols]
            .rename(columns={c: f"hmt_{c}" for c in stat_cols}))
    away = (played[~played["is_home"]][["match_id"] + stat_cols]
            .rename(columns={c: f"awt_{c}" for c in stat_cols}))
    df = home.merge(away, on="match_id")
    df = round_floats(df[["match_id"] + _columns(schema)])

    features_dir = _features_dir(group)
    features_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(features_dir / "match_team_fatigue.csv", index=False)
    df.to_parquet(features_dir / "match_team_fatigue.parquet", index=False)

    calendar.to_parquet(features_dir / "team_calendar.parquet", index=False)

    return df
