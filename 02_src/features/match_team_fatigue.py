"""match_team_fatigue — schedule-density features on each team's full fixture calendar.

The calendar unions the domestic league (01_matches), domestic cups and super cups
(07_domestic), the UEFA club competitions (06_europe) and the FIFA club competitions
(08_intl). Non-league fixtures exist only as history — the windows count them, but they
never become rows of the output table, which stays on match_id grain.

Fixtures are described on two orthogonal axes — where they were played and which
competition they belong to (see _VENUES / _COMPETITIONS) — each of which partitions the
calendar, so either family sums to the total.

The layout mirrors match_team_stats: a long frame with one row per (team, fixture),
statistics computed within (league, season, team) ordered by kick-off, then a pivot back
to match grain via is_home.
"""

import itertools
import json
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

# The gap columns are capped: past a point the gap is a winter/international break or a
# postponement, and the number stops describing recovery. The second gap spans two
# fixtures, so its cap sits proportionally higher.
_LAST_CAP_HOURS = 200
_2ND_LAST_CAP_HOURS = 500

_GAP_COLS = ["hours_since_last_match", "hours_since_2nd_last_match"]

# Competition stream per file-name prefix. "other" is everything that is neither the
# domestic league nor a UEFA club competition: domestic cups, domestic and european super
# cups, and the FIFA club competitions. Splitting it further gave cells of a few dozen rows.
_STREAMS = {
    "fa": "other", "efl": "other", "cdr": "other", "ci": "other", "dfb": "other",
    "cdf": "other", "cs": "other", "sce": "other", "sci": "other", "dfl": "other",
    "tdc": "other", "usc": "other", "cwc": "other", "ic": "other",
    "cl": "europe", "clq": "europe", "el": "europe", "elq": "europe",
    "cfl": "europe", "cflq": "europe",
}

# Domestic competitions stay inside the country, with these exceptions: the spanish and
# italian super cups are played in saudi arabia and the french one tours (tel aviv, doha,
# kuwait city). Everything else on 07_domestic — including the neutral-venue finals at
# Wembley, La Cartuja, the Olimpico, Berlin and the Stade de France — is home soil.
_ABROAD_VENUES = {
    "King Fahd International Stadium", "Al Awwal Park Stadium",
    "Alinma Stadium", "Al Inma Stadium",
    "Bloomfield Stadium", "Stadium 974", "Jaber Al-Ahmad International Stadium",
}

_FBREF_DIRS = ["06_europe", "07_domestic", "08_intl"]

# Fixtures are described on two orthogonal axes, each partitioning the calendar, so either
# family sums to the total. `domestic` is an away fixture in the team's own country or a
# neutral venue inside it; `abroad` is a trip out of it — nine in ten of those are european
# away legs and the rest are super cups and fifa competitions, so the column reads as
# "played a serious match out of the country". The full venue x stream grid is not carried
# as features (several cells hold a few dozen rows) but the calendar keeps it in `cell`.
_VENUES = ["home", "domestic", "abroad"]
_COMPETITIONS = ["league", "europe", "other"]

# indicator columns aggregated over every window and every decay; the names are the stems
_IND_COLS = (["games"]
             + [f"{venue}_games" for venue in _VENUES]
             + [f"{comp}_games" for comp in _COMPETITIONS])


def _matches_dir(group: str) -> Path:
    return _RAW / "01_matches" / group


def _features_dir(group: str) -> Path:
    return _DATA / "02_features" / group


def _schema() -> dict:
    with open(_YAML_PATH) as f:
        return yaml.safe_load(f)


def _dim(schema: dict, column: str, dim: str) -> list:
    """Window widths and decay constants come from the contract, so changing them there
    changes the builder."""
    return schema["columns"][column]["dims"][dim]


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
_COUNTRY_PREFIX = re.compile(r"^([a-z]{2,3}) ")
_COUNTRY_SUFFIX = re.compile(r" ([a-z]{2,3})$")
_NEUTRAL = " (Neutral Site)"


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
                    stream="league", is_neutral=False, is_abroad=False)
        frames.append(pd.DataFrame({**base, "team": raw["HomeTeam"].values, "is_home": True}))
        frames.append(pd.DataFrame({**base, "team": raw["AwayTeam"].values, "is_home": False}))
    return pd.concat(frames, ignore_index=True)


def _fbref_rows(membership: dict) -> pd.DataFrame:
    """Cup, european and intercontinental fixtures. Rows carry no match_id — they are
    history for the windows only. Teams outside the group's leagues drop out here.

    Whether a fixture was played abroad comes from two sources. On 06_europe / 08_intl the
    team names carry country codes, so an away leg is abroad unless both clubs share a
    country; a neutral venue is treated as abroad for both sides (the alternative would be
    a european final on a finalist's home soil, which does not occur in this data). On
    07_domestic there are no codes, so the venue name decides.
    """
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

            venue = raw["Venue"].astype(str)
            neutral = venue.str.endswith(_NEUTRAL)

            if folder == "07_domestic":
                home = raw["Home"].map(lambda t: domestic_map.get(t, t))
                away = raw["Away"].map(lambda t: domestic_map.get(t, t))
                abroad_home = venue.str.removesuffix(_NEUTRAL).isin(_ABROAD_VENUES)
                abroad_away = abroad_home
            else:
                home = raw["Home"].map(lambda t: europe_map.get(_strip_country(t)))
                away = raw["Away"].map(lambda t: europe_map.get(_strip_country(t)))
                home_country = raw["Home"].astype(str).str.extract(_COUNTRY_SUFFIX)[0]
                away_country = raw["Away"].astype(str).str.extract(_COUNTRY_PREFIX)[0]
                abroad_home = neutral
                abroad_away = neutral | (home_country != away_country)

            ts = _fbref_timestamp(raw["Date"], raw["Time"])
            base = dict(match_id=None, season=season, stream=_STREAMS[prefix],
                        ts=ts.values, is_neutral=neutral.values)
            frames.append(pd.DataFrame({**base, "team": home.values, "is_home": True,
                                        "is_abroad": abroad_home.values}))
            frames.append(pd.DataFrame({**base, "team": away.values, "is_home": False,
                                        "is_abroad": abroad_away.values}))

    rows = pd.concat(frames, ignore_index=True)
    rows["league"] = [membership.get((s, t)) for s, t in zip(rows["season"], rows["team"])]
    return rows[rows["league"].notna()].reset_index(drop=True)


def _derive_flags(calendar: pd.DataFrame) -> pd.DataFrame:
    """A neutral-venue fixture is nobody's home match."""
    calendar = calendar.copy()
    calendar["ts"] = pd.to_datetime(calendar["ts"])
    calendar.loc[calendar["is_neutral"], "is_home"] = False
    calendar["is_away"] = ~calendar["is_home"] & ~calendar["is_neutral"]
    calendar["is_europe"] = calendar["stream"] == "europe"
    calendar["venue"] = np.where(calendar["is_home"], "home",
                                 np.where(calendar["is_abroad"], "abroad", "domestic"))
    # the full grid is not a feature, but it is the natural thing to look at in the calendar
    calendar["cell"] = calendar["stream"] + "_" + calendar["venue"]
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
    for venue in _VENUES:
        calendar[f"{venue}_games"] = (calendar["venue"] == venue).astype(float)
    for comp in _COMPETITIONS:
        calendar[f"{comp}_games"] = (calendar["stream"] == comp).astype(float)
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


def _decayed_loads(calendar: pd.DataFrame, groups: dict, tau: float) -> np.ndarray:
    """Sum every indicator over all earlier fixtures of the season, each weighted by
    exp(-age_in_days / tau).

    An exponential kernel is memoryless, so the load only needs the previous value and the
    gap: load_i = (load_{i-1} + value_{i-1}) * exp(-gap / tau). No window, no cut-off.
    """
    values = calendar[_IND_COLS].to_numpy(dtype=float)
    ts = calendar["ts"].to_numpy()
    out = np.zeros_like(values)

    for idx in groups.values():
        idx = np.sort(idx)
        decay = np.exp(-(np.diff(ts[idx]) / np.timedelta64(1, "D")) / tau)
        carried = np.zeros(values.shape[1])
        for k in range(1, len(idx)):
            carried = (carried + values[idx[k - 1]]) * decay[k - 1]
            out[idx[k]] = carried
    return out


def _gaussian_loads(calendar: pd.DataFrame, groups: dict, tau: float) -> np.ndarray:
    """Same as _decayed_loads but with a gaussian kernel, exp(-(age / tau)**2).

    Squaring the age costs the memorylessness the exponential has, so there is no
    recursion to lean on and every earlier fixture is weighted explicitly. Teams have a
    few dozen fixtures a season, so the quadratic work is irrelevant.
    """
    values = calendar[_IND_COLS].to_numpy(dtype=float)
    ts = calendar["ts"].to_numpy()
    out = np.zeros_like(values)

    for idx in groups.values():
        idx = np.sort(idx)
        ages = (ts[idx][:, None] - ts[idx][None, :]) / np.timedelta64(1, "D")
        weights = np.where(ages > 0, np.exp(-((ages / tau) ** 2)), 0.0)
        out[idx] = weights @ values[idx]
    return out


def _add_stats(calendar: pd.DataFrame, windows: list[int],
               taus_exp: list[float], taus_gauss: list[float]) -> pd.DataFrame:
    calendar = _add_indicators(calendar)
    keys = ["league", "season", "team"]
    g = calendar.groupby(keys, sort=False)

    hours = lambda delta: delta.dt.total_seconds() / 3600.0
    calendar["hours_since_last_match"] = hours(g["ts"].diff()).clip(upper=_LAST_CAP_HOURS)
    calendar["hours_since_2nd_last_match"] = (
        hours(calendar["ts"] - g["ts"].shift(2)).clip(upper=_2ND_LAST_CAP_HOURS)
    )
    calendar["last_match_is_away"] = g["is_away"].shift(1)
    calendar["last_match_is_europe"] = g["is_europe"].shift(1)
    calendar["last_match_is_abroad"] = g["is_abroad"].shift(1)

    groups = g.indices
    new = {}
    for x in windows:
        sums = _window_sums(calendar, groups, x)
        new |= {f"{c}_in_{x}d": sums[:, i] for i, c in enumerate(_IND_COLS)}
    for tau in taus_exp:
        loads = _decayed_loads(calendar, groups, tau)
        new |= {f"{c}_load_{tau}d": loads[:, i] for i, c in enumerate(_IND_COLS)}
    for tau in taus_gauss:
        gauss = _gaussian_loads(calendar, groups, tau)
        new |= {f"{c}_load_gauss_{tau}d": gauss[:, i] for i, c in enumerate(_IND_COLS)}

    return pd.concat([calendar, pd.DataFrame(new, index=calendar.index)], axis=1)


def _attach_gap_cats(df: pd.DataFrame, group: str) -> pd.DataFrame:
    """Categorize the two gap columns against their pooled per-match distribution.

    cat2q: low/high either side of the median. cat3q: low/medium/high tertiles (p33/p67).
    The gaps are per-match quantities, so the boundaries follow {side}_elo_cat3q rather
    than the season-stat categories: they come from all hmt + awt values in the group
    (global, all seasons) and are recorded in thresholds.json. Both caps sit above p67, so
    the capped tail lands wholly in `high`. None where the gap is NaN."""
    df = df.copy()
    thresholds_path = _features_dir(group) / "thresholds.json"
    thresholds = json.loads(thresholds_path.read_text()) if thresholds_path.exists() else {}

    for stem in _GAP_COLS:
        pooled = pd.concat([df[f"hmt_{stem}"], df[f"awt_{stem}"]]).dropna()

        if pooled.empty:
            for side in ("hmt", "awt"):
                df[f"{side}_{stem}_cat2q"] = None
                df[f"{side}_{stem}_cat3q"] = None
            continue

        p50 = round(float(pooled.quantile(1 / 2)), 3)
        p33 = round(float(pooled.quantile(1 / 3)), 3)
        p67 = round(float(pooled.quantile(2 / 3)), 3)
        thresholds.update({f"{stem}_p50": p50, f"{stem}_p33": p33, f"{stem}_p67": p67})

        for side in ("hmt", "awt"):
            gap = df[f"{side}_{stem}"]
            df[f"{side}_{stem}_cat2q"] = np.where(
                gap.isna(), None, np.where(gap <= p50, "low", "high"))
            df[f"{side}_{stem}_cat3q"] = np.where(
                gap.isna(), None,
                np.where(gap <= p33, "low", np.where(gap <= p67, "medium", "high")))

    thresholds_path.parent.mkdir(parents=True, exist_ok=True)
    thresholds_path.write_text(json.dumps(thresholds, indent=2))
    return df


def build_match_team_fatigue(group: str = "major") -> pd.DataFrame:
    schema = _schema()
    windows = _dim(schema, "{side}_games_in_{x}d", "x")
    taus_exp = _dim(schema, "{side}_games_load_{tau}d", "tau")
    taus_gauss = _dim(schema, "{side}_games_load_gauss_{tau}d", "tau")
    calendar = _add_stats(_build_calendar(group), windows, taus_exp, taus_gauss)

    stat_cols = (["hours_since_last_match", "hours_since_2nd_last_match",
                  "last_match_is_away", "last_match_is_europe",
                  "last_match_is_abroad"]
                 + [f"{c}_in_{x}d" for x in windows for c in _IND_COLS]
                 + [f"{c}_load_{t}d" for t in taus_exp for c in _IND_COLS]
                 + [f"{c}_load_gauss_{t}d" for t in taus_gauss for c in _IND_COLS])
    played = calendar[calendar["match_id"].notna()]

    home = (played[played["is_home"]][["match_id"] + stat_cols]
            .rename(columns={c: f"hmt_{c}" for c in stat_cols}))
    away = (played[~played["is_home"]][["match_id"] + stat_cols]
            .rename(columns={c: f"awt_{c}" for c in stat_cols}))
    df = _attach_gap_cats(home.merge(away, on="match_id"), group)
    df = round_floats(df[["match_id"] + _columns(schema)])

    features_dir = _features_dir(group)
    features_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(features_dir / "match_team_fatigue.csv", index=False)
    df.to_parquet(features_dir / "match_team_fatigue.parquet", index=False)

    calendar.to_parquet(features_dir / "team_calendar.parquet", index=False)

    return df
