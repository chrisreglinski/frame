import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from features.utils import round_floats


_ROOT = Path(__file__).parents[2]
_RAW_DIR = _ROOT / "01_data" / "01_raw" / "01_matches"
_FEATURES_DIR = _ROOT / "01_data" / "02_features"
_YAML_PATH = _ROOT / "06_docs" / "data.yaml"
_THRESHOLDS_PATH = _FEATURES_DIR / "thresholds.json"

_LEAGUES = ["england", "spain", "italy", "germany", "france"]
_SEASONS = ["2223", "2324", "2425", "2526"]

_WINDOWS = {"season": None, "rolling6": 6, "rolling8": 8}

_LOCALITY_METRICS = ["goals", "shots", "shots_on_target"]
_LOCALITY_SCALAR_METRICS = ["points", "goals_diff", "goals_total"]
_LOCALITY_STAT_COLS = [
    f"{loc}_season_{m}_{k}_avg"
    for loc in ["home", "away"]
    for m in _LOCALITY_METRICS
    for k in ["for", "agst"]
] + [
    f"{loc}_season_{m}_avg"
    for loc in ["home", "away"]
    for m in _LOCALITY_SCALAR_METRICS
]

_TEAM_SEASON_COLS = [
    "match_id", "league", "season", "date", "team", "opponent", "is_home",
    "goals_for", "goals_agst", "points", "goals_diff", "goals_total",
]


def _make_stat_cols() -> list[str]:
    cols = ["game_number"]
    for w in _WINDOWS:
        cols += [f"{w}_{m}_{k}_avg" for m in ["goals", "shots", "shots_on_target", "corners", "yellow"] for k in ["for", "agst"]]
        cols += [f"{w}_{m}_{v}" for m in ["points", "goals_diff", "goals_total"] for v in ["avg", "std"]]
        cols += [f"{w}_{r}_ratio" for r in ["wins", "draws", "losses"]]
        cols += [f"{w}_goals_total_le{x}_ratio" for x in [0, 1, 2, 3, 4]]
        cols += [f"{w}_goals_{kind}_le{x}_ratio" for kind in ["for", "agst"] for x in [0, 1, 2]]
        cols += [f"{w}_shots_on_target_{k}_ratio" for k in ["for", "agst"]]
        cols += [f"{w}_red_for_avg"]
        cols += [f"{w}_impl_{o}_avg" for o in ["win", "draw", "loss"]]
        cols += [f"{w}_impl_points_avg"]
        cols += [f"{w}_profit_{o}" for o in ["win", "draw", "loss"]]
    cols += ["red_last_match"]
    return cols


_STAT_COLS = _make_stat_cols()


def _columns() -> list[str]:
    with open(_YAML_PATH) as f:
        schema = yaml.safe_load(f)
    result = []
    for name, meta in schema["columns"].items():
        if not meta:
            continue
        table = meta.get("table")
        if "team_stats" not in (table if isinstance(table, list) else [table]):
            continue
        if "dims" in meta:
            keys = list(meta["dims"].keys())
            values = [list(v) if isinstance(v, list) else [v] for v in (meta["dims"][k] for k in keys)]
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


def _build_long(raw: pd.DataFrame, league: str, season: str) -> pd.DataFrame:
    date = _parse_dates(raw["Date"])
    ids = [_match_id(league, season, h, a) for h, a in zip(raw["HomeTeam"], raw["AwayTeam"])]
    base = dict(match_id=ids, league=league, season=season, date=date.values)

    avg_h, avg_d, avg_a = raw["AvgH"].values, raw["AvgD"].values, raw["AvgA"].values

    home = pd.DataFrame({
        **base, "is_home": True, "team": raw["HomeTeam"].values, "opponent": raw["AwayTeam"].values,
        "goals_for": raw["FTHG"].values,    "goals_agst": raw["FTAG"].values,
        "shots_for": raw["HS"].values,       "shots_agst": raw["AS"].values,
        "shots_on_target_for": raw["HST"].values, "shots_on_target_agst": raw["AST"].values,
        "corners_for": raw["HC"].values,     "corners_agst": raw["AC"].values,
        "yellow_for": raw["HY"].values,      "yellow_agst": raw["AY"].values,
        "red_for": raw["HR"].values,
        "points": raw["FTR"].map({"H": 3, "D": 1, "A": 0}).values,
        "is_win":  (raw["FTR"] == "H").astype(float).values,
        "is_draw": (raw["FTR"] == "D").astype(float).values,
        "is_loss": (raw["FTR"] == "A").astype(float).values,
        "impl_win": 1.0 / avg_h, "impl_draw": 1.0 / avg_d, "impl_loss": 1.0 / avg_a,
    })

    away = pd.DataFrame({
        **base, "is_home": False, "team": raw["AwayTeam"].values, "opponent": raw["HomeTeam"].values,
        "goals_for": raw["FTAG"].values,    "goals_agst": raw["FTHG"].values,
        "shots_for": raw["AS"].values,       "shots_agst": raw["HS"].values,
        "shots_on_target_for": raw["AST"].values, "shots_on_target_agst": raw["HST"].values,
        "corners_for": raw["AC"].values,     "corners_agst": raw["HC"].values,
        "yellow_for": raw["AY"].values,      "yellow_agst": raw["HY"].values,
        "red_for": raw["AR"].values,
        "points": raw["FTR"].map({"A": 3, "D": 1, "H": 0}).values,
        "is_win":  (raw["FTR"] == "A").astype(float).values,
        "is_draw": (raw["FTR"] == "D").astype(float).values,
        "is_loss": (raw["FTR"] == "H").astype(float).values,
        "impl_win": 1.0 / avg_a, "impl_draw": 1.0 / avg_d, "impl_loss": 1.0 / avg_h,
    })

    long = pd.concat([home, away], ignore_index=True)
    long["goals_diff"] = long["goals_for"] - long["goals_agst"]
    long["goals_total"] = long["goals_for"] + long["goals_agst"]
    return long


def _agg_mean(g, col, w):
    if w is None:
        return g[col].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    return g[col].transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())


def _agg_std(g, col, w):
    if w is None:
        return g[col].transform(lambda x: x.shift(1).expanding(min_periods=1).std(ddof=0))
    return g[col].transform(lambda x: x.shift(1).rolling(w, min_periods=1).std(ddof=0))


def _agg_sum(g, col, w):
    if w is None:
        return g[col].transform(lambda x: x.shift(1).expanding(min_periods=1).sum())
    return g[col].transform(lambda x: x.shift(1).rolling(w, min_periods=1).sum())


def _add_stats(long: pd.DataFrame) -> pd.DataFrame:
    long = long.sort_values(["league", "season", "team", "date"]).reset_index(drop=True)
    grp = ["league", "season", "team"]
    g = long.groupby(grp, sort=False)

    long["game_number"] = g.cumcount() + 1

    # indicators computed once, reused across all windows
    for x in [0, 1, 2, 3, 4]:
        long[f"_ind_gt_le{x}"] = (long["goals_total"] <= x).astype(float)
    for kind in ["for", "agst"]:
        for x in [0, 1, 2]:
            long[f"_ind_g{kind[0]}le{x}"] = (long[f"goals_{kind}"] <= x).astype(float)

    for wname, wsize in _WINDOWS.items():
        for m in ["goals", "shots", "shots_on_target", "corners", "yellow"]:
            for k in ["for", "agst"]:
                long[f"{wname}_{m}_{k}_avg"] = _agg_mean(g, f"{m}_{k}", wsize)

        for m in ["points", "goals_diff", "goals_total"]:
            long[f"{wname}_{m}_avg"] = _agg_mean(g, m, wsize)
            long[f"{wname}_{m}_std"] = _agg_std(g, m, wsize)

        for r, c in [("wins", "is_win"), ("draws", "is_draw"), ("losses", "is_loss")]:
            long[f"{wname}_{r}_ratio"] = _agg_mean(g, c, wsize)

        for x in [0, 1, 2, 3, 4]:
            long[f"{wname}_goals_total_le{x}_ratio"] = _agg_mean(g, f"_ind_gt_le{x}", wsize)
        for kind in ["for", "agst"]:
            for x in [0, 1, 2]:
                long[f"{wname}_goals_{kind}_le{x}_ratio"] = _agg_mean(g, f"_ind_g{kind[0]}le{x}", wsize)

        for k in ["for", "agst"]:
            long[f"{wname}_shots_on_target_{k}_ratio"] = (
                _agg_sum(g, f"shots_on_target_{k}", wsize) / _agg_sum(g, f"shots_{k}", wsize)
            )

        long[f"{wname}_red_for_avg"] = _agg_mean(g, "red_for", wsize)

        for o in ["win", "draw", "loss"]:
            long[f"{wname}_impl_{o}_avg"] = _agg_mean(g, f"impl_{o}", wsize)
        long[f"{wname}_impl_points_avg"] = long[f"{wname}_impl_win_avg"] * 3 + long[f"{wname}_impl_draw_avg"]
        for o, r in [("win", "wins"), ("draw", "draws"), ("loss", "losses")]:
            long[f"{wname}_profit_{o}"] = long[f"{wname}_{r}_ratio"] - long[f"{wname}_impl_{o}_avg"]

    red_shifted = g["red_for"].transform(lambda x: x.shift(1))
    long["red_last_match"] = (
        (red_shifted > 0)
        .where(red_shifted.notna(), other=pd.NA)
        .astype(pd.BooleanDtype())
    )

    # home-only and away-only season averages
    for m in _LOCALITY_METRICS:
        for k in ["for", "agst"]:
            long[f"_home_{m}_{k}"] = long[f"{m}_{k}"].where(long["is_home"])
            long[f"_away_{m}_{k}"] = long[f"{m}_{k}"].where(~long["is_home"])
            long[f"home_season_{m}_{k}_avg"] = _agg_mean(g, f"_home_{m}_{k}", None)
            long[f"away_season_{m}_{k}_avg"] = _agg_mean(g, f"_away_{m}_{k}", None)
    for m in _LOCALITY_SCALAR_METRICS:
        long[f"_home_{m}"] = long[m].where(long["is_home"])
        long[f"_away_{m}"] = long[m].where(~long["is_home"])
        long[f"home_season_{m}_avg"] = _agg_mean(g, f"_home_{m}", None)
        long[f"away_season_{m}_avg"] = _agg_mean(g, f"_away_{m}", None)

    # mask rolling windows for matches with insufficient history
    for wname, wsize in _WINDOWS.items():
        if wsize is None:
            continue
        window_cols = [c for c in long.columns if c.startswith(f"{wname}_")]
        long.loc[long["game_number"] <= wsize, window_cols] = pd.NA

    return long.copy()


def _add_summary_stubs(long: pd.DataFrame) -> pd.DataFrame:
    """Append one NaN stub row per (league, season, team) dated one day after the last match.
    When _add_stats processes these with shift(1), the stub's season/rolling stats
    naturally include all N real matches — no duplicate aggregation logic needed."""
    long = long.copy()
    long["_is_summary"] = False
    keys = long.groupby(["league", "season", "team"])["date"].max().reset_index()
    stubs = pd.DataFrame({
        "league":      keys["league"].values,
        "season":      keys["season"].values,
        "team":        keys["team"].values,
        "date":        keys["date"].values + pd.Timedelta(days=1),
        "match_id":    keys["league"] + "_" + keys["season"] + "_" + keys["team"],
        "is_home":     False,
        "_is_summary": True,
    })
    return pd.concat([long, stubs], ignore_index=True)


def build_match_team_stats() -> pd.DataFrame:
    frames = []
    for league in _LEAGUES:
        for season in _SEASONS:
            path = _RAW_DIR / f"{league}_{season}.csv"
            if path.exists():
                frames.append(_build_long(pd.read_csv(path), league, season))

    long = _add_stats(_add_summary_stubs(pd.concat(frames, ignore_index=True)))

    # categorical columns based on global population thresholds
    thresholds = json.loads(_THRESHOLDS_PATH.read_text())

    def _cat(col, threshold, hi, lo):
        return np.where(long[col].isna(), None, np.where(long[col] > threshold, hi, lo))

    long["season_goals_total_cat"] = _cat("season_goals_total_avg", thresholds["goals_total_mean"], "high", "low")
    long["season_goals_diff_cat"]  = _cat("season_goals_diff_avg",  0,                              "positive", "negative")
    long["season_goals_for_cat"]   = _cat("season_goals_for_avg",   thresholds["goals_for_mean"],   "high", "low")
    long["season_goals_agst_cat"]  = _cat("season_goals_agst_avg",  thresholds["goals_agst_mean"],  "high", "low")
    _cat_cols = ["season_goals_total_cat", "season_goals_diff_cat",
                 "season_goals_for_cat",   "season_goals_agst_cat"]

    is_summary = long["_is_summary"].fillna(False).astype(bool)
    _out_cols = _TEAM_SEASON_COLS + _STAT_COLS + _cat_cols + _LOCALITY_STAT_COLS

    # team_season — long grain: (team, match), real rows only
    team_season = round_floats(long[~is_summary][_out_cols]
                   .sort_values(["league", "season", "team", "date"])
                   .reset_index(drop=True))

    # team_season_final — one row per (team, season): stats include the last match
    _final_stat_cols = [c for c in _STAT_COLS if c != "game_number"]
    _final_cols = ["league", "season", "team"] + _final_stat_cols + _cat_cols + _LOCALITY_STAT_COLS
    team_season_final = round_floats(long[is_summary][[c for c in _final_cols if c in long.columns]]
                         .sort_values(["league", "season", "team"])
                         .reset_index(drop=True))

    _FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    team_season.to_csv(_FEATURES_DIR / "team_season.csv", index=False)
    team_season.to_parquet(_FEATURES_DIR / "team_season.parquet", index=False)
    team_season_final.to_parquet(_FEATURES_DIR / "team_season_final.parquet", index=False)
    team_season_final.to_csv(_FEATURES_DIR / "team_season_final.csv", index=False)

    # match_team_stats — match grain: (match_id)
    stat_cols = _STAT_COLS + _cat_cols
    home_stats = (
        long[long["is_home"]][["match_id"] + stat_cols]
        .rename(columns={c: f"homet_{c}" for c in stat_cols})
    )
    away_stats = (
        long[~long["is_home"]][["match_id"] + stat_cols]
        .rename(columns={c: f"awayt_{c}" for c in stat_cols})
    )
    base = (
        long[long["is_home"]][["match_id", "league", "season", "date", "team"]]
        .rename(columns={"team": "home_team"})
        .merge(
            long[~long["is_home"]][["match_id", "team"]].rename(columns={"team": "away_team"}),
            on="match_id",
        )
    )

    _home_loc_cols = [c for c in _LOCALITY_STAT_COLS if c.startswith("home_")]
    _away_loc_cols = [c for c in _LOCALITY_STAT_COLS if c.startswith("away_")]
    home_locality = (
        long[long["is_home"]][["match_id"] + _home_loc_cols]
        .rename(columns={c: f"homet_{c}" for c in _home_loc_cols})
    )
    away_locality = (
        long[~long["is_home"]][["match_id"] + _away_loc_cols]
        .rename(columns={c: f"awayt_{c}" for c in _away_loc_cols})
    )

    result = (base
        .merge(home_stats, on="match_id")
        .merge(away_stats, on="match_id")
        .merge(home_locality, on="match_id")
        .merge(away_locality, on="match_id")
    )
    result = round_floats(result[["match_id"] + _columns()])

    result.to_csv(_FEATURES_DIR / "match_team_stats.csv", index=False)
    result.to_parquet(_FEATURES_DIR / "match_team_stats.parquet", index=False)

    return result
