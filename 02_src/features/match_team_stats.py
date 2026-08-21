import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from features.utils import league_set, match_files, round_floats


_ROOT = Path(__file__).parents[2]
_DATA = _ROOT / "01_data"
_YAML_PATH = _ROOT / "06_docs" / "data.yaml"


def _features_dir(group: str) -> Path:
    return _DATA / "02_features" / group


_XG_DIR = _DATA / "01_raw" / "05_xg"


def _attach_xg(raw: pd.DataFrame, league: str, season: str) -> pd.DataFrame:
    """Merge Understat home_xg/away_xg onto the raw match rows by (HomeTeam, AwayTeam).
    Understat names are translated to ours via the league's team_map; matches with no xg
    row (or leagues Understat does not cover) keep NaN xg."""
    path = _XG_DIR / league / f"{league}_{season}_xg.csv"
    tmap = _XG_DIR / league / "team_map.csv"
    if not path.exists() or not tmap.exists():
        return raw.assign(home_xg=np.nan, away_xg=np.nan)
    u2o = pd.read_csv(tmap).set_index("understat")["team"].to_dict()
    xg = pd.read_csv(path, usecols=["home_team", "away_team", "home_xg", "away_xg"])
    xg["HomeTeam"] = xg["home_team"].map(lambda t: u2o.get(t, t))
    xg["AwayTeam"] = xg["away_team"].map(lambda t: u2o.get(t, t))
    return raw.merge(xg[["HomeTeam", "AwayTeam", "home_xg", "away_xg"]],
                     on=["HomeTeam", "AwayTeam"], how="left")


_WINDOWS = {"season": None, "rolling6": 6, "rolling8": 8}

_LOCALITY_METRICS = ["goals", "shots", "shots_on_target"]
_LOCALITY_SCALAR_METRICS = ["points", "flg_diff", "goals_diff", "goals_total"]
_LOCALITY_RATIO_RESULTS = ["wins", "draws", "losses"]
_LOCALITY_ADV_METRICS = ["points", "mp_impl_points", "mc_impl_points", "flg_diff", "mp_impl_diff", "mc_impl_diff"]
_LOCALITY_STAT_COLS = [
    f"{loc}_season_{m}_{k}_avg"
    for loc in ["home", "away"]
    for m in _LOCALITY_METRICS
    for k in ["for", "agst"]
] + [
    f"{loc}_season_{m}_avg"
    for loc in ["home", "away"]
    for m in _LOCALITY_SCALAR_METRICS
] + [
    f"{loc}_season_{r}_ratio"
    for loc in ["home", "away"]
    for r in _LOCALITY_RATIO_RESULTS
] + [
    f"{loc}_season_{line}_impl_points_avg"
    for loc in ["home", "away"]
    for line in ["mp", "mc"]
] + [
    f"{loc}_season_{line}_impl_diff_avg"
    for loc in ["home", "away"]
    for line in ["mp", "mc"]
] + [
    f"{loc}_season_{m}_avg_adv"
    for loc in ["home", "away"]
    for m in _LOCALITY_ADV_METRICS
]

_TEAM_SEASON_COLS = [
    "match_id", "league", "season", "date", "team", "opponent", "is_home",
    "goals_for", "goals_agst", "points", "goals_diff", "goals_total",
]


def _make_stat_cols() -> list[str]:
    cols = ["game_number"]
    for window in _WINDOWS:
        cols += [f"{window}_{m}_{k}_avg" for m in ["goals", "shots", "shots_on_target", "corners", "yellow", "xg"] for k in ["for", "agst"]]
        cols += [f"{window}_{m}_{v}" for m in ["points", "goals_diff", "goals_total", "shots_on_target_diff", "shots_on_target_total", "xg_diff", "xg_total"] for v in ["avg", "std"]]
        cols += [f"{window}_{r}_ratio" for r in ["wins", "draws", "losses"]]
        cols += [f"{window}_flg_diff_avg"]
        cols += [f"{window}_goals_total_le{x}_ratio" for x in [0, 1, 2, 3, 4]]
        cols += [f"{window}_goals_{kind}_le{x}_ratio" for kind in ["for", "agst"] for x in [0, 1, 2]]
        cols += [f"{window}_shots_on_target_{k}_ratio" for k in ["for", "agst"]]
        cols += [f"{window}_red_for_avg"]
        cols += [f"{window}_{line}_impl_{o}_avg" for line in ["mp", "mc"] for o in ["win", "draw", "loss"]]
        cols += [f"{window}_{line}_impl_points_avg" for line in ["mp", "mc"]]
        cols += [f"{window}_{line}_impl_diff_avg" for line in ["mp", "mc"]]
    cols += ["red_last_match"]
    return cols


_STAT_COLS = _make_stat_cols()


def _columns(group: str) -> list[str]:
    with open(_YAML_PATH) as f:
        schema = yaml.safe_load(f)
    result = []
    for name, meta in schema["columns"].items():
        if not meta:
            continue
        table = meta.get("table")
        if "team_stats" not in (table if isinstance(table, list) else [table]):
            continue
        leagues = meta.get("leagues")
        if leagues is not None and not set(leagues) & set(league_set(group)):
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
    # mp = market pre-closing (AvgH/D/A); mc = market closing (AvgCH/CD/CA, may be absent -> NaN)
    def _col(name):
        return raw[name].values if name in raw.columns else np.full(len(raw), np.nan)
    avgc_h, avgc_d, avgc_a = _col("AvgCH"), _col("AvgCD"), _col("AvgCA")

    # xg is Understat-sourced (major only); absent for groups/matches without coverage -> NaN
    home_xg = raw["home_xg"].values if "home_xg" in raw else np.full(len(raw), np.nan)
    away_xg = raw["away_xg"].values if "away_xg" in raw else np.full(len(raw), np.nan)

    home = pd.DataFrame({
        **base, "is_home": True, "team": raw["HomeTeam"].values, "opponent": raw["AwayTeam"].values,
        "goals_for": raw["FTHG"].values,    "goals_agst": raw["FTAG"].values,
        "shots_for": raw["HS"].values,       "shots_agst": raw["AS"].values,
        "shots_on_target_for": raw["HST"].values, "shots_on_target_agst": raw["AST"].values,
        "corners_for": raw["HC"].values,     "corners_agst": raw["AC"].values,
        "yellow_for": raw["HY"].values,      "yellow_agst": raw["AY"].values,
        "xg_for": home_xg,                   "xg_agst": away_xg,
        "red_for": raw["HR"].values,
        "points": raw["FTR"].map({"H": 3, "D": 1, "A": 0}).values,
        "is_win":  (raw["FTR"] == "H").astype(float).values,
        "is_draw": (raw["FTR"] == "D").astype(float).values,
        "is_loss": (raw["FTR"] == "A").astype(float).values,
        "mp_impl_win": 1.0 / avg_h, "mp_impl_draw": 1.0 / avg_d, "mp_impl_loss": 1.0 / avg_a,
        "mc_impl_win": 1.0 / avgc_h, "mc_impl_draw": 1.0 / avgc_d, "mc_impl_loss": 1.0 / avgc_a,
    })

    away = pd.DataFrame({
        **base, "is_home": False, "team": raw["AwayTeam"].values, "opponent": raw["HomeTeam"].values,
        "goals_for": raw["FTAG"].values,    "goals_agst": raw["FTHG"].values,
        "shots_for": raw["AS"].values,       "shots_agst": raw["HS"].values,
        "shots_on_target_for": raw["AST"].values, "shots_on_target_agst": raw["HST"].values,
        "corners_for": raw["AC"].values,     "corners_agst": raw["HC"].values,
        "yellow_for": raw["AY"].values,      "yellow_agst": raw["HY"].values,
        "xg_for": away_xg,                   "xg_agst": home_xg,
        "red_for": raw["AR"].values,
        "points": raw["FTR"].map({"A": 3, "D": 1, "H": 0}).values,
        "is_win":  (raw["FTR"] == "A").astype(float).values,
        "is_draw": (raw["FTR"] == "D").astype(float).values,
        "is_loss": (raw["FTR"] == "H").astype(float).values,
        "mp_impl_win": 1.0 / avg_a, "mp_impl_draw": 1.0 / avg_d, "mp_impl_loss": 1.0 / avg_h,
        "mc_impl_win": 1.0 / avgc_a, "mc_impl_draw": 1.0 / avgc_d, "mc_impl_loss": 1.0 / avgc_h,
    })

    long = pd.concat([home, away], ignore_index=True)
    long["goals_diff"] = long["goals_for"] - long["goals_agst"]
    long["goals_total"] = long["goals_for"] + long["goals_agst"]
    long["shots_on_target_diff"]  = long["shots_on_target_for"] - long["shots_on_target_agst"]
    long["shots_on_target_total"] = long["shots_on_target_for"] + long["shots_on_target_agst"]
    long["xg_diff"]  = long["xg_for"] - long["xg_agst"]
    long["xg_total"] = long["xg_for"] + long["xg_agst"]
    long["mp_impl_points"] = long["mp_impl_win"] * 3 + long["mp_impl_draw"]
    long["mc_impl_points"] = long["mc_impl_win"] * 3 + long["mc_impl_draw"]
    long["flg_diff"] = long["is_win"] - long["is_loss"]  # signed result +1/0/-1 (win_flg - loss_flg)
    long["mp_impl_diff"] = long["mp_impl_win"] - long["mp_impl_loss"]  # team-oriented margin (= home_away_impl_diff)
    long["mc_impl_diff"] = long["mc_impl_win"] - long["mc_impl_loss"]
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

    for window, wsize in _WINDOWS.items():
        for m in ["goals", "shots", "shots_on_target", "corners", "yellow", "xg"]:
            for k in ["for", "agst"]:
                long[f"{window}_{m}_{k}_avg"] = _agg_mean(g, f"{m}_{k}", wsize)

        for m in ["points", "goals_diff", "goals_total", "shots_on_target_diff", "shots_on_target_total", "xg_diff", "xg_total"]:
            long[f"{window}_{m}_avg"] = _agg_mean(g, m, wsize)
            long[f"{window}_{m}_std"] = _agg_std(g, m, wsize)

        for r, c in [("wins", "is_win"), ("draws", "is_draw"), ("losses", "is_loss")]:
            long[f"{window}_{r}_ratio"] = _agg_mean(g, c, wsize)
        long[f"{window}_flg_diff_avg"] = _agg_mean(g, "flg_diff", wsize)  # = wins_ratio - losses_ratio

        for x in [0, 1, 2, 3, 4]:
            long[f"{window}_goals_total_le{x}_ratio"] = _agg_mean(g, f"_ind_gt_le{x}", wsize)
        for kind in ["for", "agst"]:
            for x in [0, 1, 2]:
                long[f"{window}_goals_{kind}_le{x}_ratio"] = _agg_mean(g, f"_ind_g{kind[0]}le{x}", wsize)

        for k in ["for", "agst"]:
            long[f"{window}_shots_on_target_{k}_ratio"] = (
                _agg_sum(g, f"shots_on_target_{k}", wsize) / _agg_sum(g, f"shots_{k}", wsize)
            )

        long[f"{window}_red_for_avg"] = _agg_mean(g, "red_for", wsize)

        for line in ["mp", "mc"]:
            for o in ["win", "draw", "loss"]:
                long[f"{window}_{line}_impl_{o}_avg"] = _agg_mean(g, f"{line}_impl_{o}", wsize)
            long[f"{window}_{line}_impl_points_avg"] = (
                long[f"{window}_{line}_impl_win_avg"] * 3 + long[f"{window}_{line}_impl_draw_avg"]
            )
            long[f"{window}_{line}_impl_diff_avg"] = (
                long[f"{window}_{line}_impl_win_avg"] - long[f"{window}_{line}_impl_loss_avg"]
            )

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

    # home/away split: win/draw/loss ratio and expected (implied) points
    for loc, mask in [("home", long["is_home"]), ("away", ~long["is_home"])]:
        for r, c in [("wins", "is_win"), ("draws", "is_draw"), ("losses", "is_loss")]:
            long[f"_{loc}_{r}"] = long[c].where(mask)
            long[f"{loc}_season_{r}_ratio"] = _agg_mean(g, f"_{loc}_{r}", None)
        for line in ["mp", "mc"]:
            long[f"_{loc}_{line}_impl_points"] = long[f"{line}_impl_points"].where(mask)
            long[f"{loc}_season_{line}_impl_points_avg"] = _agg_mean(g, f"_{loc}_{line}_impl_points", None)
            long[f"_{loc}_{line}_impl_diff"] = long[f"{line}_impl_diff"].where(mask)
            long[f"{loc}_season_{line}_impl_diff_avg"] = _agg_mean(g, f"_{loc}_{line}_impl_diff", None)

    # venue advantage: a team's form where it plays this match minus at the other venue
    for m in _LOCALITY_ADV_METRICS:
        long[f"home_season_{m}_avg_adv"] = long[f"home_season_{m}_avg"] - long[f"away_season_{m}_avg"]
        long[f"away_season_{m}_avg_adv"] = long[f"away_season_{m}_avg"] - long[f"home_season_{m}_avg"]

    # mask rolling windows for matches with insufficient history
    for window, wsize in _WINDOWS.items():
        if wsize is None:
            continue
        window_cols = [c for c in long.columns if c.startswith(f"{window}_")]
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


def build_match_team_stats(group: str = "major") -> pd.DataFrame:
    features_dir = _features_dir(group)
    thresholds_path = features_dir / "thresholds.json"
    frames = []
    for path in match_files(group):
        league, season = path.stem.removesuffix("_matches").rsplit("_", 1)
        raw = _attach_xg(pd.read_csv(path), league, season)
        frames.append(_build_long(raw, league, season))

    long = _add_stats(_add_summary_stubs(pd.concat(frames, ignore_index=True)))

    # categorical columns based on global population thresholds
    thresholds = json.loads(thresholds_path.read_text())

    def _cat(col, threshold, hi, lo):
        return np.where(long[col].isna(), None, np.where(long[col] > threshold, hi, lo))

    long["season_goals_total_cat2m"] = _cat("season_goals_total_avg", thresholds["goals_total_mean"],   "high", "low")
    long["season_goals_diff_cat2m"]  = _cat("season_goals_diff_avg",  0,                                "positive", "negative")
    long["season_goals_for_cat2m"]   = _cat("season_goals_for_avg",   thresholds["goals_foragst_mean"], "high", "low")
    long["season_goals_agst_cat2m"]  = _cat("season_goals_agst_avg",  thresholds["goals_foragst_mean"], "high", "low")

    is_summary = long["_is_summary"].fillna(False).astype(bool)

    # tertile thresholds from summary rows (one per team-season, unbiased distribution)
    summary_rows = long[is_summary]
    for metric in ["goals_total", "goals_diff", "shots_on_target_total", "shots_on_target_diff"]:
        col = f"season_{metric}_avg"
        thresholds[f"{metric}_p33"] = round(float(summary_rows[col].quantile(1 / 3)), 3)
        thresholds[f"{metric}_p67"] = round(float(summary_rows[col].quantile(2 / 3)), 3)
    # goals_for and goals_agst share a single pooled threshold
    foragst_pool = pd.concat([summary_rows["season_goals_for_avg"], summary_rows["season_goals_agst_avg"]])
    thresholds["goals_foragst_p33"] = round(float(foragst_pool.quantile(1 / 3)), 3)
    thresholds["goals_foragst_p67"] = round(float(foragst_pool.quantile(2 / 3)), 3)
    # shots_on_target — same pooled approach
    sot_pool = pd.concat([summary_rows["season_shots_on_target_for_avg"], summary_rows["season_shots_on_target_agst_avg"]])
    thresholds["shots_on_target_foragst_p33"] = round(float(sot_pool.quantile(1 / 3)), 3)
    thresholds["shots_on_target_foragst_p67"] = round(float(sot_pool.quantile(2 / 3)), 3)
    thresholds_path.write_text(json.dumps(thresholds, indent=2))

    def _cat3(col, p33, p67):
        return np.where(long[col].isna(), None,
               np.where(long[col] <= p33, "low",
               np.where(long[col] <= p67, "medium", "high")))

    long["season_goals_total_cat3q"] = _cat3("season_goals_total_avg", thresholds["goals_total_p33"],   thresholds["goals_total_p67"])
    long["season_goals_diff_cat3q"]  = _cat3("season_goals_diff_avg",  thresholds["goals_diff_p33"],    thresholds["goals_diff_p67"])
    long["season_goals_for_cat3q"]   = _cat3("season_goals_for_avg",   thresholds["goals_foragst_p33"], thresholds["goals_foragst_p67"])
    long["season_goals_agst_cat3q"]  = _cat3("season_goals_agst_avg",  thresholds["goals_foragst_p33"], thresholds["goals_foragst_p67"])

    sot_mean = thresholds["shots_on_target_foragst_mean"]
    long["season_shots_on_target_for_cat2m"]   = _cat("season_shots_on_target_for_avg",   sot_mean, "high", "low")
    long["season_shots_on_target_agst_cat2m"]  = _cat("season_shots_on_target_agst_avg",  sot_mean, "high", "low")
    long["season_shots_on_target_total_cat2m"] = _cat("season_shots_on_target_total_avg", thresholds["shots_on_target_total_mean"], "high", "low")
    long["season_shots_on_target_diff_cat2m"]  = _cat("season_shots_on_target_diff_avg",  0, "positive", "negative")
    long["season_shots_on_target_for_cat3q"]   = _cat3("season_shots_on_target_for_avg",   thresholds["shots_on_target_foragst_p33"], thresholds["shots_on_target_foragst_p67"])
    long["season_shots_on_target_agst_cat3q"]  = _cat3("season_shots_on_target_agst_avg",  thresholds["shots_on_target_foragst_p33"], thresholds["shots_on_target_foragst_p67"])
    long["season_shots_on_target_total_cat3q"] = _cat3("season_shots_on_target_total_avg", thresholds["shots_on_target_total_p33"], thresholds["shots_on_target_total_p67"])
    long["season_shots_on_target_diff_cat3q"]  = _cat3("season_shots_on_target_diff_avg",  thresholds["shots_on_target_diff_p33"],  thresholds["shots_on_target_diff_p67"])

    _cat_cols = [
        "season_goals_total_cat2m", "season_goals_diff_cat2m",
        "season_goals_for_cat2m",   "season_goals_agst_cat2m",
        "season_goals_total_cat3q", "season_goals_diff_cat3q",
        "season_goals_for_cat3q",   "season_goals_agst_cat3q",
        "season_shots_on_target_for_cat2m",   "season_shots_on_target_agst_cat2m",
        "season_shots_on_target_total_cat2m", "season_shots_on_target_diff_cat2m",
        "season_shots_on_target_for_cat3q",   "season_shots_on_target_agst_cat3q",
        "season_shots_on_target_total_cat3q", "season_shots_on_target_diff_cat3q",
    ]
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

    features_dir.mkdir(parents=True, exist_ok=True)
    team_season.to_csv(features_dir / "team_season.csv", index=False)
    team_season.to_parquet(features_dir / "team_season.parquet", index=False)
    team_season_final.to_parquet(features_dir / "team_season_final.parquet", index=False)
    team_season_final.to_csv(features_dir / "team_season_final.csv", index=False)

    # match_team_stats — match grain: (match_id)
    stat_cols = _STAT_COLS + _cat_cols
    home_stats = (
        long[long["is_home"]][["match_id"] + stat_cols]
        .rename(columns={c: f"hmt_{c}" for c in stat_cols})
    )
    away_stats = (
        long[~long["is_home"]][["match_id"] + stat_cols]
        .rename(columns={c: f"awt_{c}" for c in stat_cols})
    )
    base = (
        long[long["is_home"]][["match_id", "league", "season", "date", "team"]]
        .rename(columns={"team": "hmt_name"})
        .merge(
            long[~long["is_home"]][["match_id", "team"]].rename(columns={"team": "awt_name"}),
            on="match_id",
        )
    )

    _home_loc_cols = [c for c in _LOCALITY_STAT_COLS if c.startswith("home_")]
    _away_loc_cols = [c for c in _LOCALITY_STAT_COLS if c.startswith("away_")]
    home_locality = (
        long[long["is_home"]][["match_id"] + _home_loc_cols]
        .rename(columns={c: f"hmt_{c}" for c in _home_loc_cols})
    )
    away_locality = (
        long[~long["is_home"]][["match_id"] + _away_loc_cols]
        .rename(columns={c: f"awt_{c}" for c in _away_loc_cols})
    )

    result = (base
        .merge(home_stats, on="match_id")
        .merge(away_stats, on="match_id")
        .merge(home_locality, on="match_id")
        .merge(away_locality, on="match_id")
    )
    result = round_floats(result[["match_id"] + _columns(group)])

    result.to_csv(features_dir / "match_team_stats.csv", index=False)
    result.to_parquet(features_dir / "match_team_stats.parquet", index=False)

    return result
