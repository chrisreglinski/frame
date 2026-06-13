import hashlib
import itertools
from pathlib import Path

import pandas as pd
import yaml


_ROOT = Path(__file__).parents[2]
_RAW_DIR = _ROOT / "01_data" / "01_raw" / "01_matches"
_FEATURES_DIR = _ROOT / "01_data" / "02_features"
_YAML_PATH = _ROOT / "06_docs" / "data.yaml"

_LEAGUES = ["england", "spain", "italy", "germany", "france"]
_SEASONS = ["2223", "2324", "2425", "2526"]

_TEAM_SEASON_COLS = [
    "match_id", "league", "season", "date", "team", "opponent", "is_home",
    "goals_for", "goals_agst", "points", "goals_diff", "goals_total",
]

_STAT_COLS = [
    "game_number",
    *[f"season_{m}_{k}_avg" for m in ["goals", "shots", "shots_on_target", "corners", "yellow"] for k in ["for", "agst"]],
    *[f"season_{m}_{v}" for m in ["points", "goals_diff", "goals_total"] for v in ["avg", "std"]],
    *[f"season_{r}_ratio" for r in ["wins", "draws", "losses"]],
    *[f"season_goals_total_le{x}_ratio" for x in [0, 1, 2, 3]],
    *[f"season_goals_total_ge{x}_ratio" for x in [2, 3, 4]],
    *[f"season_goals_{kind}_le{x}_ratio" for kind in ["for", "agst"] for x in [0, 1]],
    *[f"season_goals_{kind}_ge{x}_ratio" for kind in ["for", "agst"] for x in [2, 3]],
    *[f"season_shots_on_target_{k}_ratio" for k in ["for", "agst"]],
    "season_red_for_avg",
    "red_last_match",
]


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
    })

    long = pd.concat([home, away], ignore_index=True)
    long["goals_diff"] = long["goals_for"] - long["goals_agst"]
    long["goals_total"] = long["goals_for"] + long["goals_agst"]
    return long


def _add_stats(long: pd.DataFrame) -> pd.DataFrame:
    long = long.sort_values(["league", "season", "team", "date"]).reset_index(drop=True)
    grp = ["league", "season", "team"]
    g = long.groupby(grp, sort=False)

    long["game_number"] = g.cumcount() + 1  # 1-based; stats are NaN at game_number=1 due to shift

    def em(col):
        return g[col].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())

    def es(col):
        return g[col].transform(lambda x: x.shift(1).expanding(min_periods=1).std(ddof=0))

    def esum(col):
        return g[col].transform(lambda x: x.shift(1).expanding(min_periods=1).sum())

    for m in ["goals", "shots", "shots_on_target", "corners", "yellow"]:
        for k in ["for", "agst"]:
            long[f"season_{m}_{k}_avg"] = em(f"{m}_{k}")

    for m in ["points", "goals_diff", "goals_total"]:
        long[f"season_{m}_avg"] = em(m)
        long[f"season_{m}_std"] = es(m)

    for r, c in [("wins", "is_win"), ("draws", "is_draw"), ("losses", "is_loss")]:
        long[f"season_{r}_ratio"] = em(c)

    for x in [0, 1, 2, 3]:
        long[f"_ind_gt_le{x}"] = (long["goals_total"] <= x).astype(float)
        long[f"season_goals_total_le{x}_ratio"] = em(f"_ind_gt_le{x}")
    for x in [2, 3, 4]:
        long[f"_ind_gt_ge{x}"] = (long["goals_total"] >= x).astype(float)
        long[f"season_goals_total_ge{x}_ratio"] = em(f"_ind_gt_ge{x}")
    for kind in ["for", "agst"]:
        for x in [0, 1]:
            long[f"_ind_g{kind[0]}le{x}"] = (long[f"goals_{kind}"] <= x).astype(float)
            long[f"season_goals_{kind}_le{x}_ratio"] = em(f"_ind_g{kind[0]}le{x}")
        for x in [2, 3]:
            long[f"_ind_g{kind[0]}ge{x}"] = (long[f"goals_{kind}"] >= x).astype(float)
            long[f"season_goals_{kind}_ge{x}_ratio"] = em(f"_ind_g{kind[0]}ge{x}")

    for k in ["for", "agst"]:
        long[f"season_shots_on_target_{k}_ratio"] = esum(f"shots_on_target_{k}") / esum(f"shots_{k}")

    long["season_red_for_avg"] = em("red_for")

    red_shifted = g["red_for"].transform(lambda x: x.shift(1))
    long["red_last_match"] = (
        (red_shifted > 0)
        .where(red_shifted.notna(), other=pd.NA)
        .astype(pd.BooleanDtype())
    )

    return long


def build_match_team_stats() -> pd.DataFrame:
    frames = []
    for league in _LEAGUES:
        for season in _SEASONS:
            path = _RAW_DIR / f"{league}_{season}.csv"
            if path.exists():
                frames.append(_build_long(pd.read_csv(path), league, season))

    long = _add_stats(pd.concat(frames, ignore_index=True))

    # team_season — long grain: (team, match)
    team_season = long[_TEAM_SEASON_COLS + _STAT_COLS].sort_values(
        ["league", "season", "team", "date"]
    ).reset_index(drop=True)

    _FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    team_season.to_csv(_FEATURES_DIR / "team_season.csv", index=False)
    team_season.to_parquet(_FEATURES_DIR / "team_season.parquet", index=False)

    # match_team_stats — match grain: (match_id)
    home_stats = (
        long[long["is_home"]][["match_id"] + _STAT_COLS]
        .rename(columns={c: f"home_{c}" for c in _STAT_COLS})
    )
    away_stats = (
        long[~long["is_home"]][["match_id"] + _STAT_COLS]
        .rename(columns={c: f"away_{c}" for c in _STAT_COLS})
    )
    base = (
        long[long["is_home"]][["match_id", "league", "season", "date", "team"]]
        .rename(columns={"team": "home_team"})
        .merge(
            long[~long["is_home"]][["match_id", "team"]].rename(columns={"team": "away_team"}),
            on="match_id",
        )
    )

    result = base.merge(home_stats, on="match_id").merge(away_stats, on="match_id")
    result = result[["match_id"] + _columns()]

    result.to_csv(_FEATURES_DIR / "match_team_stats.csv", index=False)
    result.to_parquet(_FEATURES_DIR / "match_team_stats.parquet", index=False)

    return result
