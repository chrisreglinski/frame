import itertools
import json
from pathlib import Path

import pandas as pd
import yaml

from features.utils import round_floats


_ROOT = Path(__file__).parents[2]
_DATA = _ROOT / "01_data"
_YAML_PATH = _ROOT / "06_docs" / "data.yaml"


def _features_dir(group: str) -> Path:
    return _DATA / "02_features" / group


def _columns() -> list[str]:
    with open(_YAML_PATH) as f:
        schema = yaml.safe_load(f)
    result = []
    for name, meta in schema["columns"].items():
        if not meta:
            continue
        table = meta.get("table")
        if "match_matchup_stats" not in (table if isinstance(table, list) else [table]):
            continue
        if "dims" in meta:
            keys = list(meta["dims"].keys())
            values = [list(v) if isinstance(v, list) else [v] for v in (meta["dims"][k] for k in keys)]
            for combo in itertools.product(*values):
                result.append(name.format(**dict(zip(keys, combo))))
        else:
            result.append(name)
    return result


def _windows() -> list[str]:
    with open(_YAML_PATH) as f:
        schema = yaml.safe_load(f)
    for name, meta in schema["columns"].items():
        if not meta:
            continue
        table = meta.get("table")
        if "match_matchup_stats" not in (table if isinstance(table, list) else [table]):
            continue
        if "dims" in meta and "window" in meta["dims"]:
            w = meta["dims"]["window"]
            return list(w) if isinstance(w, list) else [w]
    return []


def build_match_matchup_stats(group: str = "major") -> pd.DataFrame:
    features_dir = _features_dir(group)
    src = pd.read_parquet(features_dir / "match_team_stats.parquet")
    # elo and league live in match_info, not match_team_stats — pull them in for the matchup
    # elo features (league is needed for the per-league home-field intercept)
    info = pd.read_parquet(features_dir / "match_info.parquet")[["match_id", "hmt_elo", "awt_elo", "league"]]
    src = src.merge(info, on="match_id", how="left")

    computed = {}
    for window in _windows():
        computed[f"teams_{window}_goals_foragst_avg_max"] = src[[
            f"hmt_{window}_goals_for_avg",
            f"hmt_{window}_goals_agst_avg",
            f"awt_{window}_goals_for_avg",
            f"awt_{window}_goals_agst_avg",
        ]].max(axis=1)
        for side in ["hmt", "awt"]:
            computed[f"{side}_{window}_goals_foragst_avg_max"] = src[[
                f"{side}_{window}_goals_for_avg",
                f"{side}_{window}_goals_agst_avg",
            ]].max(axis=1)
        # matchup "tilt" diffs: home per-team quantity minus away per-team quantity.
        # for/agst quantities carry an inner for-agst diff already (goals_diff, shots_on_target_diff),
        # so between-teams gives *_diff_avg_diff; the per-side scalars (points, impl_points) give *_avg_diff.
        computed[f"teams_{window}_goals_diff_avg_diff"] = (
            src[f"hmt_{window}_goals_diff_avg"] - src[f"awt_{window}_goals_diff_avg"]
        )
        computed[f"teams_{window}_shots_on_target_diff_avg_diff"] = (
            src[f"hmt_{window}_shots_on_target_diff_avg"] - src[f"awt_{window}_shots_on_target_diff_avg"]
        )
        computed[f"teams_{window}_xg_diff_avg_diff"] = (
            src[f"hmt_{window}_xg_diff_avg"] - src[f"awt_{window}_xg_diff_avg"]
        )
        computed[f"teams_{window}_points_avg_diff"] = (
            src[f"hmt_{window}_points_avg"] - src[f"awt_{window}_points_avg"]
        )
        for line in ["mp", "mc"]:
            computed[f"teams_{window}_{line}_impl_points_avg_diff"] = (
                src[f"hmt_{window}_{line}_impl_points_avg"] - src[f"awt_{window}_{line}_impl_points_avg"]
            )
        # matchup "total": home per-team quantity plus away — combined intensity / level of the match.
        computed[f"teams_{window}_goals_total_avg_total"] = (
            src[f"hmt_{window}_goals_total_avg"] + src[f"awt_{window}_goals_total_avg"]
        )
        computed[f"teams_{window}_shots_on_target_total_avg_total"] = (
            src[f"hmt_{window}_shots_on_target_total_avg"] + src[f"awt_{window}_shots_on_target_total_avg"]
        )
        computed[f"teams_{window}_xg_total_avg_total"] = (
            src[f"hmt_{window}_xg_total_avg"] + src[f"awt_{window}_xg_total_avg"]
        )
        computed[f"teams_{window}_points_avg_total"] = (
            src[f"hmt_{window}_points_avg"] + src[f"awt_{window}_points_avg"]
        )
        for line in ["mp", "mc"]:
            computed[f"teams_{window}_{line}_impl_points_avg_total"] = (
                src[f"hmt_{window}_{line}_impl_points_avg"] + src[f"awt_{window}_{line}_impl_points_avg"]
            )

    # matchup elo (not windowed — elo is a raw pre-match rating)
    computed["teams_elo_diff"] = src["hmt_elo"] - src["awt_elo"]
    computed["teams_elo_total"] = src["hmt_elo"] + src["awt_elo"]

    # elo -> home-minus-away margin, split into strength (_stgh, odd in d) + league home-field
    # (_lhfa) = _diff. 2x2 grid FORM (c=cubic / l=logistic) x FIT (rr=results / mr=market), all
    # from fixed offline params in thresholds.json (see features/thresholds.py).
    thresholds = json.loads((features_dir / "thresholds.json").read_text())
    d = computed["teams_elo_diff"]

    # CUBIC g(d) = a0[league] + a1*d + a2*d^2 + a3*d^3; strength = a1*d + a3*d^3, home field = a0 + a2*d^2.
    for prefix, key in [("crr", "elo_crr"), ("cmr", "elo_cmr")]:
        fit = thresholds[key]
        a1, a2, a3 = fit["a1"], fit["a2"], fit["a3"]
        # clamp d to the cubic's monotone range: g turns over where dg/dd = 3a3*d^2 + 2a2*d + a1 = 0.
        # past those turning points a stronger favourite would get a *smaller* margin (the negative-a3
        # runaway), so hold d at the turning point — the extreme margin becomes a plateau, not a reversal.
        disc = (2 * a2) ** 2 - 12 * a3 * a1
        if a3 < 0 < disc:
            r1 = (-2 * a2 + disc ** 0.5) / (6 * a3)
            r2 = (-2 * a2 - disc ** 0.5) / (6 * a3)
            dc = d.clip(min(r1, r2), max(r1, r2))
        else:
            dc = d
        stgh = a1 * dc + a3 * dc ** 3
        lhfa = src["league"].map(fit["a0"]) + a2 * dc ** 2
        computed[f"teams_elo_diff_{prefix}_impl_stgh"] = stgh
        computed[f"teams_elo_diff_{prefix}_impl_lhfa"] = lhfa
        computed[f"teams_elo_diff_{prefix}_impl_diff"] = stgh + lhfa

    # LOGISTIC margin(x) = 2/(1 + 10^(-x/scale)) - 1 (ClubElo Elo equation). strength = margin at
    # hfa=0 (venue-free, odd); diff = margin at d + hfa[league]; home field = diff - strength (saturates,
    # so no clamping needed). Only scale (global) + hfa[league] (elo points) are fitted.
    def _logistic(x, scale):
        return 2.0 / (1.0 + 10.0 ** (-x / scale)) - 1.0
    for prefix, key in [("lrr", "elo_lrr"), ("lmr", "elo_lmr")]:
        fit = thresholds[key]
        scale = fit["scale"]
        stgh = _logistic(d, scale)
        diff = _logistic(d + src["league"].map(fit["hfa"]), scale)
        computed[f"teams_elo_diff_{prefix}_impl_stgh"] = stgh
        computed[f"teams_elo_diff_{prefix}_impl_lhfa"] = diff - stgh
        computed[f"teams_elo_diff_{prefix}_impl_diff"] = diff

    result = round_floats(pd.concat(
        [src[["match_id"]], pd.DataFrame(computed, index=src.index)],
        axis=1,
    )[["match_id"] + _columns()])

    features_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(features_dir / "match_matchup_stats.csv", index=False)
    result.to_parquet(features_dir / "match_matchup_stats.parquet", index=False)

    return result
