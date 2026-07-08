import itertools
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

    computed = {}
    for w in _windows():
        computed[f"teams_{w}_goals_foragst_avg_max"] = src[[
            f"hmt_{w}_goals_for_avg",
            f"hmt_{w}_goals_agst_avg",
            f"awt_{w}_goals_for_avg",
            f"awt_{w}_goals_agst_avg",
        ]].max(axis=1)
        for side in ["hmt", "awt"]:
            computed[f"{side}_{w}_goals_foragst_avg_max"] = src[[
                f"{side}_{w}_goals_for_avg",
                f"{side}_{w}_goals_agst_avg",
            ]].max(axis=1)
        # matchup "tilt" diffs: home per-team quantity minus away per-team quantity.
        # for/agst quantities carry an inner for-agst diff already (goals_diff, shots_on_target_diff),
        # so between-teams gives *_diff_avg_diff; the per-side scalars (points, impl_points) give *_avg_diff.
        computed[f"teams_{w}_goals_diff_avg_diff"] = (
            src[f"hmt_{w}_goals_diff_avg"] - src[f"awt_{w}_goals_diff_avg"]
        )
        computed[f"teams_{w}_shots_on_target_diff_avg_diff"] = (
            src[f"hmt_{w}_shots_on_target_diff_avg"] - src[f"awt_{w}_shots_on_target_diff_avg"]
        )
        computed[f"teams_{w}_points_avg_diff"] = (
            src[f"hmt_{w}_points_avg"] - src[f"awt_{w}_points_avg"]
        )
        computed[f"teams_{w}_impl_points_avg_diff"] = (
            src[f"hmt_{w}_impl_points_avg"] - src[f"awt_{w}_impl_points_avg"]
        )

    result = round_floats(pd.concat(
        [src[["match_id"]], pd.DataFrame(computed, index=src.index)],
        axis=1,
    )[["match_id"] + _columns()])

    features_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(features_dir / "match_matchup_stats.csv", index=False)
    result.to_parquet(features_dir / "match_matchup_stats.parquet", index=False)

    return result
