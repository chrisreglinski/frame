from pathlib import Path

import pandas as pd
import yaml

_YAML_PATH = Path(__file__).parents[2] / "06_docs" / "data.yaml"
_DATA = Path(__file__).parents[2] / "01_data"
_LEAGUE_SETS_PATH = _DATA / "league_sets.yaml"
_MATCHES_DIR = _DATA / "01_raw" / "01_matches"


def _precision() -> int:
    with open(_YAML_PATH) as f:
        return yaml.safe_load(f).get("float_precision", 6)


def round_floats(df: pd.DataFrame) -> pd.DataFrame:
    p = _precision()
    float_cols = df.select_dtypes(include="float").columns
    return df.assign(**{c: df[c].round(p) for c in float_cols})


def league_set(name: str) -> list[str]:
    """Leagues that make up a league set, from 01_data/league_sets.yaml.

    Raw data is stored per league; a set (major / minor / other) is what gets built
    together into 02_features/<set>/ and 03_abt/<set>/."""
    with open(_LEAGUE_SETS_PATH) as f:
        sets = yaml.safe_load(f)
    if name not in sets:
        raise KeyError(f"unknown league set '{name}'; known: {sorted(sets)}")
    return list(sets[name])


def match_files(group: str) -> list[Path]:
    """Match CSVs of every league in the set, ordered by file name so that the row
    order does not depend on how the set is listed in the config."""
    paths = [path for league in league_set(group)
             for path in (_MATCHES_DIR / league).glob("*_matches.csv")]
    return sorted(paths, key=lambda p: p.name)

