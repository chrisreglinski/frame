from pathlib import Path

import pandas as pd
import yaml

_YAML_PATH = Path(__file__).parents[2] / "06_docs" / "data.yaml"


def _precision() -> int:
    with open(_YAML_PATH) as f:
        return yaml.safe_load(f).get("float_precision", 6)


def round_floats(df: pd.DataFrame) -> pd.DataFrame:
    p = _precision()
    float_cols = df.select_dtypes(include="float").columns
    return df.assign(**{c: df[c].round(p) for c in float_cols})
