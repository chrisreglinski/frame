"""The model repository as a view over 04_models: every model is a params.yaml folder.

There is no in-code list of models. `get` and `names` scan `04_models/*/params.yaml` and load each
definition. A model is keyed by its human name if it has one, else its slug. Since 04_models is
gitignored, a fresh checkout has no models until params files are created.
"""
from pathlib import Path

from models.loader import load_model
from models.store import ROOT


def _index(root=ROOT):
    """name-or-slug -> Model for every params.yaml under `root`."""
    models = {}
    for params in sorted(Path(root).glob("*/params.yaml")):
        model = load_model(params)
        models[model.name or model.slug] = model
    return models


def get(name, root=ROOT):
    """The model registered under `name` (its human name or, failing that, its slug)."""
    models = _index(root)
    try:
        return models[name]
    except KeyError:
        raise KeyError(f"no model named {name!r}; known: {sorted(models)}") from None


def names(root=ROOT):
    """All model names (or slugs) found under `root`."""
    return sorted(_index(root))
