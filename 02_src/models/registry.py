"""The model repository: named models the report and scoring code look up by name.

One entry per model, keyed by its declared name. This is the working registry while models live in
code. It can later become an index over persisted runs (see 06_docs on the model repository).
"""
from models.sybilla import sybilla

_MODELS = {m.name: m for m in [sybilla]}


def get(name):
    """The model registered under `name`."""
    try:
        return _MODELS[name]
    except KeyError:
        raise KeyError(f"no model named {name!r}; known: {sorted(_MODELS)}") from None


def names():
    """All registered model names."""
    return sorted(_MODELS)
