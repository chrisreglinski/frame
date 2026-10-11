"""Estimator factory: build a fresh estimator from a name and hyperparameters.

A model records its estimator by name, and this maps the name plus the model's hyperparameters to a
fresh sklearn-style object. Register a new estimator type with `@register(name)`; the model records
stay plain data. Operational settings that are not modeling choices (n_jobs, verbosity, seed) are
added here, not stored per model.

    build("xgb", {"max_depth": 50, "n_estimators": 150})  -> a fresh XGBClassifier
"""

_BUILDERS = {}


def register(name):
    """Register a builder `hyperparams -> estimator` under `name`."""
    def deco(fn):
        _BUILDERS[name] = fn
        return fn
    return deco


def build(estimator, hyperparams):
    """A fresh estimator of type `estimator`, built from `hyperparams`."""
    try:
        builder = _BUILDERS[estimator]
    except KeyError:
        raise KeyError(f"unknown estimator {estimator!r}; known: {names()}") from None
    return builder(dict(hyperparams))


def names():
    """All registered estimator names."""
    return sorted(_BUILDERS)


@register("xgb")
def _xgb(hp):
    from xgboost import XGBClassifier
    return XGBClassifier(**hp, eval_metric="logloss", n_jobs=4, verbosity=0,
                         enable_categorical=True, random_state=0)
