"""The model contract: what every model in the repository declares, and helpers over it.

A model is a declaration, not a pipeline. It carries the parameters that define what it learns (the
data fields, snapshotted into every run) and one thin hook, make_model, that builds the estimator
from the hyperparameters. It loads no data, trains nothing, and knows nothing about the report.
`models.run` applies a model to data, and `reporting.tearsheet` renders the result.

The parameters live in code here for flexibility while the set of estimators is still moving. Because
every field except make_model is plain data, the repository can later move to a declarative form
(a per-model YAML plus an estimator factory) without touching run or the report. See
06_docs on the model repository.
"""
from dataclasses import asdict, dataclass, field
from typing import Callable


@dataclass(frozen=True)
class Domain:
    """The slice of matches a model is defined on.

    gameweek_min filters rows (both teams must be past that game number). leagues names the prebuilt
    ABT under 01_data/03_abt/<leagues>/ that the model trains and scores on, so it selects the data
    source rather than filtering rows.
    """
    gameweek_min: int
    leagues: str


@dataclass(frozen=True)
class Model:
    """A named model: its learning parameters plus the estimator hook.

    name/tagline/method are identity and display. target and implied are ABT column names (the 0/1
    outcome and the market price). domain, features and hyperparams define what is learned and where.
    make_model takes the hyperparams dict and returns a fresh sklearn-style estimator. findings holds
    model-specific notes keyed by report section (empty until written).
    """
    name: str
    tagline: str
    method: str
    target: str
    implied: str
    domain: Domain
    features: list
    hyperparams: dict
    make_model: Callable
    findings: dict = field(default_factory=dict)


def apply_domain(abt, domain):
    """Rows of `abt` inside the domain: both teams past domain.gameweek_min.

    Leagues are already fixed by the ABT that was loaded, so only the gameweek filter applies here.
    """
    gw = domain.gameweek_min
    return abt[(abt["hmt_game_number"] > gw) & (abt["awt_game_number"] > gw)]


def snapshot(model):
    """The model's data parameters as a plain dict, for freezing into a run manifest.

    Every field except make_model, which is code and cannot be serialized. asdict recurses into the
    Domain, so the result is fully plain (dicts, lists and scalars).
    """
    data = asdict(model)
    data.pop("make_model", None)
    return data
