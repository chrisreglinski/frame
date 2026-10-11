"""The model contract: what every model in the repository declares, and helpers over it.

A model is a declaration, not a pipeline. It is plain data: what it learns (target, domain, features,
estimator name, hyperparameters) with no code. It loads no data, trains nothing, and knows nothing
about the report. `models.run` applies a model to data (building the estimator through the factory),
and `reporting.tearsheet` renders the result.

Features are a plain column list, resolved from feature spaces plus extra variables upstream (see
models.loader); the `spaces` and `extra` fields keep what the features came from, as metadata for the
slug and the snapshot, and nothing in the run path reads them. See 06_docs/models_spec.
"""
import hashlib
import json
from dataclasses import asdict, dataclass, field

from models import domains


@dataclass(frozen=True)
class Domain:
    """The slice of matches a model is defined on.

    leagues names the prebuilt ABT under 01_data/03_abt/<leagues>/ that the model trains and scores
    on, so it selects the data source. name is a registered domain (see models.domains), a row
    condition applied to that ABT.
    """
    leagues: str
    name: str


@dataclass(frozen=True)
class Model:
    """A named model, plain data.

    name/tagline/method are identity and display. target and implied are ABT column names (the 0/1
    outcome and the market price). domain, features and hyperparams define what is learned and where.
    estimator is a factory name (see models.factory) built from the hyperparams. spaces are the
    feature-space names and extra the single variables outside any space that the features were
    resolved from, kept as metadata. slug is a display label computed by make_slug, never parsed for
    logic (every value is read from its own field). findings holds model-specific notes keyed by
    report section (empty until written).
    """
    name: str
    tagline: str
    method: str
    target: str
    implied: str
    domain: Domain
    features: list
    estimator: str
    hyperparams: dict
    slug: str = ""
    spaces: list = field(default_factory=list)
    extra: list = field(default_factory=list)
    findings: dict = field(default_factory=dict)


def problem_slug(target, leagues, domain):
    """The problem a model solves: outcome, league set and domain, e.g. draw-major-core. Models with
    the same problem are comparable. The target part drops the `t_` prefix and `_flg` suffix."""
    short_target = target.removeprefix("t_").removesuffix("_flg")
    return "-".join([short_target, leagues, domain])


def solution_slug(estimator, spaces, extra):
    """How a model solves its problem: estimator, spaces and extra variables, e.g.
    xgb-goals_foragst+points-d5b3b3. Extra variables add a final six-character hash of their sorted
    list, so the slug signals them and tells two sets apart without listing them."""
    parts = [estimator, "+".join(spaces)]
    if extra:
        parts.append(hashlib.sha1(json.dumps(sorted(extra)).encode()).hexdigest()[:6])
    return "-".join(parts)


def make_slug(target, leagues, domain, spaces, extra, estimator):
    """The model's full identity, problem then solution, e.g.
    draw-major-core-xgb-goals_foragst+points-d5b3b3."""
    return f"{problem_slug(target, leagues, domain)}-{solution_slug(estimator, spaces, extra)}"


def model_problem(model):
    """The problem slug of a Model."""
    return problem_slug(model.target, model.domain.leagues, model.domain.name)


def model_solution(model):
    """The solution slug of a Model."""
    return solution_slug(model.estimator, model.spaces, model.extra)


def apply_domain(abt, domain):
    """Rows of `abt` inside the domain's named condition.

    Leagues are already fixed by the ABT that was loaded, so only the row condition applies here.
    """
    return domains.apply(domain.name, abt)


def snapshot(model):
    """The model as a plain dict, for freezing into a run manifest. Every field is data, so asdict
    (which recurses into the Domain) gives a fully plain result of dicts, lists and scalars."""
    return asdict(model)


def model_from_snapshot(data):
    """Rebuild a Model from a snapshot dict (the inverse of snapshot). The features are taken as
    stored, not re-resolved from spaces, so a saved run reproduces exactly what it ran on."""
    data = dict(data)
    dom = dict(data["domain"])
    if "gameweek_min" in dom:                 # snapshots from before named domains
        dom = {"leagues": dom["leagues"], "name": f"after_gw{dom['gameweek_min']}"}
    data["domain"] = Domain(**dom)
    return Model(**data)
