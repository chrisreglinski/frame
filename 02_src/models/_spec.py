"""The model contract: what every model in the repository declares, and helpers over it.

A model is a declaration, not a pipeline. It is plain data: what it learns (target, domain, features,
estimator name, hyperparameters) with no code. It loads no data, trains nothing, and knows nothing
about the report. `models.run` applies a model to data (building the estimator through the factory),
and `reporting.tearsheet` renders the result.

Features are a plain column list, resolved from feature spaces upstream (see models.spaces); the
`spaces` field keeps the space names as metadata for the slug and the snapshot, and nothing in the
run path reads them. See 06_docs/models_spec.
"""
from dataclasses import asdict, dataclass, field


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
    """A named model, plain data.

    name/tagline/method are identity and display. target and implied are ABT column names (the 0/1
    outcome and the market price). domain, features and hyperparams define what is learned and where.
    estimator is a factory name (see models.factory) built from the hyperparams. spaces are the
    feature-space names the features were resolved from, kept as metadata. slug is a hand-written
    display label only, never parsed for logic (every value is read from its own field). findings
    holds model-specific notes keyed by report section (empty until written).
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
    findings: dict = field(default_factory=dict)


def apply_domain(abt, domain):
    """Rows of `abt` inside the domain: both teams past domain.gameweek_min.

    Leagues are already fixed by the ABT that was loaded, so only the gameweek filter applies here.
    """
    gw = domain.gameweek_min
    return abt[(abt["hmt_game_number"] > gw) & (abt["awt_game_number"] > gw)]


def snapshot(model):
    """The model as a plain dict, for freezing into a run manifest. Every field is data, so asdict
    (which recurses into the Domain) gives a fully plain result of dicts, lists and scalars."""
    return asdict(model)
