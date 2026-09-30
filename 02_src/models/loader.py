"""Load a model definition from a params.yaml into a Model.

This is the one place that turns spaces into features: it reads the authored `spaces` and `extra`
and resolves them to a plain column list, so the rest of the code (run, factory) never sees spaces.
The slug is computed from the definition, never read from the file. Everything else is copied field
for field.
"""
from pathlib import Path

import yaml

from models import spaces
from models._spec import Domain, Model, make_slug


def load_model(path):
    """Build a Model from a params.yaml file."""
    p = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    dom = p["domain"]
    space_names = p.get("spaces", [])
    extra = p.get("extra", [])

    features = spaces.resolve(space_names)
    features += [col for col in extra if col not in features]

    return Model(
        name=p.get("name", ""),
        slug=make_slug(p["target"], dom["leagues"], space_names, extra, p["estimator"]),
        tagline=p.get("tagline", ""),
        method=p.get("method", ""),
        target=p["target"],
        implied=p["implied"],
        domain=Domain(gameweek_min=dom["gameweek_min"], leagues=dom["leagues"]),
        spaces=space_names,
        extra=extra,
        features=features,
        estimator=p["estimator"],
        hyperparams=p.get("hyperparams", {}),
        findings=p.get("findings", {}),
    )
