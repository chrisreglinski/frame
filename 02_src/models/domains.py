"""Domains: named slices of matches a model is defined on.

A domain is a row condition over ABT columns, written in DataFrame.query syntax and registered under a
name, the same way spaces name groups of columns. A model refers to its domain by name, and the name
is part of the model's slug. Each domain also carries a plain description for the report. See
06_docs/models_spec.

    condition("core")        -> the query string
    apply("core", abt)       -> the rows of abt inside the domain
    describe("core")         -> the plain description
"""

# name -> (query condition, plain description)
_DOMAINS = {
    "all": (
        "",
        "every match of the season.",
    ),
    "core": (
        "(season_4phase == 'fall' and gameweek_after_break != 1)"
        " or season_4phase == 'winter'"
        " or (season_4phase == 'spring' and gameweek_before_end != -1)",
        "the season in full swing: fall without its first round, all of winter, and spring without "
        "the last round. The opening phase is still noisy form, and the last round often has nothing "
        "at stake.",
    ),
    "after_gw8": (
        "hmt_game_number > 8 and awt_game_number > 8",
        "only matches once both teams are past their eighth game of the season, so early-season noise "
        "is dropped.",
    ),
}


def _entry(name):
    try:
        return _DOMAINS[name]
    except KeyError:
        raise KeyError(f"unknown domain {name!r}; known: {names()}") from None


def condition(name):
    """The query string of a domain ('' for no filter)."""
    return _entry(name)[0]


def describe(name):
    """The plain description of a domain, for the report."""
    return _entry(name)[1]


def apply(name, abt):
    """The rows of `abt` inside the named domain."""
    cond = condition(name)
    return abt.query(cond) if cond else abt


def names():
    """Every registered domain name."""
    return list(_DOMAINS)
