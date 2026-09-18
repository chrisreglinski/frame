"""Feature spaces: named groups of ABT columns a model is built on.

A model is defined by a combination of spaces, not a raw column list. Windowed spaces (team form)
have a base version on the season window and rolling variants suffixed `_r6` / `_r8`. Non-windowed
spaces (elo, anchor) have a single version. Column names follow the data.yaml contract.

    columns("goals_foragst")            -> the four season goals for/against columns
    columns("xg_r6")                    -> the four rolling6 xg columns
    resolve(["goals_foragst", "anchor"]) -> their ordered, de-duplicated union
"""

_SIDES = ("hmt", "awt")

# suffix on a space name -> the ABT window token it maps to.
_WINDOWS = {"": "season", "_r6": "rolling6", "_r8": "rolling8"}

# Windowed families: per-side column templates on a {window} token.
_WINDOWED = {
    "goals_foragst": ["{side}_{window}_goals_for_avg", "{side}_{window}_goals_agst_avg"],
    "points":        ["{side}_{window}_points_avg", "{side}_{window}_mp_impl_points_avg"],
    "xg":            ["{side}_{window}_xg_for_avg", "{side}_{window}_xg_agst_avg"],
}

# Non-windowed spaces: fixed column lists.
_FIXED = {
    "elo":    ["hmt_elo", "awt_elo"],
    "anchor": ["mrkt_draw_impl"],
}


def _split_window(space):
    """Split a space name into its base and ABT window token (season unless _r6 / _r8 suffixed)."""
    for suffix, window in (("_r6", "rolling6"), ("_r8", "rolling8")):
        if space.endswith(suffix):
            return space[: -len(suffix)], window
    return space, "season"


def columns(space):
    """The ABT column names of one space, e.g. 'goals_foragst', 'points_r8', 'anchor'."""
    if space in _FIXED:
        return list(_FIXED[space])
    base, window = _split_window(space)
    if base in _WINDOWED:
        return [t.format(side=side, window=window) for side in _SIDES for t in _WINDOWED[base]]
    raise KeyError(f"unknown space {space!r}; known: {names()}")


def resolve(spaces):
    """Ordered, de-duplicated column union for a list of space names."""
    cols = []
    for space in spaces:
        for col in columns(space):
            if col not in cols:
                cols.append(col)
    return cols


def names():
    """Every defined space name: fixed spaces, plus each windowed family in its season, r6 and r8
    versions."""
    out = list(_FIXED)
    for base in _WINDOWED:
        out += [base, base + "_r6", base + "_r8"]
    return out


def validate(spaces, available):
    """Raise if any column resolved from `spaces` is missing from `available` (an ABT's columns)."""
    missing = [c for c in resolve(spaces) if c not in available]
    if missing:
        raise KeyError(f"columns not in the ABT: {missing}")
