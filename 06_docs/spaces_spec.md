# spaces_spec

What a feature space is, which spaces exist, and how they reach a model. How models are defined, run
and stored is in models_spec.

## What a space is

A space is a defined group of dimensions that belong together by content. Choosing one of them makes
the others meaningful too: if a model uses the home team's goals scored, the home team's goals
conceded and the away team's goals scored and conceded make just as much sense. So a space always
describes both sides the same way.

Building models on spaces and their combinations is the default way to explore: it shows which
families of features carry predictive signal, before any finer selection. Variables outside any space
are still allowed. A model lists them separately as `extra` (see below).

A space only names and groups columns. It transforms nothing: any derived quantity (a difference, a
total, a category) is built upstream as its own ABT column and defined in `06_docs/data.yaml`. So a
space is always a named subset of contract columns.

## Windows

A form space comes in three versions, one per window of `match_team_stats`:

| name | window |
|------|--------|
| `<space>` | `season` (expanding from the first match) |
| `<space>_r6` | `rolling6` |
| `<space>_r8` | `rolling8` |

A space with no window, such as elo, has a single version.

## Catalogue

The registry is `02_src/models/spaces.py`.

| space | columns (per side, `{w}` = window) | windowed |
|-------|------------------------------------|----------|
| `goals_foragst` | `{side}_{w}_goals_for_avg`, `{side}_{w}_goals_agst_avg` | yes |
| `points` | `{side}_{w}_points_avg`, `{side}_{w}_mp_impl_points_avg` | yes |
| `xg` | `{side}_{w}_xg_for_avg`, `{side}_{w}_xg_agst_avg` | yes |
| `elo` | `hmt_elo`, `awt_elo` | no |

Candidates not yet in the registry: `goals_totaldiff` (a rotation of `goals_foragst` into total and
difference), `shots_on_target_foragst`, and `impl` (the market's implied probabilities for home, draw
and away).

## Spaces and extra variables in a model

`params.yaml` lists a model's spaces and, separately, any extra variables:

```yaml
spaces: [goals_foragst, points]
extra: [mrkt_draw_impl]
```

`models.loader.load_model` resolves the spaces through `spaces.resolve` and appends the extra
variables, giving one ordered column list without duplicates. This is the only place spaces turn into
columns: `run` and the estimator factory see a plain column list. A saved run freezes that resolved
list, so a later change to a space never changes what an old run shows it was trained on.

The slug is built from the spaces, and extra variables add a short hash of their sorted list (see
models_spec):

```
draw-major-goals_foragst+points-xgb             spaces only
draw-major-goals_foragst+points-xgb-d5b3b3      spaces plus extra variables
```

`spaces.validate(spaces, abt.columns)` checks that every resolved column exists in a given ABT. It
matters for league sets without full source coverage: the `xg` and `elo` columns exist only in sets
built from the top flights.

## Adding a space

- Make sure its columns exist in `data.yaml` (and so in the ABT). If they do not, the contract
  changes first.
- A form space goes into `_WINDOWED` as per-side templates on `{side}` and `{window}`, and gets all
  three window versions for free. Anything else goes into `_FIXED` as a literal column list.
- A single column is not a space. It goes into a model's `extra`.
