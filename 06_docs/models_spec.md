# models_spec

How models are defined, run, saved and tracked. The tearsheet report is specified in tearsheet_spec.
Where the two overlap, this document is the authority on models and the report doc defers to it.

## Three entities

- **Model**: a definition with a stable identity. What it learns (target, domain, feature spaces,
  estimator) plus its canonical hyperparameters. Carries no results, and the same model holds across
  years.
- **Run**: one instantiation of a model on a specific data snapshot and protocol (seasons, fold
  scheme, buffer). Produces out-of-fold predictions and a frozen summary. The same model on different
  seasons or hyperparameters is a different run.
- **Report**: a view over one run (see tearsheet_spec).

## Spaces

A space is a named, coherent group of feature columns (goals for and against, actual and implied
points, xG, elo, and the market draw anchor as its own space). Models are built on spaces and their
combinations, not on hand-picked columns. Spaces are defined once, derived from the data contract in
`06_docs/data.yaml`, so a space is a named subset of contract columns. Exploration is space-centric:
you check how a space, and combinations of spaces, perform, rather than picking variables ad hoc.

## Model definition

A model is a data record, not code. `04_models/<name|slug>/params.yaml` holds:

- **target**: the outcome column (e.g. draw),
- **domain**: leagues (a set from `01_data/league_sets.yaml`) and the gameweek-min filter,
- **spaces**: the feature spaces the model uses,
- **estimator**: a name resolved by a factory,
- **hyperparameters**: the canonical set (from the winning run),
- **canonical_run**: the run-id whose settings are canonical.

The estimator factory maps an estimator name plus hyperparameters to a fresh sklearn-style object. It
is extensible: a new estimator type is one registration, the model records stay data. A one-off custom
construction can still fall back to a code builder.

`02_src/models/` holds only the mechanism: the spaces registry, the estimator factory, the `Model`
type, `run`, the save layer, and the index view. No individual model lives in code.

## Identity and naming

A model's identity is target + leagues + spaces + estimator. Not the hyperparameters, and not the
gameweek filter, which is near-constant and lives in the file. The identity renders as a web-style
slug:

`draw-major-goals_foragst+points+anchor-xgb`

`-` separates the parts, `+` joins spaces. A model may also get a human name (Sybilla), which takes
precedence over the slug in file and folder names: the on-disk name is coalesce(name, slug). A human
name marks a promising model, its absence marks one reviewed but not promoted.

## Runs

A model has one or more runs, varying by data (seasons) or hyperparameters. A run-id is the season
span plus a short config hash:

`2223-2526-a1b2c3`

`04_models/<name|slug>/runs/<run-id>/` holds:

- **`results.yaml`**: a frozen snapshot of the model definition, the hyperparameters actually used,
  the chosen buffer and how it was chosen, and the headline metrics,
- **`predictions.parquet`**: the out-of-fold register plus the model's feature columns, a thin
  self-contained ABT (match_id, date, season, league, teams, model_p, implied, fair, y, and the
  features),
- **`report.html`**: the rendered tearsheet for this run.

The run is self-describing: it freezes the definition redundantly, so an old run shows exactly what
produced it, independent of later edits to params.

## Buffer

The buffer (bet threshold on edge) is a run value, not a model parameter. Passed explicitly it is the
operational buffer. Left out, it is picked from the profit curve on the development seasons (middle for
twin peaks, peak for a single one). Whatever value is used is recorded in the run.

## What gets saved

Searches over hyperparameters live in scratch files and are not kept, only the best run of a search
is saved. Every reviewed model is saved though, even a weak one: a params file plus one run with its
metrics and its parquet, so a report can always be generated. A weak model simply has no human name.
So the record of what was reviewed is the set of params files, and the promising subset is the ones
with names.

## Registry and view

There is no separate registry store. The registry is a generated view over the files: list
`04_models/*/params.yaml` for every reviewed model, read each model's canonical run's `results.yaml`
for its best metrics, and render one row per model. A first version prints a concise table on demand,
so it is never stale. Later versions can emit a markdown index or an HTML hub linking to each report.

## Git and outputs

`04_models/` is gitignored for now, definitions and results alike. `05_reports/` tracks the reports
actually published, separate from each run's working `report.html`.

## Formats

`params.yaml` and `results.yaml` are both YAML, for one format across the model folder.
`predictions.parquet` is Parquet. (`params` is human-authored and `results` is generated, so a
role-split of YAML config plus JSON records would also be defensible, but uniform YAML is chosen for
legibility.)

## Status

Mostly built. In place: the feature-space registry, the estimator factory, the `Model` as plain data
(features resolved from spaces), `run` returning a `Run`, the save layer (`params.yaml` plus per-run
folders), the params loader, and the registry as a view over `04_models/*/params.yaml`. Sybilla lives
at `04_models/sybilla/params.yaml`. Pending: the report reading a saved run instead of an in-memory
one, and the generated index view over models.

## Possible later

- **Walk-forward folds** as a run option (see tearsheet_spec).
- **MLflow** as a later swap for the file-based run store, behind the same save layer, once the
  experiments are many. The `Run` object is the seam, so the store is a thin adapter.
