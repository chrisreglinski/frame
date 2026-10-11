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

A space is a defined group of dimensions that belong together by content (goals for and against for
both sides, actual and implied points, xG, elo). Models are built on spaces and their combinations
by default, and may add single variables outside any space as `extra`. Spaces are a named subset of
contract columns. The full definition and the catalogue are in spaces_spec.

## Model definition

A model is a data record, not code. `04_models/<problem>/<solution>/params.yaml` holds:

- **target**: the outcome column (e.g. draw),
- **domain**: leagues (a set from `01_data/league_sets.yaml`) and a named domain (see Domains),
- **spaces**: the feature spaces the model uses,
- **extra**: single variables outside any space (optional),
- **estimator**: a name resolved by a factory,
- **hyperparameters**: the canonical set (from the winning run),
- **canonical_run**: the run-id whose settings are canonical.

The estimator factory maps an estimator name plus hyperparameters to a fresh sklearn-style object. It
is extensible: a new estimator type is one registration, the model records stay data. A one-off custom
construction can still fall back to a code builder.

`02_src/models/` holds only the mechanism: the spaces and domains registries, the estimator factory, the `Model`
type, `run`, the save layer, and the index view. No individual model lives in code.

## Domains

A domain is the slice of matches a model trains and scores on: a row condition over ABT columns,
registered under a name in `02_src/models/domains.py`, the same way spaces name groups of columns.
Each domain also carries a plain description, which the report shows. Registered now:

- `core`: the season in full swing. Fall without its first round, all of winter, and spring without
  the last round. The opening phase is still noisy form, and the last round often has nothing at
  stake. The default domain for exploration.
- `after_gw8`: both teams past their eighth league match (Sybilla's domain).
- `all`: every match.

## Identity and naming

A model's identity is target + leagues + domain + estimator + spaces + extra variables. Not the
hyperparameters. The identity renders as a web-style slug in two parts, computed from these fields on
load rather than written by hand:

- **problem**: `draw-major-core`, the outcome, the league set and the domain. Models with the same
  problem are comparable.
- **solution**: `xgb-goals_foragst+points` or `xgb-goals_foragst+points-d5b3b3`, the estimator, the
  spaces and, when there are extra variables, a hash of them.

The full slug is the two joined, `draw-major-core-xgb-goals_foragst+points-d5b3b3`. Listings may show
the parts separately or together. `-` separates the parts, `+` joins spaces. The target part is the
target column without its `t_` prefix and `_flg` suffix. Extra variables appear as a six-character
hash of their sorted list, like the hash in a run-id: the slug only signals that there are extras and
tells two sets apart, while the list itself is in `params.yaml` and in every run's snapshot.

A model may also get a human name (Sybilla). A name marks a promising model, its absence marks one
reviewed but not promoted. The name is a label in `params.yaml` and never changes where the model is
stored. Names are unique across the whole repository: there cannot be two Sybillas, even in different
problems.

## Runs

A model has one or more runs, varying by data (seasons) or hyperparameters. A run-id is the season
span plus a short config hash:

`2223-2526-a1b2c3`

Models are stored by problem and solution, `04_models/<problem>/<solution>/`, with `params.yaml` and
`runs/<run-id>/`. A run folder holds:

- **`results.yaml`**: a frozen snapshot of the model definition, the hyperparameters actually used,
  the chosen buffer and how it was chosen, and the headline metrics,
- **`predictions.parquet`**: the out-of-fold register plus the model's feature columns, a thin
  self-contained ABT (match_id, date, season, league, teams, model_p, implied, fair, y, and the
  features),
- **`report.html`**: the rendered tearsheet for this run, and `report-compact.html`, the internal
  summary (see tearsheet_spec).

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

There is no separate registry store. The registry is a generated view over the files: it lists
`04_models/*/*/params.yaml` for every reviewed model, reads the `results.yaml` of each model's best
run (highest profit), and renders one row per model.

Two worlds come from the same tree. The tree itself is the complete record: everything reviewed for a
problem sits under it, weak models included. The view puts the named models first, so the weak ones do
not distract, and then lists every model grouped by problem. `registry.table()` returns the rows as a
DataFrame, and `registry.write_index()` writes `04_models/index.html`, each row linking to its run's
compact report. Being generated, the view is never stale, and naming a model moves no files.

## Git and outputs

`04_models/` is gitignored for now, definitions and results alike. `05_reports/` tracks the reports
actually published, separate from each run's working `report.html`.

## Formats

`params.yaml` and `results.yaml` are both YAML, for one format across the model folder.
`predictions.parquet` is Parquet. (`params` is human-authored and `results` is generated, so a
role-split of YAML config plus JSON records would also be defensible, but uniform YAML is chosen for
legibility.)

## Status

Built. In place: the feature-space registry, the estimator factory, the `Model` as plain data
(features resolved from spaces), `run` returning a `Run`, the save layer (`params.yaml` plus per-run
folders with results.yaml, predictions.parquet and report.html), the params loader, named domains, the registry as a
view over `04_models/*/*/params.yaml` with the generated index (`registry.table`, `write_index`), and
the report rendering from a loaded run (store.load_run) so it is a pure visualization of a persisted
run. Sybilla lives at `04_models/draw-major-core/xgb-goals_foragst+points+impl_points-d5b3b3/`.

## Possible later

- **Walk-forward folds** as a run option (see tearsheet_spec).
- **MLflow** as a later swap for the file-based run store, behind the same save layer, once the
  experiments are many. The `Run` object is the seam, so the store is a thin adapter.
