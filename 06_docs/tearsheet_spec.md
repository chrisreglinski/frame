# tearsheet_spec

The tearsheet is a reusable one-page model report. It renders one run of a betting model and knows
nothing about which model produced it. This spec describes the pieces and where the boundaries sit.

## Three entities: model, run, report

- **Model**: a definition with a stable identity. Its learning parameters (target, features,
  hyperparameters, domain) plus one hook that builds the estimator. Sybilla is a model. It carries
  no results, and the same model holds across years.
- **Run**: one instantiation of a model on a specific data snapshot and protocol (which seasons, the
  fold scheme, the buffer). A run produces out-of-fold predictions and a frozen summary. The same
  model on a different set of seasons is a different run.
- **Report**: a view over one run. It holds the section order, the prose and the styling, and it
  trains nothing.

## Input

The report takes a Run (`models.run.Run`). A run carries the model (for identity and the Overview),
the out-of-fold predictions frame, the buffer used and how it was chosen, the seasons, and the frozen
headline metrics.

The predictions frame (the match register) has one row per match: `model_p`, `implied`, `fair`
(devigged), `y`, plus passthrough meta (`season`, `league`, teams, `date`). Everything the report
draws is derivable from this register plus the model snapshot and the buffer.

## Building blocks

Computation lives in `evaluation/` and is format-agnostic (DataFrames and Series):

- `predictions.collect_predictions`: a model over folds to an out-of-fold predictions frame.
- `stats.portfolio_stats`: summary of any match subset under proportional (implied) staking. The
  p-value null takes the devigged probability, the staking numbers take the raw implied price.
- `stats.buffer_curve` and `stats.pick_buffers`: profit vs bet threshold, and reference buffers
  (`left`, `middle`, `right`, or a single `peak`) from the smoothed profit curve.
- `staking.kelly_bankroll`: a compounding Kelly bankroll bet by bet, with running peak and drawdown.
- `calibration.calibration_overlay`: equal-count reliability buckets of one predictor with a second
  predictor on the same bins.

Models live in `models/`:

- `_spec.Model`, `Domain`, `apply_domain`, `snapshot`.
- `sybilla.py`, one model per file.
- `registry.get`, look up a model by name.
- `run.run`, apply a model to data on a protocol and return a Run. Training lives here.

Presentation lives in `reporting/`:

- `panels`: chart recipes. Each takes a frame and returns a spec (data and axes) and computes nothing
  of its own.
- `render`: layout primitives and the browser-side chart engine.
- `tearsheet.render_tearsheet`: composes the sections, holds the prose, emits the HTML.

## The compute and render boundary

The match register is the atomic input. Everything visual is derived from it (plus the model snapshot
for identity and the chosen buffer). Derivation happens as late as it is cheap:

- **Scalar metrics** (ROI, profit, p-value, MDD, cagr) are derivable from the register, but a run
  freezes them as its summary and as the check for consistency against the register.
- **Static chart aggregates** (calibration buckets, curves, per-group tables) are computed at render
  time from the register, by the evaluation layer, and shaped into specs by panels. They are not
  frozen. A frozen aggregate is a second copy that can drift from the register, so the register plus
  the code is the safer source.
- **Interactive elements** (the bets table filtering and sorting, live aggregates, tooltips) ship the
  row-level slice to the browser and re-aggregate client-side. This computation lives in the render
  and in JS by necessity, because no precomputation covers every filter.

So persistence freezes the register plus the scalar summary. The report derives static specs from the
register. The browser does only the interactive re-aggregation.

## Model repository

Models live in code (`02_src/models/`). A model is a config of parameters plus one `make_model` hook.
The parameters are plain data, so a run freezes a snapshot of them into its manifest and the report
reads identity and settings from there without importing the model.

The buffer is a run argument, not a model parameter. Passed explicitly it is the operational buffer.
Left out, it is picked from the profit curve on the development seasons (`middle` for twin peaks,
`peak` for one). Whatever value is used is recorded on the run.

## Report structure

Sections, in order: Intro, Overview, Profitability, Calibration, Edge, Bankroll, Bets.

Two kinds of text:

- A **caption** is fixed for the report type and lives in the report.
- A **finding** is model-specific and travels on the model (empty until written).

Season counts in the prose are read from the run, so the copy states the right number rather than a
hardcoded value. A league set is named in prose through a small display map (`major` becomes "top
five European leagues"), with the set membership itself defined in `01_data/league_sets.yaml`.

**Slots.** The Overview variables and hyperparameters cards default to the model's own and can be
replaced by content passed to `render_tearsheet`. To withhold them from a public build, pass a
placeholder (for example `["hidden"]`). The report renders the content it is given and does not
interpret it, so a real feature named "hidden" is unaffected.

## Output target

- **Primary: a Claude Artifact**, a self-contained theme-aware HTML page, sections stacking
  vertically.
- **Secondary: a printable A4-portrait PDF** from the same page. The print stylesheet fits the width
  and keeps a section from splitting across pages (`page-break-inside: avoid`).

## Out of scope

Model-specific decisions (which subset, which fold is the holdout, how or whether to recalibrate for
staking) belong to the model and the run, never to the reporting module.

## Possible improvements

- **Walk-forward folds.** `run` fixes the fold scheme to leave-one-season-out. The evaluation layer
  also has expanding (walk-forward) folds, which are time-honest (train on the past, test on the next
  season) and answer the deployment question rather than the stability one. Making the fold scheme a
  run argument would expose both. It is not a pure drop-in, because the buffer-selection protocol
  assumes a single holdout and the CV prose would adapt to the scheme.
- **Declarative models.** Model parameters live in code today. Because every field except
  `make_model` is plain data, a model could move to a per-model YAML plus an estimator factory that
  maps a named estimator and its hyperparameters to an object. The run would then freeze a copy of
  the file. This trades the estimator flexibility of code for a fully declarative definition, at the
  cost of a factory that every estimator type passes through.
