# tearsheet_spec

The tearsheet is a reusable one-page model report. It renders one saved run of a betting model and
knows nothing about which model produced it. How models are defined, run and stored is in models_spec,
which is the authority where the two overlap. This document covers the report. The three entities are
model, run and report: the first two live in models_spec, the report is here.

## Input

The report renders one saved run. It reads:

- **`results.yaml`**: a frozen snapshot of the model definition, the chosen buffer, and the headline
  metrics,
- **`predictions.parquet`**: the out-of-fold match register plus the model's feature columns.

The register has one row per match: `model_p`, `implied`, `fair` (devigged), `y`, plus meta
(`season`, `league`, teams, `date`). Everything the report draws is derivable from the register plus
the snapshot and the buffer.

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

Presentation lives in `reporting/`:

- `panels`: chart recipes. Each takes a frame and returns a spec (data and axes) and computes nothing
  of its own.
- `render`: layout primitives and the browser-side chart engine.
- `tearsheet.render_tearsheet`: composes the sections, holds the prose, emits the HTML.

Models, runs and their storage are specified in models_spec.

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
staking) belong to the model and the run, never to the reporting module. See models_spec.

## Possible improvements

- **Walk-forward folds.** `run` fixes the fold scheme to leave-one-season-out. The evaluation layer
  also has expanding (walk-forward) folds, which are time-honest (train on the past, test on the next
  season) and answer the deployment question rather than the stability one. Making the fold scheme a
  run argument would expose both. It is not a pure drop-in, because the buffer-selection protocol
  assumes a single holdout and the CV prose would adapt to the scheme.
