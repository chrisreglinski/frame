# tearsheet_spec

Specification of the tearsheet — a reusable one-page model report. It works for **any** betting
model that (a) runs on some subset of the ABT and (b) can be split by season. It knows nothing
about which model, subset, segment or staking rule is used — those are the caller's decisions.

## Input

A predictions frame from `evaluation.predictions.collect_predictions`: one row per match with
`implied`, `model_p`, `y`, plus passthrough meta (`season`, `league`, teams, goals, `date`). The
caller chooses the ABT subset, features, model and fold scheme (LOSO or walk-forward) and hands in
the frame; the report only reads it.

## Building blocks

Computation lives in `evaluation/` and is format-agnostic (returns DataFrames/Series):

- `predictions.collect_predictions` — model × folds → out-of-fold predictions frame.
- `stats.portfolio_stats` — summary of any match subset (n, staked, wins, profit, ROI, hit rate,
  breakeven, one-sided p-value) under proportional (implied) staking.
- `stats.buffer_curve` / `stats.pick_buffers` — profit vs bet threshold, and reference buffers
  (`left`/`middle`/`right`) from the local maxima of the smoothed profit curve.
- `calibration.reliability_curve` — per-bucket predicted probability vs observed frequency.

Presentation lives in `reporting/` (`panels`, `render`) and only draws what the blocks compute.

## Sections

1. **header** — model identity + headline numbers.
2. **profit vs buffer** — the profit curve with the reference buffers marked, beside per-group
   breakdowns (per season, per league) from `group_stats`.
3. **calibration** — reliability / calibration panels (model and market), over all matches and
   over the bet matches.
4. **bankroll** — a Kelly bankroll curve bet-by-bet in chronological order, with drawdown. *(to build)*
5. **bets listing** *(optional)* — the placed bets, one row per bet.

## Buffer reference points

`pick_buffers` returns the local maxima of the smoothed profit curve — `left` (lowest buffer),
`right` (highest), and `middle` (their midpoint). Which one to operate on is the caller's choice.

## Output target

- **Primary: a Claude Artifact** — a self-contained HTML page, theme-aware, sections stacking
  vertically.
- **Secondary: a printable A4-portrait PDF** — from the artifact or separately. Fit the page
  width; keep a section from splitting across pages (`page-break-inside: avoid`).

Author panels as composable blocks (SVG for charts, HTML for tables) so they compose into the
artifact and print cleanly.

## Out of scope

Model-specific decisions — which ABT subset, which segment to include or exclude, which fold is
the holdout, how (or whether) to recalibrate for staking — belong to the notebook/driver that uses
the report, never to the reporting module.
