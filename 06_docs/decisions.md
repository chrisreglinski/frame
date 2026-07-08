## 2026-07-08 — Validation protocol: code is the source of truth (no versioning)
The strategy validation protocol now lives in code:
[`02_src/evaluation/model_check.py`](../02_src/evaluation/model_check.py) (folds, the honest per-fold
engine `collect_bets`, and the three gates), with a prose companion in
[`model_check.md`](model_check.md). There is **no versioned spec** ("v1/v2") to keep in sync — the
code is authoritative; if the criteria change we change the code and recompute every logged run.
Versioning would only be added if old and new results ever needed to coexist. The protocol is
space-independent: it applies to any `features` set, not only a registered `home_away_feature_space`.
The earlier dated protocol entries below are kept as history, not as the spec.

## 2026-07-06 — Home/away feature spaces: a data type + two registries (refines 2026-07-03)
A `home_away_feature_space` is just a named bundle of symmetric ABT columns (`home`,
`away`, optional derived like `draw_impl`) — a data type, not a transformer. No
transformations happen inside a space; input columns are transformed beforehand. No
dedicated code module.

Two registries live in `06_docs/spaces.md`:
- **space registry** — the catalogue of existing spaces.
- **exploration registry** — a log of models built on spaces (space, ABT filter,
  segmentation, segment filter, LOSO / drop-best-league / walk-forward results); the
  `space` column links back to the space registry.

Supersedes the 2026-07-03 specifics below: no `02_src/spaces` module and no on-the-fly
`coordinates` / transform layer.

## 2026-07-03 — Adopt the "analysis space" abstraction for match exploration
Match exploration (segmentation → profitability → model) is organized around
*analysis spaces*: each embeds a match in a 2D/2×2D home-vs-away coordinate system
(`goals_foragst`, `shots_on_target`, `elo`, `points`, `impl`, learned `strength`).
Full description in `06_docs/spaces.md`.

**Rationale.** The existing segmentation notebooks are all instances of one pattern.
Making `Space` a first-class declarative object + generic machinery makes new spaces
cheap and keeps results comparable across spaces — which is what the redundancy /
landscape work needs. Canonical transform `(home, away) → (diff = tilt, sum = level)`;
reduced vs rich spaces; the strength ladder.

**Boundaries.** Coordinate transforms live in the analysis layer (`02_src/spaces/`),
computed on the fly, and are NOT materialized into the `data.yaml` contract. New
rotations go to Space, not the contract. Existing materialized transforms
(`match_matchup_stats`, goals `diff`/`total`, …) may be redundant with Space for now;
cleanup is deferred.

**Non-breaking.** The abstraction sits on top of the current pipeline and evaluation;
old notebooks remain valid and are portable to the Space approach incrementally.

## 2026-07-01 — Validation protocol v2: add league dimension, tiered roles *(historical)*
Marks when the league dimension and tiered gate roles were adopted: LOSO as the time gate,
drop-best-league as a mild orthogonal gate, walk-forward as a floor, plus knob-robustness and
league×season concentration as directional/diagnostic support. The rationale (roles, why season
folds not random CV) now lives in [`model_check.md`](model_check.md); the criteria live in code.

## 2026-07-01 — Validation protocol: two-tier (walk-forward + leave-one-season-out) *(historical)*
The original two-tier framing — LOSO as the primary gate, walk-forward as a directional check,
random/shuffled CV rejected because it smears one lucky season across folds. Superseded by the
code-as-source-of-truth protocol above; rationale in [`model_check.md`](model_check.md).

**Leakage note (accepted, still standing).** The only cross-fold leak is in the categorical columns
(`cat2m`/`cat3q`): their bucket edges come from global `thresholds.json` computed over all seasons.
This is boundary-only (the bucketed value itself is point-in-time correct) and the threshold is
arbitrary anyway — recomputing it per training fold would only move the arbitrary cut elsewhere, not
remove it, and mean/quantiles barely shift across seasons. So thresholds stay global. If a strategy
that leans directly on categoricals ever passes the gates, recompute `build_thresholds(train_mask)`
before trusting its number — otherwise ignore.

## 2026-06-13 — Rolling windows: min_periods=1 + explicit early-season mask
Rolling aggregations use `min_periods=1` so a single missing raw value mid-season doesn't
produce NaN for the whole window. Early-season NaN is enforced separately: after all
computations, every rolling column is set to NaN where `game_number <= window_size`.
This keeps the two concerns independent and explicit.

## 2026-06-13 — ABT parametrization
`build_abt()` always produces the full wide table (all columns, all rows). No parametrization.
Filtering by rows (league, season, game_number threshold) and columns (features vs targets) is done downstream in notebooks. The full ABT is small enough to hold in memory, and slicing pandas is trivial.

## 2026-06-14 — Seasons scope: 2223–2526 only
Seasons before 2223 excluded because COVID-era matches (2020–2022) had no fans, which
measurably reduced home advantage and skewed result distributions — a different data-generating process.

## 2026-06-12 — match_id
`match_id` to be built as md5(league|season|hmt_name|awt_name).
the columns have to be in the input table and optionaly can be in (one) output table.