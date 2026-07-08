# Model-check framework

How a betting strategy is validated in this project. The **source of truth is the code**:
[`02_src/evaluation/model_check.py`](../02_src/evaluation/model_check.py). This document is the prose
companion — it explains what the code does and why. There is **no protocol versioning** (no "v1/v2"):
if the criteria change, we change the code and recompute every logged run. Versioning would only be
introduced if we ever needed old and new results to coexist.

## What a model check is

Every check asks the same question in the same shape:

> *Pick the matches we think are mispriced, fit a bet-signal model honestly (train only), place bets
> on held-out data, and see whether the edge survives three robustness gates.*

The pipeline, per cross-validation fold:

1. **ABT filtering** — restrict to the matches in scope (e.g. both teams past game 8, drop rows
   missing the needed columns).
2. **Segmentation** — carve out the region we bet on (a KMeans cluster, a grid cell, or a plain rule).
   Fit on the training fold only.
3. **Model fit** — train the bet-signal model on the training rows of that region only.
4. **Bet** — on the held-out rows of the region, bet where the model's probability beats the market's
   implied probability by `buffer`.
5. **Score** — ROI under implied-probability staking, pooled across folds.

Everything that touches the target — region / cluster / threshold selection and the model fit — is
done on TRAIN only, per fold (construction hygiene, "rule #0"). Nothing about how a bet was chosen
leaks past `collect_bets`, which returns one row per bet.

## Space-independent

The framework takes an arbitrary `features` list — it does **not** require a registered
`home_away_feature_space`. A check can run on a catalogued space (`goals_foragst`, `elo`, …), on a
single variable, or on any **mixed set of ABT input columns**. Spaces are one convenient way to name
a feature set; the validation protocol applies to any of them.

## The two plug-ins (defined per notebook)

The invariant machinery lives in the framework; the parts that change between checks are defined in
the notebook and passed in:

- **Segmenter** — `segmenter(train) -> labeler`, `labeler(rows) -> Series` where `NaN` means "do not
  bet here". Fits its region on train, then labels any rows. Two flavours seen so far: *profitable*
  (KMeans clusters / archetype-grid cells kept where the target beat the market on train) and
  *rule-based* (region defined by an existing column, e.g. the best `mrkt_favourite` segment).
- **Model** — any sklearn-style estimator with `.fit(X, y)` and `.predict_proba(X)`. Today it is
  polynomial features + standardisation + logistic regression; random forest / XGBoost drop in by
  swapping the factory.

## The three gates

Implemented in `model_check.py` (`gate1_loso`, `gate2_drop_best_league`, `gate3_walk_forward`); the
exact thresholds there are authoritative. The season is the unit of risk (same clubs, same promotion
structure), so folds split whole seasons — never random CV, which would smear one lucky season across
all folds and hide exactly the risk we care about.

- **GATE 1 — time robustness (LOSO).** Leave one season out, each season the test set once, trained
  on the other three. Requires the edge positive in **≥3 of 4** seasons. Carries the verdict — with
  only four seasons it is the only split with enough folds and training variety to bear weight.
- **GATE 2 — league robustness (drop best league).** Remove each league in turn; the pooled edge must
  stay positive without its single best league. Catches the "one-league bet". Deliberately mild (not
  leave-one-league-out ≥4/5, which would reject on variance alone); orthogonal to GATE 1, so two mild
  gates on independent axes are fine.
- **GATE 3 — walk-forward floor (time-honest).** Expanding window, trained on earlier seasons only
  (`min_train=2`). Fails if any test season is clearly negative (< **-0.05** ROI) or if both test
  seasons are negative. Deployment realism.

**Directional / diagnostic (not hard gates), reported in the notebooks:**

- **Knob robustness** — sweep the arbitrary knobs (cluster count `k`, `buffer`). An edge that only
  appears at one setting is fragile. Representation-invariance (same signal under an independent
  definition, e.g. `running` vs `final` boundaries) is convergent support.
- **league × season breakdown** — how concentrated / noisy the edge is at the finest granularity
  (`roi_std` in the registry). A passing aggregate resting on a few wild cells is weak.

**Decision rule.** Promising = GATE 1 ≥3/4 **and** GATE 2 survives dropping its best league **and**
GATE 3 does not veto; knob-robustness and low concentration as support. The number stays optimistic
(the template itself was chosen on these seasons — meta-DoF) until confirmed on genuinely fresh data
(new leagues / a forward season).

## Entry points

- `evaluate(matches, *, features, outcome, implied, segmenter, make_model, buffer)` — runs the whole
  check and returns every number; prints and logs nothing. Use it to compare variants in a notebook.
- `run_check(matches, *, …, source, space, target, abt_filter, segmentation, segment_filter)` — the
  deployed-config entry point: prints the three gates and upserts this notebook's one-line row into
  [`03_notebooks/model_checks/results.md`](../03_notebooks/model_checks/results.md). Call it once per
  notebook, on the config you are logging.

Registry columns and conventions: [`06_docs/spaces.md`](spaces.md).

## Notebook naming

A `model_checks/` notebook name reads off the same four axes as its dedicated cells, in order, joined
by single underscores:

```
{space}_{target}_{segmenter}_{model}
```

All four slots are always present (use `none` for a trivial one, e.g. a bet-everything segmenter).
Within a slot the token may itself contain underscores; the slots are positional, read against this
schema, not by counting underscores.

- **space** — a registry name (`goals_foragst`, `elo`) or an ad-hoc name for a mixed feature set.
- **target** — `draw`, `home`, `away`, `over25`, …
- **segmenter** — `kmeans_match`, `kmeans_team`, `favourite_best`, `rule_<x>`, `none`.
- **model** — `lr`, `rf`, `xgb`, `tree`, `rulefit`.

Examples: `goals_foragst_draw_kmeans_team_lr`, `goals_foragst_home_kmeans_team_lr`,
`goals_foragst_draw_favourite_best_lr`. The filename is also the `source=` key in `run_check` and the
row key in `results.md`, so the three always match.
