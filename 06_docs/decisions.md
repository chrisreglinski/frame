## 2026-07-01 — Validation protocol v2: add league dimension, tiered roles
Supersedes the tiering in the two-tier note below; its rationale (why not random CV, the
leakage/threshold note) still holds. Each element has an explicit role so hard gates are not
over-multiplied.

**0. Construction (hygiene).** Everything that touches the target — region / threshold / cluster
selection, model fit — is done on TRAIN only, per fold. Evaluate on the held-out fold via
`portfolio_roi`; pool across folds.

**1. GATE — LOSO (time).** 4 season folds, each season once as test, trained on the other 3.
Requires >=3/4 positive. Carries the verdict; covers season concentration (an outlier season
shows up in the per-fold detail).

**2. GATE — league (mild).** Pooled edge must survive removing its single best league (stays
positive without its top league). Catches the "one-league bet" failure. Deliberately NOT
leave-one-league-out >=4/5 — a second demanding gate over 5 leagues would reject on variance
alone. Orthogonal to LOSO (league axis != time axis), so two mild gates are fine; two demanding
ones are not.

**3. DIRECTIONAL (may veto, never pass on their own).**
- Walk-forward (expanding, `min_train=2`): time-honest deployment realism; vetoes if a fold is
  clearly negative while LOSO is positive.
- Knob robustness: the edge must survive a sweep of the arbitrary knobs (k, buffer); if it only
  appears at one setting it is fragile. Representation-invariance (same signal under independent
  definitions) is convergent support.

**4. DIAGNOSTIC (always reported, never a gate).** league x season breakdown
(`portfolio_breakdown`); meta-DoF and multiple-comparisons count (discount the number
accordingly); n_bets per fold and % of population bet (power + calibration sanity).

**5. CONFIRMATION (only after the spec is frozen — the sole real proof).** New leagues / divisions
(genuine OOS on the league axis the template never saw) -> forward season 2627 -> closing-line
value if obtainable.

**Decision rule.** promising = LOSO >=3/4 positive AND pooled survives dropping its best league
AND walk-forward does not veto; knob-robustness and low league x season concentration as support.
The number stays optimistic (meta-DoF) until confirmed on step-5 fresh data.

## 2026-07-01 — Validation protocol: two-tier (walk-forward + leave-one-season-out)
Strategy evaluation uses two complementary tiers, reported side by side. Random /
shuffled CV is rejected outright.

**Tier 1 — leave-one-season-out (LOSO), the primary gate.** 4 folds, each season once as
test, trained on the other 3 (three seasons of variety in every fold). Report ROI per
season + spread. This carries the verdict. It directly answers "was 2526 just favourable?":
if only 2526 is positive and the rest are flat/negative, the edge is season-specific noise.
LOSO is *not* time-honest (it trains on the future) — that is fine, it is a stability
diagnostic, not a profit estimate. With only 4 seasons LOSO is the only tier with enough
folds (4) and enough training variety (3 seasons each) to bear statistical weight.

**Tier 2 — walk-forward (expanding), a directional check.** `min_train=2`, so training
never rests on a single season:
- train `2223+2324` → test `2425`
- train `2223+2324+2425` → test `2526`

Time-honest (never trains on the future), so it confirms the edge survives realistic
deployment — still broader than a single `train ≤2425 / test 2526` split. Training starts
at 2 seasons on purpose: a single-season-trained model is one draw from the season
distribution and cannot separate time-invariant signal from that season's quirks, so its
ROI is untrustworthy. That leaves only 2 folds — too few to be a statistical gate, so WF is
directional: it can veto (a fold clearly negative contradicts LOSO) but cannot on its own
pass a strategy.

**Why not random CV.** It hides exactly the risk we care about: shuffling smears one lucky
season across all folds so season-luck becomes undetectable. It also trains on late-season
matches to predict early-season ones (mild within-season leak). Tighter confidence
intervals, but bought by discarding realism. The season is the natural unit of risk in
football (same clubs, same promotion structure) — CV must respect that boundary.

**Decision rule.** LOSO carries the decision: a strategy is "promising" only if ≥3/4 LOSO
folds are positive. Walk-forward is a consistency check, not a second gate — it may only
veto: if a WF fold is clearly negative (below a small tolerance) while LOSO is positive,
the LOSO result is suspect (an edge that only appears when the model is allowed to train on
the future). WF near zero or positive confirms.

**Leakage note (accepted).** The only cross-fold leak is in the categorical columns
(`cat2m`/`cat3q`): their bucket edges come from global `thresholds.json` computed over all
seasons. This is boundary-only (the bucketed value itself is point-in-time correct) and the
threshold is arbitrary anyway — recomputing it per training fold would only move the
arbitrary cut elsewhere, not remove it, and mean/quantiles barely shift across seasons. So
thresholds stay global. If a strategy that leans directly on categoricals ever passes both
tiers, recompute `build_thresholds(train_mask)` before trusting its number — otherwise
ignore.

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
`match_id` to be built as md5(league|season|home_team|away_team).
the columns have to be in the input table and optionaly can be in (one) output table.