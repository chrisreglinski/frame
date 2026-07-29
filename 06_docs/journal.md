# Journal

## 2026-07-29 — the market compresses home advantage toward the middle (5 truth-measures agree)

- Fitted the per-league home-field intercept (lhfa: expected p_home - p_away at equal strength) on
  gw>8, using FIVE independent strength axes as the zero-point: elo, and season-form points /
  impl_points / goals / shots-on-target diffs. They agree tightly (per-league std ~0.005) and span
  **italy 0.09 -> spain 0.20**. This is not an elo quirk — every strength measure sees the same
  league home-advantage ordering.
- Systematic detail: every form axis sits ~+0.01 ABOVE elo (impl_points highest). Likely because elo
  is the cleanest venue-neutral rating (Club Elo separates home advantage), while raw form averages
  carry residual home-advantage contamination (odds-derived impl_points most of all). So for the
  cleanest HA estimate, use the elo version.
- The MARKET's priced HA (fit mrkt_home_away_impl_diff on elo, same subset) is COMPRESSED to
  0.13-0.17 vs the truth's 0.09-0.20. Mispricing (market - true): **italy +0.040, france +0.030**
  (market overstates their weak home edge), **spain -0.021** (understates the fortress); england/
  germany ~right. France keeps showing up anomalous (also the weak league in the promoted/away bets).
- Tradeable hypothesis (for a forward-test, not proven): fade the home side in italy/france, back it
  in spain — where the market's HA error is largest and one-directional. Now backed by 5 agreeing
  truth-measures, not a single fit.
- Aside: the four season-form strength axes add essentially nothing over elo for predicting the
  result margin (all 0.83-0.96 correlated with elo, incremental R2 < 0.007, and worse early where
  form is small-sample noise). Not worth materializing as features; elo + market columns suffice.

## 2026-07-29 — late-season outcomes are less strength-determined (every rating, not just elo)

- Regressing the result margin (t_flg_diff) on a strength axis, R² drops from early to late season
  (elo: 0.20 gw1-8 -> 0.15 gw>8). First read this as "elo goes stale" — wrong.
- Controlling for the strength spread (|d|<200, since early season has more lopsided fixtures that
  inflate raw R²), **both** elo and the market's implied margin lose ~0.03 R² from gw1-8 to gw>8
  (elo 0.119->0.088, market 0.141->0.114). The market's edge over elo barely moves (+0.021->+0.026).
- So it is NOT elo-specific decay. **Late-season results are simply less determined by team strength
  — for any strength measure.** Plausible drivers: dead rubbers, rotation, secured/lost objectives,
  fixture congestion — noise no rating captures because it isn't about strength.
- Corollary for feature-fitting: elo is fine fit on all matches (it's if anything crisper early — the
  early-high R² was mostly the wider-spread composition, not elo quality). Form-window axes still fit
  on gw>8, but for the small-sample reason, not "elo is noisy early".

## 2026-07-29 — early-season mispricing, Italy's home advantage, and the elo→prob margin feature

Session mixed exploratory betting analysis with a feature-engineering outcome. All betting findings
below are **in-sample on the 4 major seasons** unless stated — noted for the record, not proven.

- **Home advantage differs sharply by league; the market doesn't price it per-league.** Strength-
  controlled HA (intercept of the elo→outcome fit, i.e. HA between equal teams) runs from **italy 0.099**
  up to **spain 0.188**, with eng/ger/fra ~0.12–0.14. The market carries one near-flat home premium per
  league across the whole season (implied HA barely moves by phase). This is a systematic gap, not noise.

- **Italy is the odd league — because its home advantage is the smallest, not because away teams win more.**
  Serie A has the FEWEST home wins and the MOST draws, not more away wins (early-season "away teams do well"
  was a perspective illusion — it's really excess draws). Same team gains ~half as much at home as in Spain
  (venue points gap 0.27 vs 0.54 ppg). Likely structural: old athletics-track stadiums, low attendance,
  defensive culture that "travels well". Why Italy looks *reversed* at season start: other leagues SUSPEND
  their large home advantage early (crowds not yet warm), Italy has little to suspend, so only the early
  draw-heavy mode shows.

- **The bookmaker's 1X2 skeleton is ~90% reconstructible from elo + a per-league home premium** (R²=0.90 on
  the implied H−A margin). The remaining ~10% (±10pp) is the market's private info; regressing outcome-
  residuals on price-residuals gave beta≈1.36 — the market slightly UNDER-weights its own extra signal.

- **Promoted away teams gw1-3 are overpriced by the market** (frozen rule: 1X, gw1-3, home odds >1.66 →
  ROI +16%, p=0.02, all 4 seasons and 5 leagues positive in major). It's the *promoted label*, not weakness:
  a placebo on the 3 lowest-elo incumbents REVERSED (they overperform). BUT it failed out-of-sample on minor
  (2nd divisions, −11%) and other (NL/PT, −6.5%) — likely top-5-only (money/crowd size). Frozen for an
  Aug-2026 forward-test at small stakes; needs a pre-set pass/fail threshold before the season.

- **Feature shipped: `teams_elo_diff_impl_{diff,tilt,lhfa}`** (matchup table). `diff` = g(d) = expected
  p_home−p_away from elo (the reconstruction above); splits additively into `tilt` (strength, odd part) and
  `lhfa` (per-league home-field advantage, even part — bakes the italy/spain gap straight into a column,
  strength-controlled). Cubic fit to realized outcomes, coefficients a fixed offline population fit stored in
  thresholds.json. The deferred `mrkt_home_away_impl_diff − teams_elo_diff_impl_diff` would isolate that ~10%
  residual where anomalies like the promoted-away edge live. First member of a possible family (goals / sot
  axes could get the same odd-part treatment).

## 2026-07-13 — PCA of inputs: nothing jumped out

- ran PCA over the numeric season-window features (landscape task). the space is low-dimensional (~5 axes,
  ~58% variance: favourite-tilt, level, tempo, tempo-mismatch, draw-balance) — confirms the correlation
  grouping, but nothing new or actionable surfaced. descriptive only, not a feature-engineering path.
- notebook was throwaway / distracting, deleted; correlation / varclus / vif notebooks stay.

## 2026-07-02 — other (first divisions) supports the draw edge; gate tooling + GATE 3

- ran the frozen draw pipeline on the **other** group (netherlands + portugal, both TOP FLIGHTS). match-draw:
  pooled +8.3% at the frozen k=5, GATE 1 3/4, walk-forward clean. team-draw pooled +22% but knob/map-fragile
  (spikes at types=4, flips negative on the final map at types=5, ~58 honest walk-forward bets) — do not
  anchor on it; match-draw +8% is the honest number.
- this REVISES the 2026-07-01 "top-5 only / does not generalize" downgrade. the split that matters is **tier**,
  not group: FIRST divisions (top-5 major + other's eredivisie/primeira) carry the draw edge; SECOND divisions
  (minor) do not. a structural boundary (market maturity / public money on favourites), not noise.
- GATE 2 (drop-best-league) is near-meaningless on a 2-league group: "drop the best of 2" = "is the single
  remaining league positive", which even a true small edge fails often from variance. its fail on other is
  discounted; on minor (5 leagues) it still carries weight.
- honest size is modest and netherlands-leaning (portugal -0.14), wide CI — real but small, clearest in first
  divisions. the one clean remaining test is forward 2627.
- TOOLING: the 3 model_checks notebooks (match/team draw, team home) now report GATE 1 / GATE 2 / GATE 3 as
  colored PASS/FAIL (green/red). GATE 3 promotes the walk-forward from a directional veto to a hard gate: it
  FAILS if any test season ROI < -0.05, or if both test seasons are negative (on major: match-draw PASS,
  team-draw PASS, team-home FAIL). each notebook refreshes its one-liner in `model_checks/results.txt`.
  GROUP is a parameter at the top of each notebook; the `_minor` duplicate notebooks were removed.
- DECISION DEFERRED: whether to extend the gated analysis systematically to minor / other is left for later.

## 2026-07-01 — draw signal does NOT generalize to second divisions (minor group)
- validation on the new minor group (second divisions england2/spain2/italy2/france2/germany2, 2223-2526,
  built via the group-parameterized pipeline). The exact same draw pipeline that passes on major FAILS every
  gate on minor, under both region structures:
  - match (4-var matchup): major +13.6% (G1 3/4, G2 pass, no veto)  ->  minor -5.3% (G1 1/3, G2 fail, VETO)
  - team (archetype grid): major +8.8% (G1 4/4, G2 pass)            ->  minor +0.7% (G1 2/4, G2 fail, VETO)
  match-draw even flips sign; home fails on both groups.
- this is the step-5 out-of-sample test on a NEW axis (leagues the template never saw). A genuinely
  behavioural, universal bias (draws under-bet, favourites over-backed) should appear in the softer minor
  markets, plausibly stronger — it does not appear at all. So the top-5 draw edge is most likely
  top-5-specific, or overfit to the 4 top-5 seasons, despite passing the major gates.
- VERDICT DOWNGRADE: the draw finding is NOT a confirmed universal edge. It survives time (LOSO) and league
  (drop-best) robustness *within* major, but fails cross-league generalization to second divisions. Treat as
  a low-confidence top-5-only pattern unless it reappears on genuinely fresh data.

## 2026-07-01 — draw value in "weak away vs strong home" matchups
- signal: draws underpriced when a clearly weak team (season goals_for low & goals_against high)
  meets a stronger side. strongly ASYMMETRIC — weak-AWAY carries it, weak-home ~null. mechanism:
  draws under-bet + clear (not crushing) home favourites over-backed (region: home wins ~61%,
  draw impl ~20% vs actual ~26%).
- pipeline: KMeans(4 goal feats) -> keep clusters with positive train draw-edge -> LR +
  PolynomialFeatures(deg2); bet draw where prob > mrkt_draw_impl + 0.02, restricted to region.
  honest per-fold (region + clusters fit on train only): passes v2 in aggregate — Gate 1 LOSO 4/4
  (~+16% b365), Gate 2 survives removing the best league (~+15% without germany), walk-forward no veto,
  robust across k=4..6.
- representation-invariant: KMeans == cat3q rule (for=low & agst=high) == distance-to-archetype.
  signal is a LUMP not a gradient; higher P(draw) != higher draw value.
- interactions matter, market impl does NOT: LR Polynomial beats plain; adding mrkt impl to the LR
  does not robustly help (poly+impl inflates 2324 but flips 2223 negative, more volatile).
- FRAGILITY: gates pass in aggregate, but the league x season breakdown is very noisy — individual cells
  swing wildly (england 2526 +0.80, spain 2425 -0.51, ~50 bets each). a passing aggregate rests on noisy
  pieces; any single league-season slice is a lottery. + meta-DoF (template chosen on these 4 seasons),
  4 seasons only. (correction: earlier "2425 dead" and "germany-carried / england-france negative" were
  wrong — they came from the region's raw edge, not the strategy's actual bets, which are positive in
  every league; Gate 2 passes.)
- map-invariant: drawing the cluster boundaries on running (point-in-time) vs final (end-of-season "true
  character") stats both pass every gate -> the signal does not depend on how the boundaries are drawn.
  running chosen for deployment: lower league x season variance, better walk-forward, and it is the state
  actually observable at bet time.
- verdict: passes v2 in aggregate and is map-invariant, but NOT proven — the per-cell breakdown is noisy
  and the whole template was chosen on these 4 seasons (meta-DoF). refines the 2026-06-17 "draws not
  profitable" note (global model). next: pull more leagues/divisions (real OOS); freeze spec ->
  strategies.md; forward-test 2627.

## 2026-07-01
- goals and shots-on-target total categorizations (global cat3q, team_season_final) differ
  sharply by league. italy and spain are dominated by low-total teams, germany and england
  by high-total teams. france is mixed on goals (23/29/22 low/med/high) but leans high on
  shots on target. "total" = for + against per match, so a "low-total" team is one whose
  matches are low-scoring overall, not one that scores little itself.

## 2026-06-17
- draws chosen based on predicted t_goals_diff == 0 by simple LR - not profitable
