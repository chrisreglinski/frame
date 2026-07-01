# Journal

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
