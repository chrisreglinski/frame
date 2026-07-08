"""Model-check framework — the invariant scaffold behind every `model_checks/` notebook.

Every model check asks the same question in the same shape: *pick the matches we think are
mispriced, fit a bet-signal model honestly (train only), place bets on held-out data, and see
whether the edge survives three robustness gates.* Only three things change between checks:

    - the SPACE     — which feature columns describe the match (goals, shots, elo, ...)
    - the SEGMENTER — how we carve out the region we bet on (KMeans clusters, a rule, ...)
    - the MODEL     — the bet-signal estimator (LR+poly today, RF/XGBoost tomorrow)

Those three are the *experiment* and live in the notebook. Everything here is the *machinery*
that never changes: the ROI scoring, the cross-validation folds, the honest per-fold engine
(`collect_bets`), the three gates, and the summary + registry write.

Segmenter contract (defined in the notebook, consumed by `collect_bets`):

    segmenter(train) -> labeler          # fits its regions on the TRAIN fold only (no leakage)
    labeler(rows)    -> pd.Series        # a segment label per row; NaN means "do not bet here"

The betting region is simply the rows the labeler did not mark NaN. A KMeans segmenter labels a
row by its (train-profitable) cluster; a rule segmenter labels the rows a boolean rule selects.

Model contract: any sklearn-style estimator with `.fit(X, y)` and `.predict_proba(X)`.

This module is the source of truth for the validation protocol — there is no separate versioned
spec. The prose companion (pipeline, gate roles, why season folds not random CV) is
`06_docs/model_check.md`; if the criteria change here, recompute every logged run.
"""
import numpy as np
import pandas as pd

from evaluation.results_registry import upsert


# --- Scoring -----------------------------------------------------------------
# ROI under implied-probability staking: on each bet you stake the implied probability and collect
# 1.0 on a win. ROI = (won - staked) / staked measures whether an edge exists, independent of bet
# size. Positive ROI = the outcome happened more often than the price we paid for it.
def pooled_roi(implied, won):
    staked = implied.sum()
    return (won.sum() - staked) / staked if staked > 0 else np.nan


# The same ROI, but computed within each group (per season, per league, per league x season cell).
# We sum stakes and wins first, then divide — this avoids groupby.apply, whose behaviour drifts
# across pandas versions.
def roi_grouped(bets, keys):
    totals = bets.groupby(keys).agg(won=("won", "sum"), staked=("implied", "sum"))
    return (totals["won"] - totals["staked"]) / totals["staked"]


# --- Cross-validation folds --------------------------------------------------
# A fold is a triple (train_mask, test_mask, fold_name) of boolean masks aligned to `matches`.
# The season is the unit of risk in football (same clubs, same promotion structure), so we never
# shuffle across it — we split whole seasons.

# Leave-one-season-out: each season is the held-out test set once, trained on the other three.
# Not time-honest (it may train on the future), but that is fine — LOSO is a stability check, and
# with only four seasons it is the only split with enough folds and training variety to carry a verdict.
def season_folds(data):
    for season in sorted(data["season"].unique()):
        yield ~(data["season"] == season), data["season"] == season, season


# Walk-forward (expanding): train only on earlier seasons, test on the next one. Time-honest, so it
# checks the edge would have survived real deployment. Starts at two training seasons on purpose —
# a single-season-trained model cannot tell signal from that season's quirks.
def expanding_folds(data, min_train_seasons=2):
    seasons = sorted(data["season"].unique())
    for split in range(min_train_seasons, len(seasons)):
        yield data["season"].isin(seasons[:split]), data["season"] == seasons[split], seasons[split]


# --- The honest per-fold engine ---------------------------------------------
# collect_bets walks a set of folds and, in each one, does the whole pipeline the honest way: fit
# the segmenter and the model on TRAIN only, then place bets on TEST. It returns one row per bet,
# with everything the gates need — nothing about how the bet was chosen leaks past this function.
#
# Empty-region folds are skipped. The returned frame has columns [implied, won, league, season, fold].
def collect_bets(matches, folds, *, features, outcome, implied, segmenter, make_model, buffer):
    placed = []
    for train_mask, test_mask, fold_name in folds:
        train, test = matches[train_mask], matches[test_mask]

        # 1. The segmenter fits its region on train only, then labels every row. A NaN label means
        #    "outside the region" — those rows are not bet on. This is where cluster boundaries and
        #    train-profitability selection happen, all without ever seeing the test fold.
        labeler = segmenter(train)
        train_region = labeler(train).notna()
        test_region = labeler(test).notna()
        if not test_region.any():
            continue

        # 2. The bet-signal model learns on the train region only.
        model = make_model()
        model.fit(train[train_region][features], train[train_region][outcome].astype(int))

        # 3. Place bets on the test region. Bet only where the model's probability beats the market's
        #    implied probability by `buffer` — a margin of caution, so we never bet on a hair's edge.
        region = test[test_region].copy()
        positive_col = list(model.classes_).index(1)
        region["model_prob"] = model.predict_proba(region[features])[:, positive_col]
        region["implied"] = region[implied]
        region["won"] = region[outcome].astype(int)
        region["fold"] = fold_name

        is_bet = region["model_prob"] > region["implied"] + buffer
        placed.append(region.loc[is_bet, ["implied", "won", "league", "season", "fold"]])

    if not placed:
        return pd.DataFrame(columns=["implied", "won", "league", "season", "fold"])
    return pd.concat(placed, ignore_index=True)


# --- The three gates ---------------------------------------------------------
# Each gate is a plain question asked of the `bets` frame. They do not know how the bets were made.

# GATE 1 — time robustness (LOSO). "Is this more than one lucky season?" Requires the per-season
# edge to be positive in at least 3 of the 4 seasons.
def gate1_loso(bets):
    per_season = roi_grouped(bets, "fold")
    passed = int((per_season > 0).sum()) >= 3
    return passed, per_season


# GATE 2 — league robustness. "Is this more than one lucky league?" Remove each league in turn; the
# league whose removal hurts most is the strongest. The pooled edge must stay positive without it.
def gate2_drop_best_league(bets):
    without_league = {
        league: pooled_roi(bets[bets["league"] != league]["implied"],
                           bets[bets["league"] != league]["won"])
        for league in bets["league"].unique()
    }
    best_league = min(without_league, key=without_league.get)  # removal hurts most => the best league
    passed = without_league[best_league] > 0
    return passed, without_league, best_league


# GATE 3 — walk-forward floor. "Would it have worked live?" Trained on the past only. Fails if any
# test season is clearly negative (< -0.05 ROI) or if both test seasons are negative.
def gate3_walk_forward(walk_bets):
    if walk_bets.empty:
        return False, pd.Series(dtype=float)
    per_fold = roi_grouped(walk_bets, "fold")
    passed = bool((per_fold >= -0.05).all()) and not bool((per_fold < 0).all())
    return passed, per_fold


# --- Orchestration -----------------------------------------------------------
# evaluate() runs the whole check once and returns every number, but neither prints nor logs. Use it
# directly when comparing variants in a notebook (e.g. running vs final boundaries), where you want
# the results but do NOT want to overwrite the registry row for each variant.
def evaluate(matches, *, features, outcome, implied, segmenter, make_model, buffer):
    kwargs = dict(features=features, outcome=outcome, implied=implied,
                  segmenter=segmenter, make_model=make_model, buffer=buffer)
    bets = collect_bets(matches, season_folds(matches), **kwargs)
    walk = collect_bets(matches, expanding_folds(matches), **kwargs)

    if bets.empty:
        return dict(bets=bets, walk=walk, empty=True)

    g1, per_season = gate1_loso(bets)
    g2, without_league, best_league = gate2_drop_best_league(bets)
    g3, wf = gate3_walk_forward(walk)
    return dict(
        bets=bets, walk=walk, empty=False,
        g1=g1, g2=g2, g3=g3,
        per_season=per_season, without_league=without_league, best_league=best_league, wf=wf,
        pooled=pooled_roi(bets["implied"], bets["won"]),
        spread=roi_grouped(bets, ["league", "season"]).std(),  # how concentrated/noisy the edge is
    )


_GREEN, _RED, _RESET = chr(27) + "[92m", chr(27) + "[91m", chr(27) + "[0m"
def _paint(ok):
    return f"{_GREEN}PASS{_RESET}" if ok else f"{_RED}FAIL{_RESET}"

def _verdict(ok):
    return "PASS" if ok else "FAIL"


# run_check() is the deployed-config entry point: run the check, print the three gates at a glance,
# and upsert this notebook's one-line row into model_checks/results.md. Call it exactly once per
# notebook (on the config you are actually logging); use evaluate() for the exploratory comparisons.
def run_check(matches, *, features, outcome, implied, segmenter, make_model, buffer,
              source, space, target, abt_filter, segmentation, segment_filter):
    r = evaluate(matches, features=features, outcome=outcome, implied=implied,
                 segmenter=segmenter, make_model=make_model, buffer=buffer)

    if r["empty"]:
        print("SUMMARY: no bets placed under this pipeline")
        return r

    wf, wf_detail = r["wf"], (
        "  ".join(f"{fold}:{value:+.3f}" for fold, value in r["wf"].items())
        if not r["walk"].empty else "no bets"
    )
    print(f"SUMMARY ({source}, b365, group from caller)")
    print(f"  pooled ROI                       {r['pooled']:+.4f}  ({len(r['bets'])} bets)")
    print(f"  GATE 1  (LOSO >= 3/4 seasons)     {_paint(r['g1'])}  "
          f"(+{int((r['per_season'] > 0).sum())}/{len(r['per_season'])})")
    print(f"  GATE 2  (drop best league)        {_paint(r['g2'])}  "
          f"(drop {r['best_league']} -> {r['without_league'][r['best_league']]:+.4f})")
    print(f"  GATE 3  (walk-forward floor)      {_paint(r['g3'])}  ({wf_detail})")
    print(f"  diagnostic spread (std league x season ROI)   {r['spread']:.3f}")

    upsert(source=source, space=space, target=target, abt_filter=abt_filter,
           segmentation=segmentation, segment_filter=segment_filter,
           g1=_verdict(r["g1"]), g2=_verdict(r["g2"]), g3=_verdict(r["g3"]),
           result=f"pooled {r['pooled']:+.3f} ({len(r['bets'])} bets)", roi_std=f"{r['spread']:.3f}")
    print(f"\nlogged to model_checks/results.md: {source}")
    return r
