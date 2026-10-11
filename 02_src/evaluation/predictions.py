"""Run a bet-signal model across CV folds and collect its out-of-fold predictions.

This is preparation for evaluation, not evaluation itself: it produces one clean frame of
predictions (model probability + market price + actual outcome per match) that the scoring,
staking and reporting layers all consume. The fold scheme is injected via `folds`, so the same
function yields LOSO predictions or walk-forward predictions depending on what is passed.
"""
import pandas as pd


# Passthrough columns carried from the ABT into the predictions frame, renamed to short,
# schema-independent names so downstream code never touches raw ABT column names.
_META = {
    "match_id":   "match_id",
    "date":       "date",
    "league":     "league",
    "season":     "season",
    "home":       "hmt_name",
    "away":       "awt_name",
    "home_goals": "t_home_goals",
    "away_goals": "t_away_goals",
    "result":     "t_result",
}


def collect_predictions(matches, folds, *, features, outcome, implied, make_model):
    """Fit a fresh model on each fold's train side, predict its test side, stack the results.

    Returns one row per test match with: the passthrough meta columns, `implied` (market price),
    `model_p` (model probability of the outcome), `y` (actual outcome 0/1), and `fold`.
    One call per fold scheme — pass season_folds for LOSO, expanding_folds for walk-forward.
    The caller is responsible for handing in clean rows (filtered, no NaN in `features`).
    """
    carried = {out: src for out, src in _META.items() if src in matches.columns}
    pieces = []

    for train_mask, test_mask, fold_name in folds:
        train = matches[train_mask]
        test = matches[test_mask]

        model = make_model()
        model.fit(train[features], train[outcome].astype(int))

        positive_col = list(model.classes_).index(1)
        model_p = model.predict_proba(test[features])[:, positive_col]

        piece = test[list(carried.values())].rename(columns={src: out for out, src in carried.items()})
        piece["implied"] = test[implied].values
        piece["model_p"] = model_p
        piece["y"] = test[outcome].astype(int).values
        piece["fold"] = fold_name
        pieces.append(piece)

    return pd.concat(pieces, ignore_index=True)
