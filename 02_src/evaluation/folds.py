"""Cross-validation folds for season-based betting evaluation.

A fold is a triple (train_mask, test_mask, fold_name) of boolean masks aligned to `data`.
The season is the unit of risk in football (same clubs, same promotion structure), so we never
shuffle across it — we split whole seasons. These generators are the general CV primitive consumed
by the model-check protocol and by prediction collection alike.
"""


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
