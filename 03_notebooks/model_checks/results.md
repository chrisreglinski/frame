# Exploration results registry

Replaces the old `results.txt`. One row per model run, keyed by `source` (a `model_checks/` notebook).
Written by `evaluation.results_registry.upsert(...)`. Convention: [`06_docs/spaces.md`](../../06_docs/spaces.md).
All runs use mrkt (market-consensus) odds.

## 3xPASS (8)

| source | space | target | abt filter | segmentation | segment filter | G1 | G2 | G3 | result | roi_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| goals_foragst_draw_kmeans_match_lr | goals_foragst | draw | major, gn>8, dropna | KMeans k=5 (match, running, 4D) | train-profitable clusters | PASS | PASS | PASS | pooled +0.183 (1003 bets) | 0.325 |
| goals_points_draw_xgboost_match | goals_foragst + points | draw | major, gn>8, dropna | none (bet all) | none | PASS | PASS | PASS | pooled +0.178 (755 bets) | 0.356 |
| goals_foragst_draw_xgboost_match | goals_foragst | draw | major, gn>8, dropna | none (bet all) | none | PASS | PASS | PASS | pooled +0.164 (673 bets) | 0.336 |
| combo_draw_startphase_xgboost | elo+impl_points+sot_foragst | draw | major, dropna (no gn>8) | season_3phase == start | start (depth=2) | PASS | PASS | PASS | pooled +0.142 (448 bets) | 0.459 |
| points_draw_kmeans_match_lr | points | draw | major, gn>8, dropna | KMeans k=8 (match, running, 5D) | train-profitable clusters | PASS | PASS | PASS | pooled +0.120 (1012 bets) | 0.176 |
| points_draw_xgboost_match | points | draw | major, gn>8, dropna | none (bet all) | none | PASS | PASS | PASS | pooled +0.113 (870 bets) | 0.228 |
| goals_totaldiff_draw_kmeans_match_lr | goals_totaldiff | draw | major, gn>8, dropna | KMeans k=5 (match, running, 4D) | train-profitable clusters | PASS | PASS | PASS | pooled +0.111 (987 bets) | 0.354 |
| elodelta_home_last_blanket | elo_delta | home | major, gw<=8, dropna | delta=market-lrr, 6 qcut buckets (train-fit), bet last bucket | last 1 of 6 (blanket) | PASS | PASS | PASS | pooled +0.082 (254 bets) | 0.241 |

## 2xPASS (4)

| source | space | target | abt filter | segmentation | segment filter | G1 | G2 | G3 | result | roi_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| elodelta_away_first_blanket | elo_delta | away | major, gw<=8, dropna | delta=market-lrr, 6 qcut buckets (train-fit), bet first bucket | first 1 of 6 (blanket) | PASS | PASS | FAIL | pooled +0.084 (260 bets) | 0.336 |
| elodelta_draw_middle_blanket | elo_delta | draw | major, gw<=8, dropna | delta=market-lrr, 6 qcut buckets (train-fit), bet middle (all but first & last) | middle of 6 (blanket) | FAIL | PASS | PASS | pooled +0.083 (1030 bets) | 0.281 |
| goals_foragst_draw_derbyregional_xgboost | goals_foragst | draw | major, gn>8, dropna | travel_distance_cat in {derby, regional} | derby+regional (depth=3) | PASS | FAIL | PASS | pooled +0.035 (261 bets) | 0.611 |
| combo_draw_startphase_histgb | elo+impl_points+sot_foragst | draw | major, dropna (no gn>8) | season_3phase == start | start (max_depth=2) | PASS | FAIL | PASS | pooled +0.018 (458 bets) | 0.377 |

## 1xPASS (5)

| source | space | target | abt filter | segmentation | segment filter | G1 | G2 | G3 | result | roi_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| goals_foragst_draw_kmeans_team_lr | goals_foragst | draw | major, gn>8, dropna | KMeans n=3 team archetypes (final) -> 3x3 grid | train-profitable cells | FAIL | PASS | FAIL | pooled +0.065 (1099 bets) | 0.265 |
| elo_draw_startphase_xgboost | elo | draw | major, dropna | season_3phase == start | start (depth=3) | FAIL | PASS | FAIL | pooled +0.034 (556 bets) | 0.328 |
| elo_draw_kmeans_match_lr | elo | draw | major, gn>8, dropna | KMeans k=2 (match, running, 2D) | train-profitable clusters | PASS | FAIL | FAIL | pooled +0.006 (801 bets) | 0.636 |
| elo_draw_startphase_lr | elo | draw | major, dropna | season_3phase == start | start | FAIL | FAIL | PASS | pooled -0.014 (394 bets) | 0.509 |
| goals_foragst_draw_derbyregional_lr | goals_foragst | draw | major, gn>8, dropna | travel_distance_cat in {derby, regional} | derby+regional | FAIL | FAIL | PASS | pooled -0.034 (246 bets) | 0.521 |

## 0xPASS (12)

| source | space | target | abt filter | segmentation | segment filter | G1 | G2 | G3 | result | roi_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| combo_draw_startphase_rf | elo+impl_points+sot_foragst | draw | major, dropna (no gn>8) | season_3phase == start | start (max_depth=4) | FAIL | FAIL | FAIL | pooled +0.061 (435 bets) | 0.321 |
| goals_foragst_draw_startphase_xgboost | goals_foragst | draw | major, dropna (no gn>8) | season_3phase == start | start (depth=3) | FAIL | FAIL | FAIL | pooled +0.032 (463 bets) | 0.456 |
| goals_foragst_draw_midweek_lr | goals_foragst | draw | major, gn>8, dropna | day_of_week_cat == midweek | midweek | FAIL | FAIL | FAIL | pooled -0.017 (166 bets) | 0.721 |
| goals_foragst_draw_favourite_best_lr | goals_foragst | draw | major, gn>8, dropna | mrkt_favourite | best train-edge segment | FAIL | FAIL | FAIL | pooled -0.047 (483 bets) | 0.454 |
| shots_on_target_foragst_draw_kmeans_match_lr | shots_on_target_foragst | draw | major, gn>8, dropna | KMeans k=5 (match, running, 4D) | train-profitable clusters | FAIL | FAIL | FAIL | pooled -0.048 (574 bets) | 0.316 |
| goals_foragst_draw_startphase_lr | goals_foragst | draw | major, dropna (no gn>8) | season_3phase == start | start | FAIL | FAIL | FAIL | pooled -0.066 (327 bets) | 0.500 |
| elo_draw_derbyregional_xgboost | elo | draw | major, gn>8, dropna | travel_distance_cat in {derby, regional} | derby+regional (depth=3) | FAIL | FAIL | FAIL | pooled -0.091 (255 bets) | 0.733 |
| goals_foragst_home_kmeans_team_lr | goals_foragst | home | major, gn>8, dropna | KMeans n=3 team archetypes (final) -> 3x3 grid | train-profitable cells | FAIL | FAIL | FAIL | pooled -0.091 (474 bets) | 0.345 |
| goals_foragst_draw_midweek_xgboost | goals_foragst | draw | major, gn>8, dropna | day_of_week_cat == midweek | midweek (depth=3) | FAIL | FAIL | FAIL | pooled -0.118 (170 bets) | 0.852 |
| elo_draw_derbyregional_lr | elo | draw | major, gn>8, dropna | travel_distance_cat in {derby, regional} | derby+regional | FAIL | FAIL | FAIL | pooled -0.208 (119 bets) | 1.143 |
| elo_draw_midweek_lr | elo | draw | major, gn>8, dropna | day_of_week_cat == midweek | midweek | FAIL | FAIL | FAIL | pooled -0.326 (95 bets) | 0.904 |
| elo_draw_midweek_xgboost | elo | draw | major, gn>8, dropna | day_of_week_cat == midweek | midweek (depth=3) | FAIL | FAIL | FAIL | pooled -0.410 (161 bets) | 0.634 |
