# Exploration results registry

Replaces the old `results.txt`. One row per model run, keyed by `source` (a `model_checks/` notebook).
Written by `evaluation.results_registry.upsert(...)`. Convention: [`06_docs/spaces.md`](../../06_docs/spaces.md).
All runs use mrkt (market-consensus) odds.

| source | space | target | abt filter | segmentation | segment filter | G1 | G2 | G3 | result | roi_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| elo_draw_kmeans_match_lr | elo | draw | major, gn>8, dropna | KMeans k=2 (match, running, 2D) | train-profitable clusters | PASS | FAIL | FAIL | pooled +0.006 (801 bets) | 0.636 |
| goals_foragst_draw_favourite_best_lr | goals_foragst | draw | major, gn>8, dropna | mrkt_favourite | best train-edge segment | FAIL | FAIL | FAIL | pooled -0.090 (514 bets) | 0.365 |
| goals_foragst_draw_kmeans_match_lr | goals_foragst | draw | major, gn>8, dropna | KMeans k=5 (match, running, 4D) | train-profitable clusters | PASS | PASS | PASS | pooled +0.112 (1013 bets) | 0.350 |
| goals_foragst_draw_kmeans_team_lr | goals_foragst | draw | major, gn>8, dropna | KMeans n=3 team archetypes (final) -> 3x3 grid | train-profitable cells | FAIL | PASS | FAIL | pooled +0.065 (1099 bets) | 0.265 |
| goals_foragst_home_kmeans_team_lr | goals_foragst | home | major, gn>8, dropna | KMeans n=3 team archetypes (final) -> 3x3 grid | train-profitable cells | FAIL | FAIL | FAIL | pooled -0.091 (474 bets) | 0.345 |
| impl_draw_kmeans_match_lr | impl | draw | major, gn>8, dropna | KMeans k=5 (match, running, 3D) | train-profitable clusters | FAIL | FAIL | FAIL | pooled -0.150 (454 bets) | 0.402 |
| points_draw_kmeans_match_lr | points | draw | major, gn>8, dropna | KMeans k=5 (match, running, 2D) | train-profitable clusters | FAIL | FAIL | FAIL | pooled -0.024 (512 bets) | 0.516 |
| shots_on_target_foragst_draw_kmeans_match_lr | shots_on_target_foragst | draw | major, gn>8, dropna | KMeans k=5 (match, running, 4D) | train-profitable clusters | PASS | FAIL | FAIL | pooled +0.045 (486 bets) | 0.341 |
