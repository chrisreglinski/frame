# Exploration results registry

Replaces the old `results.txt`. One row per model run, keyed by `source` (a `model_checks/` notebook).
Written by `evaluation.results_registry.upsert(...)`. Convention: [`06_docs/spaces.md`](../../06_docs/spaces.md).
All runs use b365 odds.

| source | space | target | abt filter | segmentation | segment filter | G1 | G2 | G3 | result | roi_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| goals_foragst_favourite_draw | goals_foragst | draw | major, gn>8, dropna | mrkt_favourite | most-profitable seg (raw) = home | FAIL | FAIL | PASS | pooled +0.010 (563 bets) | 0.420 |
| goals_foragst_match_kmeans_lr_draw | goals_foragst | draw | major, gn>8, dropna | KMeans k=5 (match, 4D) | train-profitable clusters | PASS | PASS | PASS | pooled +0.136 (997 bets) | 0.336 |
| goals_foragst_team_kmeans_lr_draw | goals_foragst | draw | major, gn>8, dropna | KMeans k=4 team archetypes → 4x4 grid | train-profitable cells | PASS | PASS | PASS | pooled +0.088 (906 bets) | 0.288 |
| goals_foragst_team_kmeans_lr_home | goals_foragst | home | major, gn>8, dropna | KMeans k=4 team archetypes → 4x4 grid | train-profitable cells | PASS | PASS | FAIL | pooled +0.031 (551 bets) | 0.201 |
