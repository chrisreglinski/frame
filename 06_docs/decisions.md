# Decisions

## 2026-07-01 - Category thresholds stay global
The bucket edges of the categorical columns (`cat2m`, `cat3q`) come from `thresholds.json`, computed
over all seasons. This is the only cross-fold leak, and it is accepted. It touches only the boundary:
the bucketed value itself is point-in-time correct. The cut is arbitrary anyway, so recomputing it per
training fold would only move it, not remove it, and the means and quantiles barely shift across
seasons. If a strategy that leans directly on categoricals ever passes the gates, recompute
`build_thresholds(train_mask)` before trusting its number.

## 2026-06-14 - Seasons scope: 2223 to 2526 only
Seasons before 2223 are excluded. In the COVID era (2020 to 2022) matches were played without fans,
which measurably reduced home advantage and skewed result distributions. Those seasons come from a
different data-generating process.

## 2026-06-13 - Rolling windows: min_periods=1 plus an explicit early-season mask
Rolling aggregations use `min_periods=1`, so a single missing raw value mid-season does not turn the
whole window into NaN. Early-season NaN is enforced separately: after all computations, every rolling
column is set to NaN where `game_number <= window_size`. The two concerns stay independent and
explicit.

## 2026-06-12 - match_id
`match_id` is built as `md5(league|season|hmt_name|awt_name)`. The key columns have to be in the input
table, and may appear in at most one output table (`match_info`).
