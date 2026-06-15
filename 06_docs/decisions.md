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