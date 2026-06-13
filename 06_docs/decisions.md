## 2026-06-13 — ABT parametrization
`build_abt()` always produces the full wide table (all columns, all rows). No parametrization.
Filtering by rows (league, season, game_number threshold) and columns (features vs targets) is done downstream in notebooks. The full ABT is small enough to hold in memory, and slicing pandas is trivial.

## 2026-06-12 — match_id
`match_id` to be built as md5(league|season|home_team|away_team).
the columns have to be in the input table and optionaly can be in (one) output table.