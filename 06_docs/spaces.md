# Home/away feature spaces

A `home_away_feature_space` is a data type: a named bundle of symmetric ABT columns
that describe the home and away team. It is just a **name + a list of ABT columns** —
`home`, `away`, and optionally derived columns (e.g. `draw_impl`).

No transformation happens inside a space. Any transformation of the input columns is
done beforehand (in the ABT / upstream); a space only names and groups existing
columns. There is no dedicated code module for spaces.

## Space registry

The catalogue of identified spaces.

| space | home | away | derived |
|---|---|---|---|
| `goals_foragst` | `homet_season_goals_for_avg`, `homet_season_goals_agst_avg` | `awayt_season_goals_for_avg`, `awayt_season_goals_agst_avg` | — |
| `shots_on_target_foragst` | `homet_season_shots_on_target_for_avg`, `homet_season_shots_on_target_agst_avg` | `awayt_season_shots_on_target_for_avg`, `awayt_season_shots_on_target_agst_avg` | — |
| `points` | `homet_season_points_avg` | `awayt_season_points_avg` | — |
| `elo` | `home_elo` | `away_elo` | — |
| `impl` | `mrkt_home_impl` | `mrkt_away_impl` | `mrkt_draw_impl` |
| `strength` *(planned)* | learned scalar per side (LR/PCA over a side's stats) | learned scalar per side | — |

## Exploration results registry

Model runs are logged in [`03_notebooks/model_checks/results.md`](../03_notebooks/model_checks/results.md),
which **replaces the old `results.txt`**. One row per run, keyed by `source` (the notebook), so re-runs update
in place. Notebooks write their row via `evaluation.results_registry.upsert(...)` (one market per notebook).

Columns:

- **source** — the `model_checks/` notebook (the row key).
- **space** — which space (from the space registry).
- **target** — market bet (draw / home / away).
- **abt filter** — row filter on the ABT (`gn` = both teams' `game_number`).
- **segmentation** — how segments are formed (fit on the training fold only).
- **segment filter** — which segments are kept (fit on the training fold only).
- **G1 / G2 / G3** — GATE 1 LOSO / GATE 2 drop-best-league / GATE 3 walk-forward (PASS/FAIL).
- **result** — pooled ROI and number of bets.
- **roi_std** — std of ROI across league×season cells (concentration diagnostic).
