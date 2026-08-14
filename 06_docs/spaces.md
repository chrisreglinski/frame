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
| `goals_foragst` | `hmt_season_goals_for_avg`, `hmt_season_goals_agst_avg` | `awt_season_goals_for_avg`, `awt_season_goals_agst_avg` | — |
| ↳ `goals_totaldiff` *(rotation of `goals_foragst`)* | `hmt_season_goals_total_avg`, `hmt_season_goals_diff_avg` | `awt_season_goals_total_avg`, `awt_season_goals_diff_avg` | — |
| `shots_on_target_foragst` | `hmt_season_shots_on_target_for_avg`, `hmt_season_shots_on_target_agst_avg` | `awt_season_shots_on_target_for_avg`, `awt_season_shots_on_target_agst_avg` | — |
| `points` | `hmt_season_points_avg` | `awt_season_points_avg` | — |
| `elo` | `hmt_elo` | `awt_elo` | — |
| `impl` | `mrkt_home_impl` | `mrkt_away_impl` | `mrkt_draw_impl` |
| `strength` *(planned)* | learned scalar per side (LR/PCA over a side's stats) | learned scalar per side | — |

## Exploration results registry

Model runs are logged in [`03_notebooks/model_checks/results.md`](../03_notebooks/model_checks/results.md),
which **replaces the old `results.txt`**. One row per run, keyed by `source` (the notebook), so re-runs update
in place. Notebooks write their row via `evaluation.results_registry.upsert(...)` (one market per notebook).

The file is split into one table per gate score — `## 3xPASS`, `## 2xPASS`, `## 1xPASS`, `## 0xPASS`, each
heading carrying its row count — so how many gates a run cleared is visible before reading any of its
numbers. The split is **derived from the G1/G2/G3 columns on every write, never stored**: a re-run whose
verdict changed moves to the right table by itself, and a row can only ever be in one of them. All four
headings are always written, an empty one as `_none_`. The preamble above the first heading is preserved;
everything from there down is regenerated. Inside a section rows run by pooled ROI descending (read out
of the `result` cell, notebook name as the tie-break) — the gate count is what the sections answer, so
the open question within one is which of the survivors earns most.

The validation protocol behind each run (folds, the three gates) lives in the model-check framework —
[`model_check.md`](model_check.md) / [`02_src/evaluation/model_check.py`](../02_src/evaluation/model_check.py)
— and is **space-independent**: a run may use a registered space or any mixed set of ABT input columns, in
which case the `space` column just names that feature set.

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
