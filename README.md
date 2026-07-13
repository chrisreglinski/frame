# frame

Football analytics framework that:
- produces a clean, point-in-time-correct Analytical Base Table (ABT)
- enables easy EDA on created features
- enables predictive/betting model investigation and validation under a fixed validation protocol

---

## Leagues & seasons

| League | Seasons |
|--------|---------|
| England (Premier League) | 2223, 2324, 2425, 2526 |
| Spain (La Liga) | 2223, 2324, 2425, 2526 |
| Italy (Serie A) | 2223, 2324, 2425, 2526 |
| Germany (Bundesliga) | 2223, 2324, 2425, 2526 |
| France (Ligue 1) | 2223, 2324, 2425, 2526 |

Seasons before 2223 excluded — COVID-era matches (2020–2022) had no fans, reducing home advantage and skewing result distributions.

Source: [football-data.co.uk](https://www.football-data.co.uk)

---

## Project structure

```
01_data/
  01_raw/
    01_matches/        # raw CSVs from football-data.co.uk (tracked in git)
    02_stadiums/       # stadium coordinates + promoted team flags per league-season
    03_dates/          # season phase boundary dates
    04_elo/            # Club Elo history (clubelo.com) + per-group team-name maps
  02_features/         # generated feature tables (gitignored, rebuild locally)
  03_abt/              # final wide ABT (gitignored, rebuild locally)
02_src/
  01_raw/              # match_raw_stats builder (aggregates raw CSVs)
  02_features/         # feature builders (one file per table)
03_notebooks/          # exploratory notebooks (gitignored except template.ipynb)
04_models/             # trained models (gitignored)
05_reports/            # outputs (gitignored)
06_docs/               # contract, decisions, raw data notes
```

---

## Key design principles

### Single source of truth — `06_docs/data.yaml`

Every column in every feature table is defined in `data.yaml` with name, dtype, dims,
and description. Builders implement the contract; the contract is never changed to match
the code. Any schema change requires updating `data.yaml` first, then the code.

### match_id

All tables join on `match_id = md5(league|season|hmt_name|awt_name)`.
Natural key columns (`league`, `season`, `hmt_name`, `awt_name`) live only
in `match_info` and are not duplicated in other tables.

### Point-in-time correctness

All rolling/expanding statistics use `shift(1)` before any aggregation.
The current match is never included in its own features.

`hmt_elo` / `awt_elo` are joined point-in-time from Club Elo: the rating whose window
contains the match date (`From <= date <= To`) is the pre-match value — Club Elo dates each
post-match update to the following day — so no result leaks into the feature.

---

## Feature tables

### `match_info`

One row per match. Context and market features:

- `league`, `season`, `date`, `time`, `day_of_week`, `hmt_name`, `awt_name`
- `season_game_number` — sequential match number in the league-season
- `gameweek` — derived as `ceil(season_game_number / (n_teams / 2))`; `n_teams` computed dynamically per league-season (handles France's drop from 20 to 18 teams after 2223)
- `season_4phase` — `summer / fall / winter / spring` based on hand-coded boundary dates in `01_raw/03_dates/season_limit_dates.csv`
- `season_3phase` — `start / mid / end` (fall+winter merged into mid)
- `day_of_week_cat` — `weekend` (sat/sun) / `shoulder` (fri/mon) / `midweek` (tue/wed/thu)
- `time_uk_num` — kick-off time (UK) as a number (`hour + minute/60`); `time_uk_cat` — bucketed by floor(hour): `early` (11–13) / `early_afternoon` (14–15) / `late_afternoon` (16–17) / `evening` (18+)
- `hmt_is_promoted`, `awt_is_promoted`, `travel_distance_km`
- `hmt_elo`, `awt_elo` — Club Elo rating (clubelo.com) of each team as of the match date, joined point-in-time (pre-match; see below)
- `hmt_elo_cat2m` / `awt_elo_cat2m` (high/low vs the global elo mean) and `hmt_elo_cat3q` / `awt_elo_cat3q` (tertiles of the pooled per-match elo distribution; thresholds in `thresholds.json`)
- `b365_*` / `mrkt_*` — odds, implied probabilities (1/odds), bookmaker margin, Shannon entropy of normalized implied probs
- `mrkt_favourite`, `mrkt_impl_order`, `mrkt_favrt_impl`, `mrkt_undrd_impl`, `mrkt_home_away_impl_diff` — derived market signals: favoured side (home/away/balanced), H/D/A ordering by implied prob, stronger/weaker side implied prob, home − away implied gap

### `match_team_stats`

One row per match, wide format. Team statistics computed from all previous matches in
the season for both the home team (`hmt_` prefix) and away team (`awt_` prefix).

Three window variants for every statistic:

| Window | Logic |
|--------|-------|
| `season` | expanding from match 1 (`min_periods=1`) |
| `rolling6` | last 6 matches (`min_periods=6`, NaN if fewer) |
| `rolling8` | last 8 matches (`min_periods=8`, NaN if fewer) |

Statistics per team per window:
- Goals, shots, shots on target, corners, yellow cards — for & against averages
- Points — average
- goals_diff, goals_total, shots_on_target_diff, shots_on_target_total — average & std
- Win / draw / loss ratio
- Goals total / for / against threshold ratios (e.g. over 2.5, clean sheets)
- Shots on target conversion ratio
- Red cards average; red card in last match flag
- Home-only (`hmt_home_*`) and away-only (`awt_away_*`) season splits: goals / shots / shots-on-target for & against, plus points / goals_diff / goals_total, over that team's home (resp. away) matches only
- `implied_win/draw/loss_avg` — bookmaker's (mrkt) average implied probability for this team's outcome
- `impl_points_avg` — bookmaker expected points per match (`impl_win_avg * 3 + impl_draw_avg`)
- `win/draw/loss_profit` — edge: actual ratio minus implied avg (positive = team undervalued by market)
- Season-level categoricals (`season` window only): `*_cat2m` (binary — vs global mean, or sign) and `*_cat3q` (tertiles from `team_season_final`) for goals and shots-on-target total / diff / for / against

### `match_matchup_stats`

Derived from `match_team_stats`. Comparative features per window:
- `teams_{window}_goals_foragst_avg_max` — max of the four goals averages (home for, home agst, away for, away agst)
- `homet/awt_{window}_goals_foragst_avg_max` — per-team max(goals_for_avg, goals_agst_avg) for the home (`hmt_`) and away (`awt_`) side
- Matchup pairs, per axis (goals, shots_on_target, points, impl_points, elo):
  - **tilt** (home − away, positive = home stronger): `teams_{window}_goals_diff_avg_diff`,
    `teams_{window}_shots_on_target_diff_avg_diff`, `teams_{window}_points_avg_diff`,
    `teams_{window}_impl_points_avg_diff`, `teams_elo_diff`.
  - **total** (home + away, combined intensity / level): `teams_{window}_goals_total_avg_total`,
    `teams_{window}_shots_on_target_total_avg_total`, `teams_{window}_points_avg_total`,
    `teams_{window}_impl_points_avg_total`, `teams_elo_total`.
- Naming: a for/agst quantity carries an inner for−agst / for+agst term (`goals_diff`, `goals_total`),
  so its matchup feature is `_diff_avg_diff` / `_total_avg_total`; a per-side average (points, impl_points)
  gives `_avg_diff` / `_avg_total`. Elo is a raw rating, so `teams_elo_diff` / `teams_elo_total` (no window).

### `match_target`

Targets prefixed `t_`:
- `t_home_goals`, `t_away_goals`, `t_result`
- `t_home_flg`, `t_draw_flg`, `t_away_flg`
- `t_goals_diff`, `t_goals_total`
- `t_home_profit`, `t_draw_profit`, `t_away_profit` — profit from a unit stake on that outcome at mrkt odds (win: `1 - impl`; lose: `-impl`)

### ABT (`01_data/03_abt/abt.parquet`)

Wide join of all four tables on `match_id`. ~7 000 rows, ~380 columns.
Full table always generated; filter by league/season/phase/game_number downstream.

---

## Home/away feature spaces

A `home_away_feature_space` is a named bundle of symmetric ABT columns describing the
home and away team — a name plus a list of ABT columns (`home`, `away`, and optionally
derived columns like `draw_impl`). No transformation happens inside a space; input
columns are transformed beforehand. Two registries sit on top: a catalogue of existing
spaces (`goals_foragst`, `shots_on_target_foragst`, `points`, `elo`, `impl`, …) and a
log of models built on them. Full description:
[`06_docs/spaces.md`](06_docs/spaces.md).

---

## Model-check framework

Betting strategies are validated by the model-check framework in
[`02_src/evaluation/model_check.py`](02_src/evaluation/model_check.py) — the source of truth for the
protocol (no versioned spec). A check is: filter the ABT → segment the matches → fit a bet-signal
model (train only, per fold) → bet where the model beats the market → score, then pass three
robustness gates (LOSO time, drop-best-league, walk-forward floor). Each `model_checks/` notebook
supplies only the changing parts — the **space**, **target**, **segmenter** and **model** — in
dedicated cells; the folds, gates and registry write come from the framework. The protocol is
space-independent (works on any feature set). Full description:
[`06_docs/model_check.md`](06_docs/model_check.md).

---

## Setup

```bash
pip install -e .
```

Required once after cloning. Installs the `02_src/` package in editable mode so
notebooks can import directly: `from features.match_info import build_match_info`,
`from evaluation.portfolio import portfolio_roi`, etc.

---

## Rebuild pipeline

```bash
python build_abt.py               # rebuild the pipeline (default: the top-5 'major' group)
python build_abt.py --group NAME  # rebuild a different league group (a folder under 01_matches/)
python build_abt.py --skip-raw    # skip match_raw_stats (when raw CSVs are unchanged)
```

`build_abt.py` in the project root runs all builders in dependency order for one league
group and prints timing for each step. The pipeline is parameterized by group: a group reads
its match CSVs from `01_data/01_raw/01_matches/<group>/` and writes to
`01_data/02_features/<group>/` and `01_data/03_abt/<group>/`, so additional league groups can
be dropped in as new folders. Only the top-5 `major` group is active for now. Use `--skip-raw`
when the source CSVs have not changed.

Or use `03_notebooks/template.ipynb` to load all tables directly.

---

## Known data quality issues

See `06_docs/01_raw/comments.txt` for full notes. Key issues:

- **germany_2425**: one match (Union Berlin vs Bochum, 14/12/2024) has NaN for all
  shot/corner/card stats. Propagates as NaN in rolling windows for both clubs for
  several subsequent matches. Left as-is.
- **france_2526**: 305 matches instead of 306 — Nantes vs Toulouse abandoned mid-match.
