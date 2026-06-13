# frame

Football analytics pipeline for building predictive models on top-5 European leagues.
The goal is to produce a clean, point-in-time-correct Analytical Base Table (ABT)
that can be sliced and fed into betting/prediction models downstream.

---

## Leagues & seasons

| League | Seasons |
|--------|---------|
| England (Premier League) | 2223, 2324, 2425, 2526 |
| Spain (La Liga) | 2223, 2324, 2425, 2526 |
| Italy (Serie A) | 2223, 2324, 2425, 2526 |
| Germany (Bundesliga) | 2223, 2324, 2425, 2526 |
| France (Ligue 1) | 2223, 2324, 2425, 2526 |

Source: [football-data.co.uk](https://www.football-data.co.uk)

---

## Project structure

```
01_data/
  01_raw/
    01_matches/        # raw CSVs from football-data.co.uk (tracked in git)
    02_stadiums/       # stadium coordinates + promoted team flags per league-season
    03_dates/          # season phase boundary dates
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

All tables join on `match_id = md5(league|season|home_team|away_team)`.
Natural key columns (`league`, `season`, `home_team`, `away_team`) live only
in `match_info` and are not duplicated in other tables.

### Point-in-time correctness

All rolling/expanding statistics use `shift(1)` before any aggregation.
The current match is never included in its own features.

---

## Feature tables

### `match_info`

One row per match. Context and market features:

- `league`, `season`, `date`, `home_team`, `away_team`
- `season_game_number` — sequential match number in the league-season
- `gameweek` — derived as `ceil(season_game_number / (n_teams / 2))`; `n_teams` computed dynamically per league-season (handles France's drop from 20 to 18 teams after 2223)
- `season_4phase` — `summer / fall / winter / spring` based on hand-coded boundary dates in `01_raw/03_dates/season_limit_dates.csv`
- `season_3phase` — `start / mid / end` (fall+winter merged into mid)
- `home_is_promoted`, `away_is_promoted`, `travel_distance_km`
- `b365_*` / `mrkt_*` — odds, implied probabilities (1/odds), bookmaker margin, Shannon entropy of normalized implied probs

### `match_team_stats`

One row per match, wide format. Team statistics computed from all previous matches in
the season for both the home team (`homet_` prefix) and away team (`awayt_` prefix).

Three window variants for every statistic:

| Window | Logic |
|--------|-------|
| `season` | expanding from match 1 (`min_periods=1`) |
| `rolling6` | last 6 matches (`min_periods=6`, NaN if fewer) |
| `rolling8` | last 8 matches (`min_periods=8`, NaN if fewer) |

Statistics per team per window:
- Goals, shots, shots on target, corners, yellow cards — for & against averages
- Points, goals_diff, goals_total — average & std
- Win / draw / loss ratio
- Goals total / for / against threshold ratios (e.g. over 2.5, clean sheets)
- Shots on target conversion ratio
- Red cards average; red card in last match flag
- `implied_win/draw/loss_avg` — bookmaker's (mrkt) average implied probability for this team's outcome
- `profit_win/draw/loss` — edge: actual ratio minus implied avg (positive = team undervalued by market)

### `match_matchup_stats`

Derived from `match_team_stats`. Comparative features per window:
- `teams_{window}_goals_foragst_avg_max` — max of the four goals averages (home for, home agst, away for, away agst)
- `teams_{window}_goals_diff_diff` — home goals_diff_avg minus away goals_diff_avg

### `match_target`

Targets prefixed `t_`:
- `t_home_goals`, `t_away_goals`, `t_result`
- `t_home_flg`, `t_draw_flg`, `t_away_flg`
- `t_goals_diff`, `t_goals_total`

### ABT (`01_data/03_abt/abt.parquet`)

Wide join of all four tables on `match_id`. ~7 000 rows, ~307 columns.
Full table always generated; filter by league/season/phase/game_number downstream.

---

## Rebuild pipeline

```bash
python build_abt.py             # rebuild everything
python build_abt.py --skip-raw  # skip match_raw_stats (when raw CSVs are unchanged)
```

`build_abt.py` in the project root runs all builders in dependency order and prints
timing for each step. Use `--skip-raw` for the common case where the source CSVs
in `01_data/01_raw/01_matches/` have not changed.

Or use `03_notebooks/template.ipynb` to load all tables directly.

---

## Known data quality issues

See `06_docs/01_raw/comments.txt` for full notes. Key issues:

- **germany_2425**: one match (Union Berlin vs Bochum, 14/12/2024) has NaN for all
  shot/corner/card stats. Propagates as NaN in rolling windows for both clubs for
  several subsequent matches. Left as-is.
- **france_2526**: 305 matches instead of 306 — Nantes vs Toulouse abandoned mid-match.
