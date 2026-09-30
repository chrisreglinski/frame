# frame

A football analytics framework for European league matches. It does three things:

- builds a clean, point-in-time-correct Analytical Base Table (ABT), one row per match,
- makes it easy to explore the features it contains,
- defines, validates and reports betting models under one fixed protocol.

---

## Quick start

```bash
pip install -e .          # once after cloning: installs 02_src/ as an editable package
python build_abt.py       # build the ABT for the default league set (major)
```

```python
import pandas as pd
abt = pd.read_parquet("01_data/03_abt/major/abt.parquet")
```

Notebooks import straight from the package, e.g. `from features.match_info import build_match_info`
or `from models import registry`. `03_notebooks/template.ipynb` loads every table.

Build options:

```bash
python build_abt.py --group minor     # another league set from league_sets.yaml
python build_abt.py --skip-raw        # raw CSVs unchanged, skip match_raw_stats
python build_abt.py --skip-thresholds # reuse the existing thresholds.json
```

Tests run on small hand-built data: `pytest`.

---

## Data

Twelve leagues, grouped into league sets in `01_data/league_sets.yaml`. A set is built as one
unit: its features, thresholds and ABT live under its own name.

| Set | Leagues |
|-----|---------|
| `major` | England, Spain, Italy, France, Germany (top flights) |
| `minor` | the second tiers of the same five countries |
| `other` | Netherlands, Portugal |

Only `major` is actively used for now. Every league covers seasons **2223 to 2526**. Earlier seasons
are left out on purpose: in the COVID years matches were played without fans, home advantage shrank,
and the results come from a different process.

| Source | What it provides | Reach |
|--------|------------------|-------|
| [football-data.co.uk](https://www.football-data.co.uk) | results, match stats, pre-closing and closing odds | all leagues |
| [clubelo.com](http://clubelo.com) | Elo and Golo ratings per club | top flights only |
| [Understat](https://understat.com) | match xG | top flights only |
| [FBref](https://fbref.com) | domestic cups, European and FIFA club competitions | all leagues |
| hand-built | stadium coordinates, promoted / relegated / top-3 / island flags, season phases, international breaks | varies |

Columns that depend on a source with limited reach (xG, Elo, travel distance) are scoped with
`leagues:` in the contract. A set built from other leagues does not get them at all, rather than
getting them empty. Source folders carry their own notes:
[`01_matches/README.md`](01_data/01_raw/01_matches/README.md) for the football-data files and their
known issues, [`04_elo/README.md`](01_data/01_raw/04_elo/README.md) for the Club Elo pull and its gaps.

---

## Project structure

```
01_data/
  league_sets.yaml     # which leagues make up each set
  01_raw/              # raw inputs, stored per league: <domain>/<league>/...
    01_matches/        #   football-data CSVs
    02_attributes/     #   per-team attributes (coordinates, flags)
    03_dates/          #   season phase boundaries and international breaks
    04_elo/            #   Club Elo series and team-name maps
    05_xg/             #   Understat xG and team-name maps
    06_europe/         #   UEFA competitions
    07_domestic/       #   domestic cups and super cups
    08_intl/           #   FIFA club competitions
  02_features/<set>/   # generated feature tables (gitignored)
  03_abt/<set>/        # generated ABT (gitignored)
02_src/
  raw/                 # match_raw_stats, Club Elo fetcher
  features/            # one builder per feature table, plus thresholds and the ABT join
  models/              # spaces, estimator factory, run, save layer, registry
  evaluation/          # folds, predictions, stats, staking, calibration
  reporting/           # tearsheet: panels, render primitives, two report variants
03_notebooks/          # exploration notebooks
04_models/             # saved models and runs (gitignored)
05_reports/            # published reports (gitignored)
06_docs/               # contract, specs, decisions
tests/                 # pytest suite
build_abt.py           # runs the whole pipeline for one league set
```

---

## Design principles

**The contract comes first.** `06_docs/data.yaml` defines every table and column: name, dtype,
dimensions, description, and which leagues it covers. Builders implement the contract, never the
other way round. A schema change starts in `data.yaml` and only then reaches the code.

**One key.** Every table joins on `match_id = md5(league|season|hmt_name|awt_name)`. The natural key
columns live only in `match_info`.

**Point-in-time correctness.** No feature may see its own match. Rolling and expanding statistics
shift by one match before aggregating. Club Elo publishes the post-match rating on the match date, so
the pre-match rating is the last point strictly before it. Calendar windows are half-open and end
just before kick-off.

**Pre-closing vs closing odds.** `mrkt` and `b365` are the pre-closing line, the price available when
a match is evaluated, and the only one a model may use. `mrktc` and `b365c` are the closing line near
kick-off. They serve validation only: settling a signal at the closing price checks whether the edge
survives a sharper market.

---

## ABT tables

The ABT (`01_data/03_abt/<set>/abt.parquet`) is a wide join of five tables on `match_id`. It is
always built in full, and filtering by league, season or phase happens downstream. Every column is
described in [`data.yaml`](06_docs/data.yaml). The summaries below only say what each table is for.

- **`match_info`**: context and market. Dates, gameweeks and season phases, counters relative to
  international breaks, kick-off time, promoted and top-3 flags, travel distance, Club Elo ratings,
  and odds, implied probabilities and margins for both the pre-closing and closing lines.
- **`match_team_stats`**: each team's form before the match, over three windows (`season`,
  `rolling6`, `rolling8`). Goals, shots, xG, points, result ratios and market-implied points, with
  home-only and away-only splits for the venue each team plays at.
- **`match_matchup_stats`**: the two teams compared. Home minus away (`_diff`) and home plus away
  (`_total`) for every form metric, Elo mapped to an expected result margin, and the fatigue
  columns combined per match.
- **`match_team_fatigue`**: schedule density on each team's full calendar, cups included. Hours since
  the last three fixtures, fixture counts over 8, 15 and 45 days, and smoothly decaying loads, each
  split by venue (home / domestic / abroad) and competition (league / europe / other).
- **`match_target`**: the outcomes, prefixed `t_`. Goals, result, outcome flags, goal difference and
  total, and the profit of a unit bet on each outcome.

---

## Models

Model exploration is built around **spaces**: groups of ABT columns that belong together by content
and describe both teams the same way, such as `goals_foragst`, `points`, `xg` or `elo`. Trying spaces
and their combinations first shows which families of features carry signal, before any finer
selection. Form spaces also come in rolling versions (`_r6`, `_r8`). A model can also take single
variables outside any space, listed as `extra`. The registry is `02_src/models/spaces.py`, described
in [`spaces_spec.md`](06_docs/spaces_spec.md).

- **model**: `04_models/<name>/params.yaml` with target, league set, spaces, extra, estimator and
  hyperparameters. Slug: `draw-major-goals_foragst+points-xgb-d5b3b3`.
- **run**: leave-one-season-out predictions, bets above a buffer, money metrics. Saved to
  `runs/<run-id>/`.
- **report**: a tearsheet over a saved run, `render_tearsheet` or `render_simple`.

```python
from models import registry
from models.run import run
from models.store import save_run
from reporting.tearsheet import render_tearsheet

r = run(registry.get("Sybilla"))        # train, predict, bet, score
r.metrics                               # yield, p-value, drawdown, ...
save_run(r, html=render_tearsheet(r))   # -> 04_models/sybilla/runs/<run-id>/
```

Details: [`models_spec.md`](06_docs/models_spec.md), [`tearsheet_spec.md`](06_docs/tearsheet_spec.md).

---

## Evaluation

Plain functions in `02_src/evaluation/`, usable with or without the model framework. Column names are
arguments, so they run on model predictions or straight on the ABT. The snippets below assume `preds`,
out-of-fold predictions with `model_p`, `implied`, `fair` (devigged) and `y`, e.g.
`run(model).predictions`.

**Forecast quality**

- `scoring`: skill scores against a constant baseline (R² for a margin, Brier skill for a
  probability, RPSS for H/D/A).
- `compare_*`: a model and the market scored on the same matches, so a model is measured directly
  against the odds.
- `calibration`: predicted probabilities against observed frequencies.

```python
from evaluation.scoring import compare_outcome

# does the model forecast the outcome better than the market?
compare_outcome(preds["model_p"], preds["fair"], preds["y"])
```

**Portfolio quality**

The odds already reflect most of what is known, so a model can forecast well and still lose money.
The final test is profit at the odds on offer. `portfolio_stats` stakes each bet proportionally to its
implied probability and ignores the bankroll, so the result does not depend on scale or bet order and
isolates the edge.

| Key | Meaning |
|-----|---------|
| `n_matches` | number of bets |
| `staked` / `wins` / `profit` | total stake, winning bets, `wins - staked` |
| `roi` | yield: profit per unit staked |
| `hit_rate` / `breakeven` | share of bets won, and the share needed to break even (mean implied) |
| `p_value` | how likely the result is with no edge: against `prob=` if given (devigged), else the raw implied |

```python
from evaluation.stats import portfolio_stats

# bet where the model beats the price by 2 points: is there an edge?
bets = preds[preds["model_p"] > preds["implied"] + 0.02]
stats = portfolio_stats(bets, prob="fair")
```

**Bankroll management**

`bankroll_stats` stakes a Kelly fraction of the current bankroll, sized from the implied price lifted
by the measured `roi`. Gains and losses compound, so it shows what the money would actually do. The
bet-by-bet path behind it, for charts, comes from `kelly_bankroll`.

| Key | Meaning |
|-----|---------|
| `start` / `final` / `profit` | starting bankroll, bankroll after the last bet, `final - start` |
| `max_drawdown` | deepest fall from the running peak |
| `cagr` | annual growth of the bankroll over the betting period |

```python
from evaluation.staking import bankroll_stats

# the same bets at half-Kelly, staked in date order
bankroll_stats(bets.sort_values("date"), roi=stats["roi"], kelly_fraction=0.5)
```

---

## Documentation

| File | Contents |
|------|----------|
| [`data.yaml`](06_docs/data.yaml) | the data contract: every table and column |
| [`models_spec.md`](06_docs/models_spec.md) | models, runs and how they are stored |
| [`tearsheet_spec.md`](06_docs/tearsheet_spec.md) | the model report |
| [`spaces_spec.md`](06_docs/spaces_spec.md) | feature spaces: what they are and which exist |
| [`decisions.md`](06_docs/decisions.md) | the few design decisions worth their reasons |
| [`journal.md`](06_docs/journal.md) | working journal |

---

## Known data issues

Listed per league in [`01_matches/README.md`](01_data/01_raw/01_matches/README.md). The `major` set
has two: one Bundesliga match without stats and one abandoned Ligue 1 match.
