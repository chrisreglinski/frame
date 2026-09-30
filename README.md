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

- A **model** is a data record, not code: `04_models/<name>/params.yaml` holds the target, the league
  set, the spaces, the extra variables, the estimator name and its hyperparameters. Its identity
  renders as a slug like `draw-major-goals_foragst+points-xgb-d5b3b3`, where the final hash stands for
  the extra variables. A model that looks promising also gets a human name (e.g. Sybilla).
- A **run** trains and evaluates a model on specific seasons. It saves to `runs/<run-id>/` a frozen
  snapshot of the definition, the out-of-fold predictions and the metrics, so it can be reloaded
  without retraining.
- A **report** is a view over a saved run. The tearsheet comes in a technical variant
  (`render_tearsheet`) and a plain-language one (`render_simple`).

What a run does:

1. **Leave-one-season-out folds**: each season is predicted by a model trained on the other seasons,
   so every prediction is out of sample.
2. **Bets** are placed where the model's probability beats the market price by more than a buffer.
   Unless given, the buffer is picked from the profit curve on every season except the newest, which
   stays out as a holdout.
3. **Metrics** are the money metrics from Evaluation: proportional staking for the edge, a half-Kelly
   bankroll for the money path.

```python
from models import registry
from models.run import run
from models.store import save_run
from reporting.tearsheet import render_tearsheet

model = registry.get("Sybilla")         # definition from 04_models/sybilla/params.yaml
r = run(model)                          # season-out CV: train, predict, bet, score
r.metrics                               # yield, p-value, drawdown, ...
r.predictions                           # one row per match: model_p, implied, y, ...
save_run(r, html=render_tearsheet(r))   # -> 04_models/sybilla/runs/<run-id>/
```

A saved run is reloaded with `models.store.load_run(run_dir)`. Details:
[`models_spec.md`](06_docs/models_spec.md) and [`tearsheet_spec.md`](06_docs/tearsheet_spec.md).

---

## Evaluation

Plain functions in `02_src/evaluation/`, usable on any predictions or bets, with or without the model
framework. Two layers:

**Forecast quality.** `scoring` gives skill scores against a constant baseline: R² for a predicted
margin, Brier skill score for a probability, and optionally RPSS for the H/D/A triple. Every
`compare_*` scores a model and the market on the same matches and returns the gap, so a model can be
measured directly against the odds. `calibration` checks whether predicted probabilities match
observed frequencies.

**Money.** Beating the market's forecast is not enough: the odds already reflect most of what is
known about a match, so a model can forecast well and still lose money. The final test is profit at
the odds on offer:

| Metric | Key | Meaning |
|--------|-----|---------|
| yield | `roi` | profit per unit staked |
| profit | `profit` | total profit in stake units |
| p-value | `p_value` | how likely the result is if the model has no edge over the fair market price |
| max drawdown | `bank_maxdd` | largest fall of the bankroll from its peak |
| CAGR | `bank_cagr` | annual growth of the bankroll |

- **Proportional staking** (`portfolio_stats`) sizes each bet by the implied probability and ignores
  the bankroll. The result does not depend on scale or bet order, so it isolates the edge (`roi`,
  `profit`, `p_value`).
- **Kelly on a bankroll** (`kelly_bankroll`) stakes a fraction of the current bankroll, so gains and
  losses compound. It shows what the money would actually do (`bank_maxdd`, `bank_cagr`).

```python
from evaluation.stats import portfolio_stats

# back every away favourite: is there an edge?
bets = abt[abt["mrkt_favourite"] == "away"]
portfolio_stats(bets, implied="mrkt_away_impl", outcome="t_away_flg")
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
