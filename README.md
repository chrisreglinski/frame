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
  01_raw/              # per-league-season payload: {group}/{league}_{season}_{domain}.csv
    01_matches/        # raw CSVs from football-data.co.uk (tracked in git)
    02_attributes/     # per-team attributes: venue coordinates + promoted / reigning-top3 flags
                       # (minor/other groups: promotee + island flags only, coordinates
                       #  empty -> travel_distance stays NaN there)
    03_dates/          # season phase boundary dates (shared, global)
    04_elo/            # Club Elo history (clubelo.com, shared) + per-group team-name maps
    05_xg/             # Understat match xG per league-season + per-group team-name map
    06_europe/         # UEFA club competitions per season (FBref): CL / EL / Conference
                       # + qualifying, UEFA Super Cup, and a team-name map
    07_domestic/       # domestic cups and super cups per league-season (FBref)
                       # + a team-name map (only the names that differ from football-data)
    08_intl/           # FIFA club competitions (Club World Cup, Intercontinental Cup)
  02_features/         # generated feature tables (gitignored, rebuild locally)
  03_abt/              # final wide ABT (gitignored, rebuild locally)
02_src/
  01_raw/              # match_raw_stats builder (aggregates raw CSVs)
  02_features/         # feature builders (one file per table)
03_notebooks/          # exploratory notebooks (gitignored except template.ipynb)
04_models/             # trained models (gitignored)
05_reports/            # outputs (gitignored)
06_docs/               # contract, decisions, raw data notes
tests/                 # pytest suite on small hand-built data (run: `pytest`)
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

### Pre-closing vs closing odds

`mrkt`/`b365` are the **pre-closing** line; `mrktc`/`b365c` are the **closing** line (near
kickoff). Only the pre-closing line may enter a model — it is the only price available when the match
is evaluated; the closing line does not exist yet. The closing line is for **validation only**:
settling a pre-closing signal at closing odds checks whether the edge still earns at that sharper
price. It is never a model input.

---

## Feature tables

### `match_info`

One row per match. Context and market features:

- `league`, `season`, `date`, `time`, `day_of_week`, `hmt_name`, `awt_name`
- `season_game_number` — sequential match number in the league-season; `season_game_number_inv` — the same counted from the end (`-1` = last match, `-N` = first, `N` = full-season match count from team count; a skipped match leaves the last at `-2`)
- `gameweek` — derived as `ceil(season_game_number / (n_teams / 2))`; `n_teams` computed dynamically per league-season (handles France's drop from 20 to 18 teams after 2223); `gameweek_inv` — same counted from the end (`-1` = last gameweek)
- `season_4phase` — `summer / fall / winter / spring` based on hand-coded boundary dates in `01_raw/03_dates/season_limit_dates.csv`
- `season_3phase` — `start / mid / end` (fall+winter merged into mid)
- `day_of_week_cat` — `weekend` (sat/sun) / `shoulder` (fri/mon) / `midweek` (tue/wed/thu)
- `time_uk_num` — kick-off time (UK) as a number (`hour + minute/60`); `time_uk_cat` — bucketed by floor(hour): `early` (11–13) / `early_afternoon` (14–15) / `late_afternoon` (16–17) / `evening` (18+)
- `hmt_is_promoted`, `awt_is_promoted`, `travel_distance_km`
- `hmt_is_top3_last`, `awt_is_top3_last` — team finished top 3 in this league last season (reigning top-3; top-tier groups `major`/`other` only). Earliest season seeded from external final tables; later seasons match `team_season_final` rankings. The `minor` group carries the second-tier analogue `hmt_is_relegated` / `awt_is_relegated` (relegated from the tier above) instead — both are group-scoped in `data.yaml` via `groups:`
- `hmt_is_island`, `awt_is_island` — team is on a geographically isolated island (Las Palmas, Mallorca, Cagliari, Ajaccio)
- `travel_distance_cat` — `travel_distance_km` bucketed: `derby` (<30 km) / `regional` (30–100 km) / `domestic` (100–500 km, coach / high-speed rail) / `long_haul` (500+ km, flights)
- `hmt_elo`, `awt_elo` — Club Elo rating (clubelo.com) of each team as of the match date, joined point-in-time (pre-match; see below)
- `hmt_elo_cat2m` / `awt_elo_cat2m` (high/low vs the global elo mean) and `hmt_elo_cat3q` / `awt_elo_cat3q` (tertiles of the pooled per-match elo distribution; thresholds in `thresholds.json`)
- `b365_*` / `mrkt_*` — odds, implied probabilities (1/odds), bookmaker margin, Shannon entropy of normalized implied probs (pre-closing line). `b365c_*` / `mrktc_*` — the same for the **closing** (kickoff) line (football-data C columns); empty where no closing price
- `mrkt_favourite`, `mrkt_impl_order`, `mrkt_favrt_impl`, `mrkt_undrd_impl` — derived market signals (mrkt only): favoured side (home/away/balanced), H/D/A ordering by implied prob, stronger/weaker side implied prob
- `{bookmaker}_home_away_impl_diff` — home − away implied gap, for all 4 books (`b365`/`mrkt` pre-closing, `b365c`/`mrktc` closing)

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
- Goals, shots, shots on target, corners, yellow cards, xG — for & against averages
- Points — average
- goals_diff, goals_total, shots_on_target_diff, shots_on_target_total, xg_diff, xg_total — average & std
- xG is sourced from Understat (`05_xg/`, mapped via `team_map`); available only where Understat covers the league (major group) — NaN elsewhere
- Win / draw / loss ratio
- Goals total / for / against threshold ratios (e.g. over 2.5, clean sheets)
- Shots on target conversion ratio
- Red cards average; red card in last match flag
- Home-only (`hmt_home_*`) and away-only (`awt_away_*`) season splits over that team's home (resp. away) matches only: goals / shots / shots-on-target for & against, points / `flg_diff` / goals_diff / goals_total, win/draw/loss ratio, and `{mp,mc}_impl_points_avg` (expected points) / `{mp,mc}_impl_diff_avg` (implied margin). Plus `{points,flg_diff}_avg_adv` / `{mp,mc}_impl_{points,diff}_avg_adv` — venue advantage: the team's form where it plays this match minus its form at the other venue (home team: home − away, positive = better at home; away team: away − home, usually negative = away disadvantage)
- `{mp,mc}_impl_{win,draw,loss}_avg` — market average implied probability for this team's outcome, on the **mp** (market pre-closing) / **mc** (market closing) line
- `{mp,mc}_impl_points_avg` — market-expected points per match (`impl_win_avg * 3 + impl_draw_avg`), mp / mc lines
- `{mp,mc}_impl_diff_avg` — market-implied margin per match (`impl_win_avg − impl_loss_avg`), the team-oriented `mrkt_home_away_impl_diff`, mp / mc lines
- `flg_diff_avg` — realized signed-result margin per match (`wins_ratio − losses_ratio`, i.e. mean of `win_flg − loss_flg` ∈ {+1,0,−1}, same axis as `t_flg_diff`); realized counterpart of `impl_diff_avg`
- Season-level categoricals (`season` window only): `*_cat2m` (binary — vs global mean, or sign) and `*_cat3q` (tertiles from `team_season_final`) for goals and shots-on-target total / diff / for / against

### `match_matchup_stats`

Derived from `match_team_stats`. Comparative features per window:
- `teams_{window}_goals_foragst_avg_max` — max of the four goals averages (home for, home agst, away for, away agst)
- `homet/awt_{window}_goals_foragst_avg_max` — per-team max(goals_for_avg, goals_agst_avg) for the home (`hmt_`) and away (`awt_`) side
- Matchup pairs, per axis (goals, shots_on_target, xg, points, impl_points, elo):
  - **tilt** (home − away, positive = home stronger): `teams_{window}_goals_diff_avg_diff`,
    `teams_{window}_shots_on_target_diff_avg_diff`, `teams_{window}_xg_diff_avg_diff`,
    `teams_{window}_points_avg_diff`, `teams_{window}_{mp,mc}_impl_points_avg_diff`, `teams_elo_diff`.
  - **total** (home + away, combined intensity / level): `teams_{window}_goals_total_avg_total`,
    `teams_{window}_shots_on_target_total_avg_total`, `teams_{window}_xg_total_avg_total`,
    `teams_{window}_points_avg_total`, `teams_{window}_{mp,mc}_impl_points_avg_total`, `teams_elo_total`.
- Naming: a for/agst quantity carries an inner for−agst / for+agst term (`goals_diff`, `goals_total`),
  so its matchup feature is `_diff_avg_diff` / `_total_avg_total`; a per-side average (points, impl_points)
  gives `_avg_diff` / `_avg_total`. Elo is a raw rating, so `teams_elo_diff` / `teams_elo_total` (no window).
- **Elo → probability margin**: the expected `p_home − p_away` implied by `teams_elo_diff`, mapped four
  ways on a 2×2 grid — **form** (cubic / logistic) × **fit** (to results / to market) — each fit once
  (offline, per group) and stored in `thresholds.json`. The tag is `{c,l}{r,m}r`:
  `crr`/`cmr` cubic, `lrr`/`lmr` logistic; `*rr` fit to realized outcomes (the true margin), `*mr` fit to
  `mrkt_home_away_impl_diff` (the market's pricing skeleton). Each splits additively into `..._stgh`
  (strength, odd in `d`) and `..._lhfa` (league home-field advantage) = `..._diff`, giving
  `teams_elo_diff_{crr,cmr,lrr,lmr}_impl_{diff,stgh,lhfa}` — all in the same units as
  `mrkt_home_away_impl_diff`.
  - **cubic** `g(d) = a0[league] + a1·d + a2·d² + a3·d³`; strength `a1·d + a3·d³`, home field
    `a0[league] + a2·d²`. `d` is clamped to the cubic's monotone range so extreme mismatches plateau.
  - **logistic** `margin = 2/(1 + 10^(−(d + hfa[league])/scale)) − 1` (the ClubElo Elo equation): a fixed
    shape with just a global `scale` and a per-league home shift `hfa` (in elo points) fitted; strength is
    the same formula at `hfa=0`, home field is the remainder. It saturates toward ±1 (no clamping needed),
    matches the cubic's accuracy with fewer parameters, and its home-field term shrinks for lopsided
    matches. The `*rr` vs `*mr` `hfa` gap is the market's per-league home-advantage mispricing (it
    underprices spain's fortress, overprices italy's/france's weak home edge); the logistic `scale` is
    near-identical for `rr` and `mr`, so the market reads elo strength at the right steepness and errs
    only on the home shift.

### `match_team_fatigue`

One row per match, wide format. Schedule density for both teams, computed on each team's
**full fixture calendar** — the domestic league plus every cup it played: domestic cups
(`07_domestic`), European cups incl. qualifying (`06_europe`), domestic and UEFA super cups,
and the FIFA club competitions (`08_intl`). Non-league fixtures enter the calendar as history
only; they are never rows of the table.

Fixtures are described on two orthogonal axes, each of which partitions the calendar, so
either family sums to the total:

- **venue** — `home` / `domestic` (away in the team's own country, or a neutral venue inside
  it: Wembley, La Cartuja, the Olimpico, Berlin, the Stade de France) / `abroad` (a trip out
  of the country). Roughly nine in ten `abroad` fixtures are European away legs; the rest are
  super cups and FIFA competitions, so the column reads as "played a serious match out of the
  country".
- **stream** — `league` / `europe` (CL / EL / Conference incl. qualifying) / `other`
  (domestic cups, domestic and UEFA super cups, FIFA club competitions).

The full venue × stream grid is *not* carried as features — several of its cells hold a few
dozen rows across four seasons — but `team_calendar.parquet`, written alongside the table,
keeps it in a `cell` column. No competition carries a weight: the parts are counted separately
so their relative cost is estimated downstream instead of asserted here.

- `{side}_hours_since_last_match` (capped at 200) and `{side}_hours_since_2nd_last_match`
  (capped at 500) — hours since that team's previous fixture in any competition, and since the
  one before it. Kick-offs are normalised to CET first (football-data prints UK times, FBref
  prints venue-local with CET in brackets)
- `{side}_games_in_{x}d`, `{side}_{venue}_games_in_{x}d`, `{side}_{stream}_games_in_{x}d` —
  fixtures in the `x` days before kick-off, in total and split along each axis; windows
  `8, 15, 45` days
- `{side}_games_load_{tau}d` / `{side}_games_load_gauss_{tau}d` (and their per-axis versions) —
  the same fixtures under a smooth kernel instead of a hard window. Each earlier fixture of the
  season contributes `exp(−(age_in_days / tau)^k)`, so nothing is dropped and nothing counts in
  full. `k = 1` (exponential) at `tau = 7, 14` — recent congestion and a chronic season load;
  `k = 2` (gaussian) at `tau = 4, 7` — flat for the first days then falling away sharply, which
  separates 3 / 4 / 5 days of rest where the exponential barely does
- `{side}_last_match_is_away`, `{side}_last_match_is_europe`, `{side}_last_match_is_abroad` —
  attributes of the previous fixture, whatever competition it was

Every backward window is half-open (`[kick-off − x days, kick-off)`), so the current match is
never counted, and is truncated at the season start.

### `match_target`

Targets prefixed `t_`:
- `t_home_goals`, `t_away_goals`, `t_result`
- `t_home_flg`, `t_draw_flg`, `t_away_flg`
- `t_flg_diff` — signed result: 1 home / 0 draw / −1 away (`t_home_flg − t_away_flg`)
- `t_goals_diff`, `t_goals_total`
- `t_home_profit`, `t_draw_profit`, `t_away_profit` — profit from a unit stake on that outcome at mrkt odds (win: `1 - impl`; lose: `-impl`)

### ABT (`01_data/03_abt/abt.parquet`)

Wide join of all five tables on `match_id`. ~7 000 rows, ~700 columns.
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

## Two ways to stake a portfolio

Once a model selects which matches to bet, there are two distinct ways to size those
bets — and they answer two different questions. The project uses both, deliberately, and
they should never be conflated.

**Proportional (implied) staking — "is there an edge, and where?"**
Every bet stakes an amount proportional to the market's implied probability. Exposure per bet is
fixed and the current bankroll is ignored, so the result is **scale-independent**: the headline
number is ROI / yield — profit per unit staked. This is the right lens for *measuring the edge
itself* and for comparing decisions like the bet threshold (buffer), because it isolates selection
quality from the path-dependent luck of when wins and losses arrive.

**Kelly on a bankroll — "what would the money actually do?"**
Each bet stakes a fraction of the **current** bankroll, where the fraction comes from the Kelly
formula and grows or shrinks with both wealth and the probability edge (usually run at a
conservative fraction — half- or quarter-Kelly). The bankroll compounds, so this is
**path-dependent**: order matters, drawdowns are real, and the output is a capital trajectory, not
a single yield. This is the right lens for *the realistic money story* and for judging survival
(drawdown, risk of ruin).

A subtlety that ties the two together: the probability fed into Kelly need not be the model's raw
`p`. Because the model is a good *selector* but not necessarily well *calibrated*, staking can use
a recalibrated estimate — e.g. the market price scaled by the measured edge, `implied × (1 + ROI)` —
so selection stays with the model while sizing rests on the more trustworthy, aggregate edge.

Rule of thumb: use **proportional staking to prove and tune the edge**, and **Kelly-on-bankroll to
show and stress-test the capital**.

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
