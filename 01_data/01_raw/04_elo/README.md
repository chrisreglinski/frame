# Club Elo

`england/`, `spain/`, `italy/`, `france/`, `germany/` hold rating histories of the top-flight
clubs of each league, pulled **2026-08-21** from clubelo.com by `02_src/raw/fetch_clubelo.py`.
This is the rating system the site switched to around 2026-08; `_archive/clubelo_history.csv`
is the old one and the two are not comparable.

One folder per league, matching the rest of `01_raw`. The seven leagues without a pull
(second tiers, netherlands, portugal) carry only a team map.

- `<slug>.json` — the club page's chart data verbatim, one point per match played:
  `{"Date": "2022-08-20T00:00:00", "Elo": 1852.0478683793012, "Golo": 1.3944618, "segment_id": 0}`
- `team_map.csv` — our team name, the old clubelo name, and the current slug. An empty
  slug means the club has no page on the site (see below); an empty clubelo name means it
  was not in the old maps either.
- `_manifest.csv` — slug, url, point count, first/last date. Only where a pull happened.

## What the data is like

- The value at date D is the **post-match** rating. The archived file was the opposite: the
  value valid on D was the pre-match rating. A point-in-time join has to take the last point
  strictly *before* the match date.
- `Golo` is a second metric the rebuilt site introduced. Pulled along, not yet understood.
- Every club's series starts 2022-08-20/21/22 regardless of how many matches it has played.
  That is a hard cutoff, roughly four years back, identical for every club — the site does
  not serve anything older to anonymous visitors. Whether it moves day by day is untested;
  re-pulling one club in a few days would settle it.

## Coverage against our matches

| league | clubs | points | matches | covered | gap |
|---|---:|---:|---:|---:|---:|
| england | 25 | 4825 | 1520 | 1490 | 30 |
| spain | 26 | 4657 | 1520 | 1496 | 24 |
| italy | 22 | 3926 | 1520 | 1217 | 303 |
| france | 22 | 3522 | 1297 | 1164 | 133 |
| germany | 23 | 3622 | 1224 | 1195 | 29 |

Two separate causes. The small gaps (24-30) are the four-year cutoff eating gameweeks 1-3
of 22/23. The large ones are **seven clubs with no page at all**: Empoli, Pisa, Salernitana,
Sampdoria, Spezia in Italy, Ajaccio and Clermont in France — every match they played is
uncovered.

The site only publishes a club page for roughly the top 600 clubs worldwide. Below that a
club appears in ranking tables as plain text with a current rating, but is never linked and
has no page; all slug guesses for these seven return a redirect to the front page. The same
wall blocks the second tiers entirely (78 of 174 clubs in `minor/`), so `england2` and
friends cannot be rebuilt from this source as things stand.
