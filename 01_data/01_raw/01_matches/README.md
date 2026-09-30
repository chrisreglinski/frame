# Matches

Results, match statistics and odds from [football-data.co.uk](https://www.football-data.co.uk), one
CSV per league-season: `<league>/<league>_<season>_matches.csv`. The files are stored as downloaded.
`football_data_notes.txt` is the source's own key to the columns.

`02_src/raw/match_raw_stats.py` reads a subset of them: date and kick-off time, teams, full-time goals
and result, shots, shots on target, corners, cards, and the Bet365 (`B365`) and market average (`Avg`)
odds. Odds with a `C` after the bookmaker (`B365CH`, `AvgCH`) are the closing line, the rest are
pre-closing. Our `b365` and `mrkt` columns come from these two.

## Known issues

Checked across all twelve leagues on 2026-09-30. Problems in the `minor` and `other` sets are only
recorded here and not handled in the builders, since those sets are not in use yet.

### major

- **germany_2425**: Union Berlin vs Bochum (14/12/2024) has no shot, corner or card stats. Goals and
  result are present. Rolling averages skip the missing value, so the only effect is a NaN
  `red_last_match` in the next match of both clubs. Left as-is.
- **france_2526**: 305 matches instead of 306. Nantes vs Toulouse (17/05/2026, final matchday) was
  abandoned in the 22nd minute at 0-0 after Nantes fans stormed the pitch in protest against
  relegation. Football-data has no rows for unresolved matches. If the LFP awards a forfeit, the
  source file may be updated to 306.

### minor

- **france2_2526**: Bastia vs Red Star (05/12/2025) is in the file with no result and no stats.
- **france2_2324**: 379 matches instead of 380. Troyes vs Valenciennes is missing.
- **france2_2223**: Bordeaux vs Rodez (02/06/2023) has no shot, corner or card stats.
- **italy2_2324**: Lecco vs Catanzaro (03/09/2023) has neither Bet365 nor market average odds.
- Bet365 odds missing, market average present: two matches in france2_2223, three in spain2_2223, one
  each in france2_2324, france2_2526, spain2_2324 and italy2_2324.

### other

- **portugal_2223**: one match without Bet365 odds (market average present).
