# Archived Club Elo input

`clubelo_history.csv` — snapshot of clubelo.com ratings downloaded **2026-07-02**
(added in commit 55d6190, never refreshed). Covers 2021-05-11 .. 2026-05-30, with the
last value carried forward to 2026-12-31. Format: `Club,Country,Level,Elo,From,To,clubelo`,
one row per rating window; the value valid on match day D is the PRE-match rating,
because the old clubelo dated each post-match update to D+1.

Archived on 2026-08-21. Around 2026-08 clubelo.com was rebuilt and its rating algorithm
changed, restating the entire history: same-date values move by -59 to +52 points, all-time
peaks changed (this file has Arsenal at 2069.87 on 2026-03-07, the site now calls that day
Arsenal's best ever at 2024), and the club ordering changed. The old `api.clubelo.com`
endpoint is dead. So this file is not a stale copy of the current rating — it is a different
rating system, and mixing the two would be meaningless.

Kept because every elo-derived feature built so far, and every model check recorded against
them, rests on these numbers. Do not feed it to the builders again; the columns it fed are
marked `active: false` in `06_docs/data.yaml`.
