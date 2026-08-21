"""Download Club Elo rating histories from clubelo.com club pages.

The old api.clubelo.com endpoint is dead (TCP 80 accepts but never answers, 443
refused) and the rebuilt site keeps its date-addressed pages behind a login. The
public club page, however, embeds the club's full rating series as the data of its
Vega-Lite chart:

    {"Date": "2026-03-07T00:00:00", "Elo": 2023.5542793195677, "Golo": 1.0908995,
     "segment_id": 0}

That is what this module pulls out. One point per match played, at full precision,
truncated by the site to roughly the last 220-240 matches. The series is stored raw,
one JSON file per club plus a manifest; reshaping it into the ABT contract is a
separate step.

Usage:
    python -m raw.fetch_clubelo england Arsenal ManCity Liverpool

The first argument is the destination folder under 01_raw/04_elo, the rest are
clubelo.com slugs, which are neither our team names nor the old `Club` values
(`Atletico` is now `atletico`, `Ath Bilbao` is `athletic-club`).
"""

import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


_ROOT = Path(__file__).parents[2]
_ELO_DIR = _ROOT / "01_data" / "01_raw" / "04_elo"

_BASE = "https://clubelo.com"
_UA = "frame-research/0.1 (personal football modelling project)"
_DELAY_S = 1.0
_TIMEOUT_S = 60


def fetch_page(slug: str) -> str:
    """GET the club page. Raises on any non-200; an unknown slug 302s to '/'."""
    req = urllib.request.Request(f"{_BASE}/{slug}", headers={"User-Agent": _UA})
    with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
        if resp.geturl().rstrip("/") == _BASE:
            raise ValueError(f"unknown slug '{slug}' (redirected to the front page)")
        return resp.read().decode("utf-8", errors="replace")


def extract_series(html: str) -> list[dict]:
    """Pull the rating series out of the embedded chart spec, values verbatim.

    The spec carries its data under 'datasets' keyed by a content hash, and the
    page holds more than one array of records; the rating one is identified by its
    records carrying an 'Elo' key."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r":\s*\[\s*\{", html):
        start = html.index("[", match.start())
        if '"Elo"' not in html[start:start + 400]:
            continue
        try:
            series, _ = decoder.raw_decode(html[start:])
        except json.JSONDecodeError:
            continue
        if series and isinstance(series[0], dict) and "Elo" in series[0]:
            return series
    raise ValueError("no rating series found in the page")


def fetch_club(slug: str, out_dir: Path) -> dict:
    """Fetch one club and write its raw series. Returns a manifest row."""
    series = extract_series(fetch_page(slug))
    (out_dir / f"{slug}.json").write_text(
        json.dumps(series, indent=1), encoding="utf-8"
    )
    dates = sorted(point["Date"][:10] for point in series)
    return {
        "slug": slug,
        "url": f"{_BASE}/{slug}",
        "points": len(series),
        "first_date": dates[0],
        "last_date": dates[-1],
        "distinct_dates": len(set(dates)),
    }


def main(dest: str, slugs: list[str]) -> int:
    out_dir = _ELO_DIR / dest
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, failed = [], []
    for n, slug in enumerate(slugs):
        if n:
            time.sleep(_DELAY_S)
        try:
            row = fetch_club(slug, out_dir)
        except (urllib.error.URLError, ValueError, OSError) as exc:
            failed.append(slug)
            print(f"  FAIL {slug}: {exc}")
            continue
        rows.append(row)
        print(f"  ok   {slug}: {row['points']} points, "
              f"{row['first_date']} .. {row['last_date']}")

    if rows:
        header = list(rows[0])
        lines = [",".join(header)]
        lines += [",".join(str(row[key]) for key in header) for row in rows]
        (out_dir / "_manifest.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n{len(rows)} ok, {len(failed)} failed -> {out_dir}")
    return 1 if failed else 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2:]))
