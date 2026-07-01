"""Rebuild the full pipeline and produce the ABT for a league group.

Usage:
    python build_abt.py                   # rebuild the 'major' group (default)
    python build_abt.py --group minor     # rebuild the 'minor' group
    python build_abt.py --skip-raw        # skip match_raw_stats (raw CSVs unchanged)

Each group reads its match CSVs from 01_data/01_raw/01_matches/<group>/ and writes its
features and ABT to 01_data/02_features/<group>/ and 01_data/03_abt/<group>/.
"""
import argparse
import time

from raw.match_raw_stats import build_match_raw_stats
from features.thresholds import build_thresholds
from features.match_info import build_match_info
from features.match_team_stats import build_match_team_stats
from features.match_matchup_stats import build_match_matchup_stats
from features.match_target import build_match_target
from features.abt import build_abt


def _step(n: int, total: int, name: str, fn, group: str):
    print(f"[{n}/{total}] {name}...", end=" ", flush=True)
    t0 = time.time()
    result = fn(group)
    elapsed = time.time() - t0
    if isinstance(result, dict):
        print(f"{result}  ({elapsed:.1f}s)")
    else:
        print(f"{result.shape[0]} rows × {result.shape[1]} cols  ({elapsed:.1f}s)")
    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", default="major",
                        help="league group to build (folder under 01_matches/, default: major)")
    parser.add_argument("--skip-raw", action="store_true",
                        help="skip match_raw_stats (use when raw CSVs are unchanged)")
    parser.add_argument("--skip-thresholds", action="store_true",
                        help="skip build_thresholds (use when thresholds.json already exists)")
    args = parser.parse_args()

    steps = []
    if not args.skip_raw:
        steps.append(("match_raw_stats", build_match_raw_stats))
    if not args.skip_thresholds:
        steps.append(("thresholds", build_thresholds))
    steps += [
        ("match_info",         build_match_info),
        ("match_team_stats",   build_match_team_stats),
        ("match_matchup_stats", build_match_matchup_stats),
        ("match_target",       build_match_target),
        ("abt",                build_abt),
    ]

    total = len(steps)
    t_total = 0.0
    print(f"group: {args.group}")
    for i, (name, fn) in enumerate(steps, 1):
        t_total += _step(i, total, name, fn, args.group)

    print(f"\nDone. Total: {t_total:.1f}s")


if __name__ == "__main__":
    main()
