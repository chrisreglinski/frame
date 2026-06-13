"""Rebuild the full pipeline and produce abt.parquet.

Usage:
    python build_abt.py             # rebuild everything
    python build_abt.py --skip-raw  # skip match_raw_stats (raw CSVs unchanged)
"""
import argparse
import time
from pathlib import Path
import sys

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "02_src" / "01_raw"))
sys.path.insert(0, str(ROOT / "02_src" / "02_features"))

from match_raw_stats import build_match_raw_stats
from match_info import build_match_info
from match_team_stats import build_match_team_stats
from match_matchup_stats import build_match_matchup_stats
from match_target import build_match_target
from abt import build_abt


def _step(n: int, total: int, name: str, fn):
    print(f"[{n}/{total}] {name}...", end=" ", flush=True)
    t0 = time.time()
    df = fn()
    elapsed = time.time() - t0
    print(f"{df.shape[0]} rows × {df.shape[1]} cols  ({elapsed:.1f}s)")
    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-raw", action="store_true",
                        help="skip match_raw_stats (use when raw CSVs are unchanged)")
    args = parser.parse_args()

    steps = []
    if not args.skip_raw:
        steps.append(("match_raw_stats", build_match_raw_stats))
    steps += [
        ("match_info",         build_match_info),
        ("match_team_stats",   build_match_team_stats),
        ("match_matchup_stats", build_match_matchup_stats),
        ("match_target",       build_match_target),
        ("abt",                build_abt),
    ]

    total = len(steps)
    t_total = 0.0
    for i, (name, fn) in enumerate(steps, 1):
        t_total += _step(i, total, name, fn)

    print(f"\nDone. Total: {t_total:.1f}s")


if __name__ == "__main__":
    main()
