"""Exploration results registry — replaces the old results.txt.

Notebooks call `upsert(...)` to record one run. Rows live in
`03_notebooks/model_checks/results.md`, keyed by `source` (the notebook name), so re-runs
update in place. Any leading non-table text (a title/preamble) is preserved.
Convention documented in `06_docs/spaces.md`.

The file is split into one table per gate score — 3xPASS down to 0xPASS — so that how many
gates a run cleared is visible before reading any of its numbers. Sections are rebuilt from
the gate columns on every write, not stored, so a re-run that changes a verdict moves its
row to the right table by itself.
"""
import re
from pathlib import Path

_REGISTRY = Path(__file__).parents[2] / "03_notebooks" / "model_checks" / "results.md"
_SECTION = re.compile(r"^## \d+xPASS")

COLUMNS = ["source", "space", "target", "abt filter", "segmentation",
           "segment filter", "G1", "G2", "G3", "result", "roi_std"]
GATES = ["G1", "G2", "G3"]


def _row(cells):
    return "| " + " | ".join(cells) + " |"


def _passes(cells):
    """How many of the three gates this row cleared."""
    return sum(cells[COLUMNS.index(g)].strip().upper() == "PASS" for g in GATES)


def upsert(source, space, target, abt_filter, segmentation, segment_filter,
           g1, g2, g3, result, roi_std):
    """Insert or replace this run's row (keyed by `source`) in results.md."""
    values = {
        "source": source, "space": space, "target": target, "abt filter": abt_filter,
        "segmentation": segmentation, "segment filter": segment_filter,
        "G1": g1, "G2": g2, "G3": g3, "result": result, "roi_std": str(roi_std),
    }

    # Rows are collected across all sections and re-bucketed below, so a verdict that
    # changed on this run does not leave the row in its old table. The preamble ends at the
    # first section heading — everything from there down is regenerated.
    preamble, rows, in_body = [], {}, False
    if _REGISTRY.exists():
        for line in _REGISTRY.read_text(encoding="utf-8").splitlines():
            if _SECTION.match(line.strip()):
                in_body = True
            elif line.strip().startswith("|"):
                in_body = True
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                if cells == COLUMNS or set("".join(cells)) <= {"-", " "}:
                    continue  # header or separator
                rows[cells[0]] = cells
            elif not in_body:
                preamble.append(line)

    rows[source] = [values[c] for c in COLUMNS]

    blocks = []
    for score in (3, 2, 1, 0):
        keys = sorted(k for k in rows if _passes(rows[k]) == score)
        body = ([_row(COLUMNS), _row(["---"] * len(COLUMNS))] + [_row(rows[k]) for k in keys]
                if keys else ["_none_"])
        blocks.append(f"## {score}xPASS ({len(keys)})\n\n" + "\n".join(body) + "\n")

    text = ("\n".join(preamble).rstrip() + "\n\n" if preamble else "") + "\n".join(blocks)
    _REGISTRY.write_text(text, encoding="utf-8")
