"""Exploration results registry — replaces the old results.txt.

Notebooks call `upsert(...)` to record one run. Rows are a markdown table in
`03_notebooks/model_checks/results.md`, keyed by `source` (the notebook name), so
re-runs update in place. Any leading non-table text (a title/preamble) is preserved.
Convention documented in `06_docs/spaces.md`.
"""
from pathlib import Path

_REGISTRY = Path(__file__).parents[2] / "03_notebooks" / "model_checks" / "results.md"

COLUMNS = ["source", "space", "target", "abt filter", "segmentation",
           "segment filter", "G1", "G2", "G3", "result", "roi_std"]


def _row(cells):
    return "| " + " | ".join(cells) + " |"


def upsert(source, space, target, abt_filter, segmentation, segment_filter,
           g1, g2, g3, result, roi_std):
    """Insert or replace this run's row (keyed by `source`) in results.md."""
    values = {
        "source": source, "space": space, "target": target, "abt filter": abt_filter,
        "segmentation": segmentation, "segment filter": segment_filter,
        "G1": g1, "G2": g2, "G3": g3, "result": result, "roi_std": str(roi_std),
    }

    preamble, rows, in_table = [], {}, False
    if _REGISTRY.exists():
        for line in _REGISTRY.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("|"):
                in_table = True
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                if cells == COLUMNS or set("".join(cells)) <= {"-", " "}:
                    continue  # header or separator
                rows[cells[0]] = cells
            elif not in_table:
                preamble.append(line)

    rows[source] = [values[c] for c in COLUMNS]

    table = [_row(COLUMNS), _row(["---"] * len(COLUMNS))] + [_row(rows[k]) for k in sorted(rows)]
    text = ("\n".join(preamble).rstrip() + "\n\n" if preamble else "") + "\n".join(table) + "\n"
    _REGISTRY.write_text(text, encoding="utf-8")
