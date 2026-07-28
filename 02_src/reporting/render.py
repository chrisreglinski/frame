"""HTML rendering helpers for the tearsheet — turn figures and tables into a self-contained page.

Everything is embedded (figures as base64 PNGs, tables as inline HTML) so the report is a single
file you can open or send without any assets alongside it.
"""
import base64
import io

import matplotlib.pyplot as plt


def fig_to_uri(fig, dpi=200):
    """Render a matplotlib figure to a base64 data URI and close it."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"


def image_html(fig):
    """A figure as an <img> fragment that fills its container width."""
    return f'<img src="{fig_to_uri(fig)}" style="width:100%">'


def two_col(left, right):
    """Two side-by-side columns, roughly 50/50."""
    return f'<div class="cols"><div class="col">{left}</div><div class="col">{right}</div></div>'


def table_html(df, float_fmt="{:.3f}"):
    """A DataFrame as an HTML table fragment, with NaNs blanked."""
    def fmt(x):
        return float_fmt.format(x) if isinstance(x, float) and x == x else ("" if x != x else str(x))
    return df.to_html(float_format=lambda x: fmt(x), border=0, na_rep="")


_LABELS = {"n_matches": "matches", "breakeven": "breakeven", "hit_rate": "hit rate", "roi": "ROI"}


def _fmt(col, value):
    if col == "n_matches":
        return f"{int(value)}"
    if col == "profit":
        return f"{value:+.1f}"
    if col == "roi":
        return f"{value * 100:+.1f}%"
    return f"{value * 100:.1f}%"


def _row_label(name):
    text = str(name)
    if len(text) == 4 and text.isdigit():
        return f"season {text[:2]}/{text[2:]}"
    return text


# (label, column, value-field width) — fixed widths so metric names line up as a text table.
_COLS = [("matches", "n_matches", 4), ("breakeven", "breakeven", 6),
         ("hit rate", "hit_rate", 6), ("ROI", "roi", 6)]
_LABEL_WIDTH = 14


def _line(label, row, bold=False):
    shown = f"<b>{label}</b>" if bold else label
    pad = " " * max(1, _LABEL_WIDTH - len(label))
    cells = "   ".join(f"{name} {_fmt(col, row[col]):>{width}}" for name, col, width in _COLS)
    return shown + pad + cells


def stats_block(title, all_stats, group_df):
    """A threshold section as an aligned monospace block: a bold title, an 'all' row, then one row
    per group. Fixed-width fields line the metric names up under each other."""
    lines = [f"<b>{title}</b>", "  " + _line("all", all_stats)]
    for name, row in group_df.iterrows():
        lines.append("  " + _line(_row_label(name), row, bold=True))
    return '<div class="blk">' + "\n".join(lines) + "</div>"


# Compact table variant for narrow columns: metric names once as a header row, values below.
# (label, column, value-field width) — each column is as wide as the wider of its label and values,
# so a long two-word label like "hit rate" fills its column exactly, like the others.
_TCOLS = [("matches", "n_matches", 4), ("breakeven", "breakeven", 5),
          ("hit rate", "hit_rate", 5), ("ROI", "roi", 6), ("profit", "profit", 6)]
_TSEP = "  "
_TLABEL_WIDTH = 13


def _colwidth(label, value_width):
    return max(len(label), value_width)


# One uniform width for every metric column (the widest label/value across all of them).
_UNIFORM_W = max(_colwidth(label, w) for label, _, w in _TCOLS)


def _trow(label, cells, bold=False):
    shown = f"<b>{label}</b>" if bold else label
    return shown + " " * max(1, _TLABEL_WIDTH - len(label)) + cells


_TINDENT = "    "


def _cells(row):
    return _TSEP.join(f"{_fmt(col, row[col]):>{_UNIFORM_W}}" for _, col, _w in _TCOLS)


def stats_header():
    """The metric-name header row, rendered once above a set of stats_table blocks. Its columns line
    up with every block's rows because they share the same indent, label width and column widths.
    Padding stays as real spaces and only the label text is bolded, so alignment is preserved."""
    header = _TSEP.join(" " * (_UNIFORM_W - len(label)) + f"<b>{label}</b>"
                        for label, _, _w in _TCOLS)
    return '<div class="blk">' + _TINDENT + _trow("", header) + "</div>"


def stats_table(title, all_stats, group_df):
    """Compact block for a narrow column: a bold title, one row per group, and an 'all seasons'
    summary row last. Metric names are not repeated here — they come from stats_header() above."""
    body = [_trow(_row_label(name), _cells(row), bold=True) for name, row in group_df.iterrows()]
    body.append(_trow("all seasons", _cells(all_stats), bold=True))

    lines = [f"<b>{title}</b>"] + [_TINDENT + line for line in body]
    return '<div class="blk">' + "\n".join(lines) + "</div>"


def stats_block_rows(title, rows_df, highlight=False):
    """A block like stats_table but with arbitrary labelled rows (index = row label) and no
    all-seasons summary. Columns align with stats_header. Used for the inverted out-of-sample
    block: a 'season 25/26' title with the left/peak/right buffers indented beneath it."""
    body = [_TINDENT + _trow(str(name), _cells(row), bold=True) for name, row in rows_df.iterrows()]
    lines = [f"<b>{title}</b>"] + body
    css = "blk hl" if highlight else "blk"
    return f'<div class="{css}">' + "\n".join(lines) + "</div>"


def cols(*fragments):
    """Equal-width side-by-side columns."""
    inner = "".join(f'<div class="col-eq">{f}</div>' for f in fragments)
    return f'<div class="cols-eq">{inner}</div>'


def split_2_1(main, side):
    """A wide left panel (2/3) beside a narrow right panel (1/3)."""
    return f'<div class="split"><div class="side-main">{main}</div><div class="side-narrow">{side}</div></div>'


def row(*fragments):
    """Lay fragments left-to-right (chart beside stat columns)."""
    return '<div class="hrow">' + "".join(fragments) + "</div>"


_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
 body {{ font-family: "Consolas", ui-monospace, "Cascadia Code", monospace; margin: 32px; color: #222; }}
 h1 {{ font-size: 20px; }} h2 {{ font-size: 15px; margin-top: 28px; }}
 table {{ border-collapse: collapse; font-size: 13px; margin: 8px 0; }}
 th, td {{ padding: 4px 10px; text-align: right; border-bottom: 1px solid #eee; }}
 .section {{ margin-bottom: 24px; }}
 .cols {{ display: flex; align-items: flex-start; gap: 40px; }}
 .col {{ flex: 1; min-width: 0; }}
 .col:last-child {{ padding-top: 14px; border-left: 1px solid #eee; padding-left: 32px; }}
 .cols-eq {{ display: flex; align-items: flex-start; gap: 32px; margin-top: 18px; }}
 .col-eq {{ flex: 1; min-width: 0; text-align: center; }}
 .split {{ display: flex; align-items: stretch; gap: 36px; }}
 .side-main {{ flex: 1; min-width: 0; }}
 .side-narrow {{ flex: 1; min-width: 0; padding-top: 6px; }}
 .side-narrow .blk {{ display: block; text-align: left; margin-bottom: 16px; }}
 .side-narrow .blk:last-child {{ margin-bottom: 0; }}
 .hl {{ background: #fbecea; padding: 10px 0; margin-top: 6px; }}
 .hrow {{ display: flex; align-items: flex-start; gap: 28px; flex-wrap: wrap; }}
 .blk {{ font-family: "Consolas", ui-monospace, "Cascadia Code", monospace;
         font-size: 15px; white-space: pre; line-height: 1.1; margin-bottom: 20px; color: #444;
         display: inline-block; text-align: left; }}
 .blk:last-child {{ margin-bottom: 0; }}
</style></head><body>
<h1>{title}</h1>
{body}
</body></html>"""


def write_report(path, title, sections):
    """Write a full HTML page. `sections` is a list of (heading, html_fragment) tuples."""
    body = "\n".join(
        f'<div class="section"><h2>{heading}</h2>{fragment}</div>'
        for heading, fragment in sections
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(_PAGE.format(title=title, body=body))
    return path
