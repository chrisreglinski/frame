"""HTML rendering for the tearsheet — an artifact-ready, theme-aware, self-contained page.

Charts are drawn in the browser from small data specs (inline SVG + one shared JS engine), not
baked as images, so every chart resolves its colours from CSS tokens and redraws when the viewer
switches light/dark. The page carries no <!doctype>/<html>/<head>/<body> wrapper — it publishes
straight through the Artifact tool and still opens as a standalone file.

Panels hand this layer a spec (axes, ticks, series, reference lines); `line_chart` turns a spec
into a chart card. Monospace stat blocks and layout helpers round out the report.
"""
import itertools
import json
import math


# ---------------------------------------------------------------------------------------------
# Number formatting and axis ticks (shared by panels when they build chart specs)
# ---------------------------------------------------------------------------------------------

def _fmtnum(v, fmt):
    if fmt == "int":   return f"{v:.0f}"
    if fmt == "f1":    return f"{v:.1f}"
    if fmt == "f2":    return f"{v:.2f}"
    if fmt == "f3":    return f"{v:.3f}"
    if fmt == "plus2": return f"{v:+.2f}"
    if fmt == "plus3": return f"{v:+.3f}"
    if fmt == "pct0":  return f"{v * 100:.0f}%"
    if fmt == "pct1":  return f"{v * 100:.1f}%"
    return f"{v:g}"


def ticks(lo, hi, step, fmt="f2"):
    """Evenly spaced ticks from `lo` to `hi` at `step`, each labelled with `fmt`."""
    n = int(round((hi - lo) / step))
    return [{"v": round(lo + i * step, 10),
             "label": _fmtnum(round(lo + i * step, 10), fmt)} for i in range(n + 1)]


def auto_ticks(lo, hi, approx=5, fmt="f2"):
    """~`approx` ticks spanning [lo, hi] on a 1/2/2.5/5 * 10^k step."""
    span = hi - lo
    if span <= 0:
        return [{"v": lo, "label": _fmtnum(lo, fmt)}]
    raw = span / approx
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if raw <= m * mag)
    start = math.ceil(lo / step) * step
    out, v = [], start
    while v <= hi + step * 1e-9:
        out.append(round(v, 10))
        v += step
    return [{"v": float(x), "label": _fmtnum(float(x), fmt)} for x in out]


def fixed_ticks(lo, hi, n=4, fmt="f2"):
    """Exactly `n` evenly spaced ticks on a nice (1/2/2.5/5 * 10^k) step spanning [lo, hi].

    Unlike auto_ticks (which targets a count but yields whatever fits), this guarantees `n` ticks by
    widening the range: it picks the smallest nice step whose n-1 intervals cover the data from a
    floored start, then returns (domain, ticks) with the domain set to the tick range, so ticks sit at
    both ends and the axis may run a little wider than [lo, hi].
    """
    span = (hi - lo) or 1.0
    intervals = max(1, n - 1)
    mag = 10 ** math.floor(math.log10(span / intervals))
    step = intervals * mag  # placeholder, overwritten in the loop below
    for m in (1, 2, 2.5, 5, 10):
        step = m * mag
        t0 = math.floor(lo / step) * step
        if t0 + intervals * step >= hi - step * 1e-9:
            break
    t0 = math.floor(lo / step) * step
    vals = [round(t0 + i * step, 10) for i in range(n)]
    domain = [float(vals[0]), float(vals[-1])]
    return domain, [{"v": float(v), "label": _fmtnum(v, fmt)} for v in vals]


def nice_scale_from_zero(dmax, step_base=0.05, full_cutoff=0.75, fmt="f2"):
    """A scale [0, top] whose ticks sit at 0, step, 2*step, ... for a shared, comparable axis.

    Default is 4 ticks (3 sections) on the smallest step that is a multiple of `step_base` and still
    covers `dmax`. Once that would push the top past `full_cutoff`, the axis instead spans the full
    0 to 1 range with 5 ticks (step 0.25), the only case that uses five ticks. Returns (domain, ticks).
    """
    intervals = 3
    k = 1
    while intervals * k * step_base < dmax - 1e-9:
        k += 1
    step = k * step_base
    top = intervals * step
    if top > full_cutoff + 1e-9:
        top, step, nticks = 1.0, 0.25, 5
    else:
        top, nticks = round(top, 10), 4
    vals = [round(i * step, 10) for i in range(nticks)]
    return [0.0, float(top)], [{"v": float(v), "label": _fmtnum(v, fmt)} for v in vals]


def wrap_scale(dmin, dmax, n=4, base=0.05, fmt="f2"):
    """A scale of exactly `n` ticks that wraps [dmin, dmax] tightly, not anchored to zero.

    The first tick is the largest multiple of `base` at or below dmin; the step is the smallest
    multiple of `base` whose n-1 intervals reach dmax. So every tick is a multiple of `base` and the
    width (n-1 steps) is a multiple of (n-1)*base. Returns (domain, ticks).
    """
    intervals = n - 1
    start = math.floor(dmin / base + 1e-9) * base
    k = 1
    while start + intervals * (k * base) < dmax - 1e-9:
        k += 1
    step = k * base
    vals = [round(start + i * step, 10) for i in range(n)]
    return [float(vals[0]), float(vals[-1])], [{"v": float(v), "label": _fmtnum(v, fmt)} for v in vals]


def pad_domain(lo, hi, frac=0.06):
    """A domain [lo, hi] widened by `frac` of its span on each side (so points don't touch edges)."""
    span = (hi - lo) or 1.0
    return [float(lo - span * frac), float(hi + span * frac)]


# ---------------------------------------------------------------------------------------------
# Charts — browser-drawn inline SVG from a data spec
# ---------------------------------------------------------------------------------------------

_cid = itertools.count(1)


def _esc(text):
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def line_chart(spec):
    """A chart card from a spec: a titled panel holding a responsive inline SVG and a legend.

    The spec (dict) carries: w/h and margins (viewBox geometry), xDomain/yDomain, xTicks/yTicks
    (each {v, label}), series (each {name, color (a CSS var name like '--market'), points [[x,y]],
    width, r marker radius, dash, legend}), refLines ({o: 'h'|'v'|'diag', v, color, label}), axis
    labels, an aria string, and optional tooltip flags. Drawing happens in the browser via the
    shared engine below, so colours track the theme.
    """
    cid = f"c{next(_cid)}"
    spec = {**spec, "id": cid}
    title = spec.get("title")
    title_html = f'<div class="panel-title">{_esc(title)}</div>' if title else ""
    def _entry(swatch_class, color_style, name, desc):
        if name and desc:
            body = f'<b>{_esc(name)}</b> - {_esc(desc)}'
        else:
            body = _esc(name or desc)
        return f'<span><i class="swatch{swatch_class}" style="{color_style}"></i>{body}</span>'

    # A legend row is "keyed" (spread full width, each item with a short description) when its
    # entries carry a `desc`; otherwise it stays a plain left-aligned row of swatch + name.
    shown = [s for s in spec["series"] if s.get("legend", True)]
    ref_items = [r for r in spec.get("refLines", []) if r.get("desc")]
    series = "".join(_entry("", f'background:var({s["color"]})', s["name"], s.get("desc"))
                     for s in shown)
    refs = "".join(_entry(" dash", f'border-top-color:var({r["color"]})', r.get("name", ""), r["desc"])
                   for r in ref_items)
    # Keyed rows (entries carry descriptions) render as an aligned grid whose column count follows
    # the reference row, so the series entries line up column-for-column above the reference buffers.
    series_keyed = any(s.get("desc") for s in shown)
    ncols = len(ref_items) or len(shown)
    grid = f' style="grid-template-columns:repeat({ncols},1fr)"'
    if series_keyed:
        # Keyed layout: series row above an aligned reference row (columns line up).
        legend_html = f'<div class="legend keyed"{grid}>{series}</div>'
        if refs:
            legend_html += f'<div class="legend keyed refs"{grid}>{refs}</div>'
    else:
        # Plain layout: series and reference entries share one inline row.
        legend_html = f'<div class="legend">{series}{refs}</div>'
    svg = (f'<svg id="{cid}" viewBox="0 0 {spec["w"]} {spec["h"]}" role="img" '
           f'aria-label="{_esc(spec.get("aria", ""))}"></svg>')
    reg = f'<script>(window.__frameCharts=window.__frameCharts||[]).push({json.dumps(spec)});</script>'
    zoom_btn = (f'<button class="chart-zoom" type="button" data-chart="{cid}" aria-pressed="false">'
                f'<span class="z-in">unzoom</span><span class="z-out">zoom</span></button>'
                if spec.get("zoom") else "")
    return (f'<div class="panel">{zoom_btn}{title_html}<figure>{svg}{legend_html}</figure></div>{reg}')


# ---------------------------------------------------------------------------------------------
# Monospace stat blocks (aligned text tables)
# ---------------------------------------------------------------------------------------------

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


# (header label, column) for the metric columns shown in a stats table.
_TCOLS = [("matches", "n_matches"), ("breakeven", "breakeven"), ("hit rate", "hit_rate"),
          ("yield", "roi"), ("profit", "profit")]


def _cell(col, value):
    cls = "num"
    if col in ("roi", "profit"):
        cls += " pos" if value > 0 else (" neg" if value < 0 else "")
    return f'<td class="{cls}">{_fmt(col, value)}</td>'


def _srow(label, series, total=False, hl=False):
    cells = "".join(_cell(col, series[col]) for _, col in _TCOLS)
    css = "tot" if total else ("hl" if hl else "")
    attr = f' class="{css}"' if css else ""
    return f"<tr{attr}><th>{label}</th>{cells}</tr>"


def stats_table(title, rows_df, total=None, highlight=()):
    """A stats breakdown as a clean HTML table: one row per group (index = row label), metric names
    in the header, right-aligned tabular numbers, ROI/profit tinted by sign. `total` is an optional
    (label, Series) summary row rendered last with a rule above it; `highlight` is a collection of
    index values whose rows get a subtle tint.
    """
    head = "".join(f"<th>{label}</th>" for label, _ in _TCOLS)
    hl = set(highlight)
    body = [_srow(_row_label(name), row, hl=(name in hl)) for name, row in rows_df.iterrows()]
    if total is not None:
        body.append(_srow(total[0], total[1], total=True))
    return (f'<div class="table-scroll"><table class="stats"><caption>{_esc(title)}</caption>'
            f"<thead><tr><th></th>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>")


# ---------------------------------------------------------------------------------------------
# Page chrome and layout
# ---------------------------------------------------------------------------------------------

def panel(inner):
    """Wrap a fragment in a card panel (matches the chart cards)."""
    return f'<div class="panel">{inner}</div>'


def info_card(title, body, mode="visible", placeholder=None):
    """A titled card (variables, hyperparameters, ...). `mode` is one of:

    - "visible" — show the body.
    - "hidden"  — show `placeholder` instead (title stays, content withheld).
    - "toggle"  — show `placeholder`, with a small reveal control in the title row that swaps in the
      body on click (and hides it again).

    `placeholder` is a collapsed stand-in built like the body, so the withheld state reads as the
    card's first and only entry — e.g. `var_list(["hidden"])` or `param_list([("hidden", "")])`. It
    defaults to a single 'hidden' line. Build `body` with `var_list` or `param_list`.
    """
    ph = placeholder if placeholder is not None else '<div class="vlist"><div>hidden</div></div>'
    toggle = ("" if mode != "toggle" else
              '<button type="button" class="card-toggle" aria-expanded="false">'
              '<span class="t-closed">show</span><span class="t-open">hide</span></button>')
    head = f'<div class="card-head"><span class="card-title">{_esc(title)}</span>{toggle}</div>'

    if mode == "hidden":
        body_html, open_attr = ph, ""
    elif mode == "toggle":
        body_html = f'<div class="card-ph">{ph}</div><div class="card-real">{body}</div>'
        open_attr = ' data-open="0"'
    else:
        body_html, open_attr = body, ""
    return f'<div class="panel"{open_attr}>{head}{body_html}</div>'


def var_list(names):
    """A card body listing names one per line, each with a hanging indent and marker (e.g. features)."""
    return '<div class="vlist">' + "".join(f"<div>{_esc(n)}</div>" for n in names) + "</div>"


def param_list(params):
    """A card body of name → value rows on an aligned key/value grid (e.g. hyperparameters).
    `params` is a dict or a list of pairs."""
    items = params.items() if hasattr(params, "items") else params
    cells = "".join(f'<span class="k">{_esc(k)}</span><span class="v">{_esc(v)}</span>'
                    for k, v in items)
    return f'<div class="kv">{cells}</div>'


def caption(html):
    """A reading note under a panel (serif, muted)."""
    return f'<p class="caption">{html}</p>'


def finding(html, label="finding"):
    """A model-specific observation, set apart from the generic caption.

    Captions describe the instrument and ship with the panel (same text for every model); a finding
    is the analyst's reading of THIS model's result, written after the fact and usually tied to one
    chart. It carries an accent rule and a small label so a reader can tell interpretation from the
    fixed reading notes. Omit it entirely when there is nothing model-specific to say.
    """
    lab = f'<span class="finding-lab">{_esc(label)}</span>' if label else ""
    return f'<div class="finding">{lab}{html}</div>'


def foot(text):
    """A small monospace footnote (units, counts)."""
    return f'<p class="foot">{_esc(text)}</p>'


def filter_table(columns, rows, filters=(), count_noun="rows", metrics=()):
    """A filterable HTML table: chip groups narrow the rows client-side, a live count and aggregate
    metrics of the visible rows update with them.

    `columns` is a list of {key, header, align ('l'|'r'|'c')}. `rows` is a list of dicts; each cell
    value is a plain string, or {text, cls} to tag one cell (e.g. a positive/negative tint). A row also
    carries its raw filter-dimension values under each filter's `dim` key, and, when `metrics` are
    given, a `nums` dict of the numeric fields they aggregate. `filters` is a list of
    {dim, label, values: [(value, label), ...]}, each drawn as a single-select chip group with an
    'all' chip. `metrics` is a list of {label, decimals, suffix, scale} that either sum a field
    (`key`) or take a ratio (`kind: 'ratio'`, `num`, `den`) over the visible rows. Presentation only
    and fully generic: the caller decides every string, colour class and number.
    """
    tid = f"t{next(_cid)}"
    align = {"r": "right", "c": "center"}
    groups = ""
    for f in filters:
        chips = '<button class="chip is-on" data-val="__all">all</button>'
        chips += "".join(f'<button class="chip" data-val="{_esc(v)}">{_esc(lab)}</button>'
                         for v, lab in f["values"])
        groups += (f'<div class="chip-group"><span class="chip-lab">{_esc(f["label"])}</span>'
                   f'<div class="chips" data-dim="{_esc(f["dim"])}">{chips}</div></div>')
    head = "".join(f'<th style="text-align:{align.get(c.get("align"), "left")}">{_esc(c["header"])}</th>'
                   for c in columns)
    num_fields = []
    for m in metrics:
        num_fields += [m["num"], m["den"]] if m.get("kind") == "ratio" else [m["key"]]
    num_fields = list(dict.fromkeys(num_fields))
    dims = [f["dim"] for f in filters]
    body = ""
    for row in rows:
        attrs = "".join(f' data-{d}="{_esc(row[d])}"' for d in dims)
        attrs += "".join(f' data-n-{k}="{float(row["nums"][k]):.6g}"' for k in num_fields)
        cells = ""
        for c in columns:
            v = row[c["key"]]
            cls = f' class="{v["cls"]}"' if isinstance(v, dict) and v.get("cls") else ""
            text = v["text"] if isinstance(v, dict) else v
            cells += f'<td style="text-align:{align.get(c.get("align"), "left")}"{cls}>{_esc(text)}</td>'
        body += f"<tr{attrs}>{cells}</tr>"
    agg = f'<span class="agg-item">{_esc(count_noun)} <b class="bets-count">{len(rows)}</b></span>'
    for m in metrics:
        d = f' data-num="{m["num"]}" data-den="{m["den"]}"' if m.get("kind") == "ratio" \
            else f' data-key="{m["key"]}"'
        agg += (f'<span class="agg-item">{_esc(m["label"])} '
                f'<b class="agg" data-kind="{m.get("kind", "sum")}"{d} data-dec="{m.get("decimals", 1)}" '
                f'data-scale="{m.get("scale", 1)}" data-suffix="{_esc(m.get("suffix", ""))}"></b></span>')
    # A colgroup shared by the header and body tables keeps their columns aligned while the header
    # sits outside the scroll area (so the scrollbar runs beside the rows only, not the header).
    cols = "<colgroup>" + "".join(f'<col style="width:{c["w"]}px">' if c.get("w") else "<col>"
                                  for c in columns) + "</colgroup>"
    return (f'<div class="bets-table" id="{tid}"><div class="bets-bar">{groups}'
            f'<div class="bets-count-wrap">{agg}</div></div>'
            f'<div class="bets-head"><table>{cols}<thead><tr>{head}</tr></thead></table></div>'
            f'<div class="bets-scroll"><table>{cols}<tbody>{body}</tbody></table></div></div>')


def cols(*fragments):
    """Equal-width side-by-side columns that wrap on narrow screens."""
    inner = "".join(f'<div class="col-eq">{f}</div>' for f in fragments)
    return f'<div class="cols-eq">{inner}</div>'


def split_2_1(main, side):
    """A wide left panel beside a narrow right panel; stacks on narrow screens."""
    return (f'<div class="split"><div class="side-main">{main}</div>'
            f'<div class="side-narrow">{side}</div></div>')


def row(*fragments):
    """Lay fragments left-to-right (chart beside stat columns)."""
    return '<div class="hrow">' + "".join(fragments) + "</div>"


_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Libre+Franklin:wght@400;500;600&display=swap');
:root{
  --serif:"Iowan Old Style","Palatino Linotype",Palatino,Georgia,"Times New Roman",serif;
  --ui:"Libre Franklin","Helvetica Neue",Helvetica,Arial,"Liberation Sans",sans-serif;
  --bg:#f6f5f2; --panel:#fffffe; --ink:#1c1d21; --ink-soft:#3a3b40; --muted:#6c6e77;
  --grid:#e7e5df; --axis:#b6b4ac; --hl:#f0efe9; --border:#e7e5df; --accent:#0072b2;
  --shadow:0 1px 2px rgba(0,0,0,.05),0 8px 24px rgba(0,0,0,.06);
  --market:#0072b2; --model:#d55e00; --observed:#1c1d21;
  --profit:#0072b2; --profit-faint:#cdcbc3;
  --mark-left:#009e73; --mark-mid:#cc79a7; --mark-right:#8d8f98; --mark-op:#8d8f98;
  --pos:#0a7d5c; --neg:#b8472f;
}
@media (prefers-color-scheme:dark){ :root:not([data-theme="light"]){
  --bg:#111318; --panel:#181b21; --ink:#eceef2; --ink-soft:#c3c6ce; --muted:#9a9da6;
  --grid:#262a31; --axis:#3f434c; --hl:#20242b; --border:#262a31; --accent:#4aa3e0;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.4);
  --market:#4aa3e0; --model:#f0873f; --observed:#eceef2;
  --profit:#4aa3e0; --profit-faint:#3a3d45;
  --mark-left:#2fbf95; --mark-mid:#e58fc0; --mark-right:#9a9da6; --mark-op:#9a9da6;
  --pos:#2fbf95; --neg:#e0685a;
}}
:root[data-theme="dark"]{
  --bg:#111318; --panel:#181b21; --ink:#eceef2; --ink-soft:#c3c6ce; --muted:#9a9da6;
  --grid:#262a31; --axis:#3f434c; --hl:#20242b; --border:#262a31; --accent:#4aa3e0;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.4);
  --market:#4aa3e0; --model:#f0873f; --observed:#eceef2;
  --profit:#4aa3e0; --profit-faint:#3a3d45;
  --mark-left:#2fbf95; --mark-mid:#e58fc0; --mark-right:#9a9da6; --mark-op:#9a9da6;
  --pos:#2fbf95; --neg:#e0685a;
}
*{box-sizing:border-box}
body{margin:0; background:var(--bg); color:var(--ink); font-family:var(--serif);
     -webkit-font-smoothing:antialiased; line-height:1.5;}
.wrap{max-width:1000px; margin:0 auto; padding:40px 28px 72px;}
.eyebrow{font-family:var(--ui); font-size:12.5px; letter-spacing:.06em; text-transform:uppercase;
         color:var(--muted); margin:0 0 10px;}
h1{font-family:var(--serif); font-weight:600; font-size:30px; line-height:1.15; margin:0 0 12px;
   text-wrap:balance;}
.lede{font-size:16.5px; color:var(--ink-soft); margin:0 0 8px; max-width:66ch;}
.lede b{font-weight:600; color:var(--ink);}
.sec{margin-top:34px;}
h2{font-family:var(--serif); font-weight:600; font-size:19px; margin:0 0 6px; color:var(--ink);}
.panel{background:var(--panel); border:1px solid var(--border); border-radius:14px;
       box-shadow:var(--shadow); padding:20px 20px 14px; margin:16px 0 0; position:relative;}
.chart-zoom{position:absolute; top:12px; right:14px; z-index:2; font-family:var(--ui);
  font-size:11px; letter-spacing:.04em; color:var(--muted); background:none; border:none;
  padding:0; cursor:pointer;}
.chart-zoom:hover{color:var(--accent); text-decoration:underline;}
.chart-zoom .z-out{display:none;}
.chart-zoom[aria-pressed="true"] .z-in{display:none;}
.chart-zoom[aria-pressed="true"] .z-out{display:inline;}
.panel-title{font-family:var(--serif); font-size:15px; color:var(--muted); margin:0 0 6px;
             text-align:center;}
figure{margin:0;}
.panel svg{display:block; width:100%; height:auto; overflow:visible;}
.legend{display:flex; flex-wrap:wrap; gap:16px; padding:8px 4px 2px; margin-top:4px;
        font-family:var(--ui); font-size:12.5px;}
.legend span{display:inline-flex; align-items:center; gap:7px; color:var(--ink);}
.legend .swatch{width:20px; height:3px; border-radius:2px; display:inline-block;}
.caption{font-size:14.5px; color:var(--muted); margin:16px 2px 0; max-width:72ch;}
.caption b{color:var(--ink); font-weight:600;}
.finding{border-left:2px solid var(--accent); padding-left:16px; margin:16px 2px 0; max-width:72ch;
         font-size:15px; color:var(--ink); line-height:1.6;}
.finding-lab{display:block; font-family:var(--ui); font-size:11px; letter-spacing:.08em;
             text-transform:uppercase; color:var(--accent); margin-bottom:4px;}
.foot{font-family:var(--ui); font-size:12px; color:var(--muted); margin:12px 2px 0;}
.table-scroll{overflow-x:auto;}
table.stats{border-collapse:collapse; width:100%; font-family:var(--ui); font-size:12.5px;
            color:var(--ink-soft);}
table.stats caption{font-family:var(--serif); font-size:15px; font-weight:600; color:var(--ink);
                    text-align:left; caption-side:top; margin-bottom:8px;}
table.stats thead th{color:var(--muted); font-weight:600; text-align:right; padding:0 0 6px 16px;
                     border-bottom:1px solid var(--border);}
table.stats tbody th{text-align:left; font-weight:400; color:var(--ink); padding:4px 0; white-space:nowrap;}
table.stats td{text-align:right; padding:4px 0 4px 16px; font-variant-numeric:tabular-nums;}
table.stats tr.tot th{font-weight:600;}
table.stats tr.tot th, table.stats tr.tot td{border-top:1px solid var(--border); padding-top:7px;
                                             color:var(--ink);}
table.stats td.pos{color:var(--pos);} table.stats td.neg{color:var(--neg);}
.card-head{display:flex; align-items:baseline; justify-content:space-between; gap:12px; margin:0 0 10px;}
.card-title{font-family:var(--serif); font-size:15px; font-weight:600; color:var(--ink);}
.card-toggle{font-family:var(--ui); font-size:11.5px; color:var(--muted); background:none;
  border:none; padding:0; cursor:pointer; letter-spacing:.03em;}
.card-toggle:hover{color:var(--ink-soft);}
.card-toggle .t-open{display:none;}
.panel[data-open="1"] .card-toggle .t-closed{display:none;}
.panel[data-open="1"] .card-toggle .t-open{display:inline;}
.panel[data-open="1"] .card-ph{display:none;}
.panel:not([data-open="1"]) .card-real{display:none;}
.kv{display:grid; grid-template-columns:auto 1fr; gap:3px 18px; font-family:var(--ui);
    font-size:12.5px; align-items:baseline;}
.kv .k{color:var(--muted); white-space:nowrap;}
.kv .v{color:var(--ink); text-align:right; font-variant-numeric:tabular-nums;}
.vlist{font-family:var(--ui); font-size:12.5px; color:var(--muted); line-height:1.7;}
.cols-eq{display:flex; flex-wrap:wrap; align-items:stretch; gap:20px;}
.col-eq{flex:1; min-width:300px; display:flex;}
.col-eq > .panel{flex:1; min-width:0; margin-top:0;}
.split{display:flex; flex-wrap:wrap; align-items:flex-start; gap:24px;}
.side-main{flex:2; min-width:340px;}
.side-narrow{flex:1; min-width:240px;}
.hrow{display:flex; align-items:flex-start; gap:24px; flex-wrap:wrap;}
.frame-tt{position:fixed; pointer-events:none; opacity:0; transform:translate(-50%,-115%);
  background:var(--panel); border:1px solid var(--axis); border-radius:9px; padding:8px 10px;
  box-shadow:var(--shadow); font-family:var(--ui); font-size:11.5px; color:var(--ink);
  white-space:nowrap; transition:opacity .09s; z-index:20;}
.frame-tt .row{display:flex; justify-content:space-between; gap:14px;}
.frame-tt .k{color:var(--muted);}
.num{font-variant-numeric:tabular-nums;}
@media print{
  @page{size:A4 portrait; margin:14mm;}
  body{background:#fff;}
  .wrap{max-width:none; padding:0;}
  .panel{box-shadow:none; break-inside:avoid;}
  .sec{break-inside:avoid;}
}
"""

_CHART_JS = """
(function(){
  var NS="http://www.w3.org/2000/svg";
  function css(n){return getComputedStyle(document.documentElement).getPropertyValue(n).trim();}
  function color(c){return (typeof c==="string"&&c[0]==="-")?css(c):c;}
  function el(t,a){var e=document.createElementNS(NS,t);for(var k in a){if(a[k]!=null&&a[k]!=="")e.setAttribute(k,a[k]);}return e;}
  var TT;
  function tip(){if(!TT)TT=document.getElementById("frame-tt");return TT;}
  function f3(v){return v.toFixed(3);}
  var MONO="'Libre Franklin','Helvetica Neue',Helvetica,Arial,sans-serif";
  function draw(spec){
    var svg=document.getElementById(spec.id); if(!svg) return;
    svg.textContent="";
    var z=(spec._zoom&&spec.zoom)?spec.zoom:spec;
    var m=spec.margins,W=spec.w,H=spec.h,x0=z.xDomain[0],x1=z.xDomain[1],y0=z.yDomain[0],y1=z.yDomain[1];
    function px(v){return m.l+(v-x0)/(x1-x0)*(W-m.l-m.r);}
    function py(v){return m.t+(y1-v)/(y1-y0)*(H-m.t-m.b);}
    var grid=css("--grid"),axis=css("--axis"),muted=css("--muted"),ink=css("--ink"),surf=css("--panel");
    (z.yTicks||[]).forEach(function(t){var y=py(t.v);
      svg.appendChild(el("line",{x1:m.l,y1:y,x2:W-m.r,y2:y,stroke:grid,"stroke-width":1}));
      var tx=el("text",{x:m.l-9,y:y+4,"text-anchor":"end",fill:muted,"font-size":11.5,"font-family":MONO});tx.textContent=t.label;svg.appendChild(tx);});
    (z.xTicks||[]).forEach(function(t){var x=px(t.v);
      svg.appendChild(el("line",{x1:x,y1:H-m.b,x2:x,y2:H-m.b+5,stroke:axis,"stroke-width":1}));
      var tx=el("text",{x:x,y:H-m.b+19,"text-anchor":"middle",fill:muted,"font-size":11.5,"font-family":MONO});tx.textContent=t.label;svg.appendChild(tx);});
    (spec.refLines||[]).forEach(function(r){
      if(r.o==="h"){var y=py(r.v);svg.appendChild(el("line",{x1:m.l,y1:y,x2:W-m.r,y2:y,stroke:color(r.color||"--axis"),"stroke-width":1.3,"stroke-dasharray":r.dash?"5 5":""}));}
      else if(r.o==="v"){var x=px(r.v);svg.appendChild(el("line",{x1:x,y1:m.t,x2:x,y2:H-m.b,stroke:color(r.color||"--axis"),"stroke-width":1.2,"stroke-dasharray":"3 4"}));}
      else if(r.o==="diag"){var a=Math.max(x0,y0),b=Math.min(x1,y1);svg.appendChild(el("line",{x1:px(a),y1:py(a),x2:px(b),y2:py(b),stroke:axis,"stroke-width":1.1,"stroke-dasharray":"3 4"}));}
    });
    if(spec.xLabel){var xl=el("text",{x:(m.l+W-m.r)/2,y:H-5,"text-anchor":"middle",fill:ink,"font-size":12.5,"font-family":MONO});xl.textContent=spec.xLabel;svg.appendChild(xl);}
    if(spec.yLabel){var cy=(m.t+H-m.b)/2;var yl=el("text",{x:15,y:cy,"text-anchor":"middle",fill:ink,"font-size":12.5,"font-family":MONO,transform:"rotate(-90 15 "+cy+")"});yl.textContent=spec.yLabel;svg.appendChild(yl);}
    spec.series.forEach(function(s){var col=color(s.color);
      if(s.points.length>1){var d="";s.points.forEach(function(p,i){d+=(i?"L":"M")+px(p[0]).toFixed(1)+" "+py(p[1]).toFixed(1)+" ";});
        svg.appendChild(el("path",{d:d,fill:"none",stroke:col,"stroke-width":s.width||2,"stroke-linejoin":"round","stroke-linecap":"round","stroke-dasharray":s.dash?"5 5":"",opacity:s.opacity!=null?s.opacity:1}));}
      var r=s.r||0; if(r>0) s.points.forEach(function(p){svg.appendChild(el("circle",{cx:px(p[0]),cy:py(p[1]),r:r+1.5,fill:surf}));svg.appendChild(el("circle",{cx:px(p[0]),cy:py(p[1]),r:r,fill:col}));});
    });
    (spec.refLines||[]).forEach(function(r){
      if(r.o==="v"&&r.label){var x=px(r.v);var t=el("text",{x:x+4,y:m.t+11,fill:color(r.color||"--axis"),"font-size":10.5,"font-family":MONO});t.textContent=r.label;svg.appendChild(t);}
      else if(r.o==="h"&&r.label){var y=py(r.v);var t=el("text",{x:W-m.r-2,y:y-4,"text-anchor":"end",fill:color(r.color||"--axis"),"font-size":10.5,"font-family":MONO});t.textContent=r.label;svg.appendChild(t);} });
    if(spec.tip){var T=tip();var xs=spec.series[0].points.map(function(p){return p[0];});
      xs.forEach(function(xv,i){var x=px(xv);var lp=i>0?(x-px(xs[i-1]))/2:16;var np=i<xs.length-1?(px(xs[i+1])-x)/2:16;
        var hit=el("rect",{x:x-lp,y:m.t,width:lp+np,height:H-m.t-m.b,fill:"transparent"});hit.style.cursor="crosshair";
        hit.addEventListener("mousemove",function(ev){T.style.opacity=1;T.style.left=ev.clientX+"px";T.style.top=ev.clientY+"px";
          var xf=spec.tipXFmt==="int"?String(Math.round(xv)):f3(xv);
          var h="<b>"+(spec.tipXLabel||"x")+" "+xf+"</b>";
          var rows=spec.tipData?spec.tipData[i]:spec.series.map(function(s){return [s.name,f3(s.points[i][1])];});
          rows.forEach(function(r){h+='<div class="row"><span class="k">'+r[0]+'</span><span class="num">'+r[1]+'</span></div>';});
          T.innerHTML=h;});
        hit.addEventListener("mouseleave",function(){T.style.opacity=0;});svg.appendChild(hit);});
    }
  }
  function drawAll(){(window.__frameCharts||[]).forEach(draw);}
  window.__frameDrawAll=drawAll;
  function applyFilters(root){ if(!root) return; var active={};
    root.querySelectorAll(".chips").forEach(function(g){var on=g.querySelector(".chip.is-on");
      active[g.getAttribute("data-dim")]=on?on.getAttribute("data-val"):"__all";});
    var vis=0, sums={};
    root.querySelectorAll("tbody tr").forEach(function(tr){var show=true;
      for(var dim in active){ if(active[dim]!=="__all"&&tr.getAttribute("data-"+dim)!==active[dim]){show=false;break;} }
      tr.hidden=!show;
      if(show){vis++; for(var i=0;i<tr.attributes.length;i++){var a=tr.attributes[i];
        if(a.name.indexOf("data-n-")===0){var k=a.name.slice(7); sums[k]=(sums[k]||0)+parseFloat(a.value);}}}});
    var c=root.querySelector(".bets-count"); if(c)c.textContent=vis;
    root.querySelectorAll(".agg").forEach(function(b){var val;
      if(b.getAttribute("data-kind")==="ratio"){var n=sums[b.getAttribute("data-num")]||0,
        d=sums[b.getAttribute("data-den")]||0; val=d?n/d:0;}
      else{val=sums[b.getAttribute("data-key")]||0;}
      b.textContent=(val*parseFloat(b.getAttribute("data-scale")||"1")).toFixed(+b.getAttribute("data-dec"))
        +(b.getAttribute("data-suffix")||"");}); }
  function equalizeCards(){
    document.querySelectorAll("[data-equal-cards]").forEach(function(box){
      var reals=[].slice.call(box.querySelectorAll(".card-real")); if(reals.length<2) return;
      var max=0;
      reals.forEach(function(r){var d=r.style.display; r.style.minHeight=""; r.style.display="block";
        if(r.offsetHeight>max)max=r.offsetHeight; r.style.display=d;});
      reals.forEach(function(r){r.style.minHeight=max+"px";});});
  }
  function boot(){drawAll();
    document.querySelectorAll(".bets-table").forEach(function(t){
      var chips=t.querySelectorAll(".chip:not([data-val='__all'])"),w=0;
      chips.forEach(function(c){w=Math.max(w,c.offsetWidth);});
      chips.forEach(function(c){c.style.width=w+"px";});
      applyFilters(t);});
    equalizeCards();
    if(document.fonts&&document.fonts.ready)document.fonts.ready.then(equalizeCards);}
  if(document.readyState==="loading")document.addEventListener("DOMContentLoaded",boot);else boot();
  new MutationObserver(drawAll).observe(document.documentElement,{attributes:true,attributeFilter:["data-theme"]});
  try{matchMedia("(prefers-color-scheme:dark)").addEventListener("change",drawAll);}catch(e){}
  document.addEventListener("click",function(ev){
    var chip=ev.target&&ev.target.closest?ev.target.closest(".chip"):null;
    if(chip){var grp=chip.closest(".chips");
      if(grp){grp.querySelectorAll(".chip").forEach(function(c){c.classList.remove("is-on");});
        chip.classList.add("is-on");applyFilters(chip.closest(".bets-table"));} return;}
    var sth=ev.target&&ev.target.closest?ev.target.closest(".bets-head th"):null;
    if(sth){var tbl=sth.closest(".bets-table"), scr=tbl&&tbl.querySelector(".bets-scroll tbody");
      if(scr){var rows=[].slice.call(scr.children), ci=sth.cellIndex,
        re=/^[+-]?[0-9][0-9,]*[.]?[0-9]*%?$/,
        numeric=rows.every(function(r){var c=r.children[ci],t=(c?c.textContent:"").trim(); return t===""||re.test(t);}),
        dir=sth.getAttribute("data-sort")==="asc"?-1:1;
        rows.sort(function(a,b){var x=a.children[ci].textContent.trim(),y=b.children[ci].textContent.trim();
          return numeric?(((parseFloat(x.replace(/,/g,""))||0)-(parseFloat(y.replace(/,/g,""))||0))*dir):x.localeCompare(y)*dir;});
        rows.forEach(function(r){scr.appendChild(r);});
        tbl.querySelectorAll(".bets-head th").forEach(function(h){h.removeAttribute("data-sort");});
        sth.setAttribute("data-sort", dir===1?"asc":"desc");}
      return;}
    var zb=ev.target&&ev.target.closest?ev.target.closest(".chart-zoom"):null;
    if(zb){var id=zb.getAttribute("data-chart");
      var spec=(window.__frameCharts||[]).filter(function(s){return s.id===id;})[0];
      if(spec){spec._zoom=!spec._zoom;zb.setAttribute("aria-pressed",spec._zoom?"true":"false");draw(spec);} return;}
    var b=ev.target&&ev.target.closest?ev.target.closest(".card-toggle"):null; if(!b) return;
    var card=b.closest(".panel"); if(!card) return;
    var open=card.getAttribute("data-open")==="1";
    card.setAttribute("data-open",open?"0":"1");
    b.setAttribute("aria-expanded",open?"false":"true");
  });
})();
"""


def write_report(path, title, sections, *, eyebrow=None, lede=None):
    """Write the artifact-ready HTML report. `sections` is a list of (heading, html_fragment)
    tuples; `eyebrow` and `lede` are optional intro lines above the first section. The output has
    no document wrapper, so it publishes directly as an Artifact and opens as a standalone file.
    """
    parts = ['<div class="wrap">']
    if eyebrow:
        parts.append(f'<p class="eyebrow">{eyebrow}</p>')
    parts.append(f"<h1>{_esc(title)}</h1>")
    if lede:
        parts.append(f'<p class="lede">{lede}</p>')
    for heading, fragment in sections:
        head = f"<h2>{_esc(heading)}</h2>" if heading else ""
        parts.append(f'<section class="sec">{head}{fragment}</section>')
    parts.append("</div>")
    parts.append('<div class="frame-tt" id="frame-tt" aria-hidden="true"></div>')

    html = (f"<title>{_esc(title)}</title>\n<style>{_CSS}</style>\n"
            + "\n".join(parts)
            + f"\n<script>{_CHART_JS}</script>\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path
