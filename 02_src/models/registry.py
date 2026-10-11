"""The model repository as a view over 04_models/<problem>/<solution>/params.yaml.

There is no in-code list of models. `get` and `names` scan the tree and load each definition. A model
is keyed by its human name if it has one, else its full slug. Human names are unique across the whole
repository. `table` and `write_index` render the view: named models first, then every model grouped
by problem. Since 04_models is gitignored, a fresh checkout has no models until params files exist.
"""
import os
from pathlib import Path

import pandas as pd
import yaml

from models._spec import model_problem, model_solution
from models.loader import load_model
from models.store import ROOT


def _index(root=ROOT):
    """name-or-slug -> (Model, model folder) for every params.yaml two levels under `root`."""
    models = {}
    for params in sorted(Path(root).glob("*/*/params.yaml")):
        model = load_model(params)
        key = model.name or model.slug
        if model.name and key in models:
            raise ValueError(f"model name {key!r} is used twice: {models[key][1]} and {params.parent}")
        models[key] = (model, params.parent)
    return models


def get(name, root=ROOT):
    """The model registered under `name` (its human name or, failing that, its full slug)."""
    models = _index(root)
    try:
        return models[name][0]
    except KeyError:
        raise KeyError(f"no model named {name!r}; known: {sorted(models)}") from None


def names(root=ROOT):
    """All model names (or slugs) found under `root`."""
    return sorted(_index(root))


def _best_run(model_folder):
    """The model's saved run with the highest profit, as (run folder, results dict), or None."""
    runs = []
    for res_path in (model_folder / "runs").glob("*/results.yaml"):
        res = yaml.safe_load(res_path.read_text(encoding="utf-8"))
        runs.append((res["metrics"]["profit"], res_path.parent, res))
    if not runs:
        return None
    _, folder, res = max(runs, key=lambda r: r[0])
    return folder, res


def table(root=ROOT):
    """One row per model: name, problem, solution, its size (spaces, variables) and the headline
    metrics of its best run. Sorted by problem, then by number of spaces, then by profit."""
    rows = []
    for model, folder in _index(root).values():
        best = _best_run(folder)
        m = best[1]["metrics"] if best else {}
        rows.append({
            "name": model.name, "problem": model_problem(model), "solution": model_solution(model),
            "n_spaces": len(model.spaces), "n_vars": len(model.features),
            "profit": m.get("profit"), "roi": m.get("roi"), "n_bets": m.get("n_bets"),
            "p_value": m.get("p_value"), "bank_cagr": m.get("bank_cagr"),
            "bank_maxdd": m.get("bank_maxdd"), "run": best[0] if best else None,
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (df.sort_values(["problem", "n_spaces", "profit"], ascending=[True, True, False])
              .reset_index(drop=True))


def _index_rows(df, root, with_problem):
    cells = []
    for r in df.itertuples(index=False):
        link = ""
        if r.run is not None:
            for report in ("report-compact.html", "report.html"):
                if (Path(r.run) / report).exists():
                    href = os.path.relpath(Path(r.run) / report, root).replace(os.sep, "/")
                    link = f'<a href="{href}">report</a>'
                    break
        num = lambda v, f: "" if v is None or pd.isna(v) else f.format(v)
        cells.append(
            "<tr>"
            + (f"<td>{r.name}</td><td>{r.problem}</td>" if with_problem else "")
            + f"<td class='r'>{r.n_spaces}</td><td class='r'>{r.n_vars}</td>"
            + f"<td class='mono'>{r.solution}"
            + (f" <b>· {r.name}</b>" if r.name and not with_problem else "") + "</td>"
            + f"<td class='r'>{num(r.profit, '{:+.1f}')}</td><td class='r'>{num(r.roi, '{:+.1%}')}</td>"
            + f"<td class='r'>{num(r.n_bets, '{:.0f}')}</td><td class='r'>{num(r.p_value, '{:.4f}')}</td>"
            + f"<td class='r'>{num(r.bank_cagr, '{:+.0%}')}</td><td class='r'>{num(r.bank_maxdd, '{:.0%}')}</td>"
            + f"<td>{link}</td></tr>")
    return "".join(cells)


_HEAD = ("<tr>{extra}<th class='r'>spaces</th><th class='r'>vars</th><th>solution</th><th class='r'>profit</th><th class='r'>yield</th>"
         "<th class='r'>bets</th><th class='r'>p value</th><th class='r'>CAGR</th>"
         "<th class='r'>MDD</th><th></th></tr>")

_CSS = """
:root { --bg:#fbfaf7; --ink:#1d2433; --muted:#6b7280; --line:#e5e2da; --accent:#1f5f4a; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#14181f; --ink:#e7e9ee; --muted:#9aa3b2; --line:#2a303a; --accent:#7cc4a6; }
}
body { background:var(--bg); color:var(--ink); font:15px/1.5 'Helvetica Neue',Arial,sans-serif;
       margin:0; padding:32px 16px; }
.wrap { max-width:1100px; margin:0 auto; }
h1 { font-size:26px; margin:0 0 4px; } h2 { font-size:20px; margin:40px 0 4px; }
h3 { font-size:15px; margin:22px 0 8px; color:var(--muted); }
p.muted { color:var(--muted); margin:0; }
.scroll { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:14px; }
th, td { padding:6px 10px; border-bottom:1px solid var(--line); text-align:left; white-space:nowrap; }
th { color:var(--muted); font-weight:600; }
.r { text-align:right; font-variant-numeric:tabular-nums; }
.mono { font-family:Menlo,Consolas,monospace; font-size:13px; }
a { color:var(--accent); }
"""


def write_index(root=ROOT):
    """Write 04_models/index.html: named models on top, then every model grouped by problem and, inside
    a problem, by number of spaces (6 down to 1). Each row links to its run's compact report. Returns
    the path."""
    root = Path(root)
    df = table(root)
    named = df[df["name"] != ""] if not df.empty else df
    parts = [f"<h1>Models</h1><p class='muted'>{len(df)} models in "
             f"{df['problem'].nunique() if not df.empty else 0} problems</p>",
             "<h2>Named</h2>"]
    if named.empty:
        parts.append("<p class='muted'>none yet</p>")
    else:
        parts.append("<div class='scroll'><table>"
                     + _HEAD.format(extra="<th>name</th><th>problem</th>")
                     + _index_rows(named.sort_values("profit", ascending=False), root, True)
                     + "</table></div>")
    for problem, group in (df.groupby("problem") if not df.empty else []):
        parts.append(f"<h2>{problem} <span class='muted'>({len(group)})</span></h2>")
        for k in range(max(6, int(group["n_spaces"].max())), 0, -1):
            sub = group[group["n_spaces"] == k].sort_values("profit", ascending=False)
            label = f"{k} space" + ("s" if k > 1 else "")
            parts.append(f"<h3>{label} <span class='muted'>({len(sub)})</span></h3>")
            if sub.empty:
                parts.append("<p class='muted'>none yet</p>")
            else:
                parts.append("<div class='scroll'><table>" + _HEAD.format(extra="")
                             + _index_rows(sub, root, False) + "</table></div>")
    html = (f"<!doctype html><meta charset='utf-8'><title>Models</title><style>{_CSS}</style>"
            f"<div class='wrap'>{''.join(parts)}</div>\n")
    path = root / "index.html"
    path.write_text(html, encoding="utf-8")
    return path
