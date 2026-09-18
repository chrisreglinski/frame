"""The model tearsheet: a one-page report rendering one Run.

Everything specific to this report type lives here: the section order, the prose (ledes and captions),
the stylesheet and the page skeleton. It consumes a finished Run and draws it, so it trains nothing
and imports no estimator. The chart recipes come from reporting.panels, the layout primitives from
reporting.render.

The split of responsibilities: panels are chart recipes (spec only), render holds layout primitives
and the chart engine, and this module composes them and writes all the prose. A caption is fixed for
the report type and lives here; a finding is model-specific and travels on the model.
"""
import pandas as pd

from reporting import panels, render

# Rolling window (in matches) for the edge-ranking curve.
EDGE_WINDOW = 300

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@400&family=Fraunces:opsz,wght@9..144,400&display=swap');
:root{
  --serif:"Fraunces",Georgia,"Times New Roman",serif;
  --ui:"Archivo","Helvetica Neue",Arial,sans-serif;
  --bg:#f4f0e6; --panel:#fffffe; --ink:#221a16; --ink-soft:#4a3f39; --muted:#8a7f74;
  --border:#e7e0d3; --grid:#e3e3e1; --axis:#bfbfbd; --hl:#eee6d8; --accent:#a6192e; --numeral:#1e3a8a;
  --shadow:0 1px 2px rgba(0,0,0,.05),0 8px 24px rgba(0,0,0,.06);
  --market:#8a8275; --model:#1e3a8a; --observed:#2d5a27;
  --profit:#1e3a8a; --profit-faint:#d8d0c0;
  --mark-left:#8e908f; --mark-mid:#8e908f; --mark-right:#8e908f; --mark-op:#8e908f;
  --pos:#2d5a27; --neg:#a6192e;
}
@media (prefers-color-scheme:dark){ :root:not([data-theme="light"]){
  --bg:#17130f; --panel:#211b16; --ink:#ece6dc; --ink-soft:#cabfb0; --muted:#9a9084;
  --border:#332c25; --grid:#2c2e2d; --axis:#474a48; --hl:#211b16; --accent:#db5f74; --numeral:#8ea6c4;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.45);
  --market:#9a9084; --model:#8ea6c4; --observed:#6bb15f;
  --profit:#8ea6c4; --profit-faint:#3a342c;
  --mark-left:#9a9c9b; --mark-mid:#9a9c9b; --mark-right:#9a9c9b; --mark-op:#9a9c9b;
  --pos:#6bb15f; --neg:#db5f74;
}}
:root[data-theme="dark"]{
  --bg:#17130f; --panel:#211b16; --ink:#ece6dc; --ink-soft:#cabfb0; --muted:#9a9084;
  --border:#332c25; --grid:#2c2e2d; --axis:#474a48; --hl:#211b16; --accent:#db5f74; --numeral:#8ea6c4;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.45);
  --market:#9a9084; --model:#8ea6c4; --observed:#6bb15f;
  --profit:#8ea6c4; --profit-faint:#3a342c;
  --mark-left:#9a9c9b; --mark-mid:#9a9c9b; --mark-right:#9a9c9b; --mark-op:#9a9c9b;
  --pos:#6bb15f; --neg:#db5f74;
}
*{box-sizing:border-box}
body{margin:0; background:var(--bg); color:var(--ink); font-family:var(--ui);
     -webkit-font-smoothing:antialiased; line-height:1.5;}
.wrap{max-width:1060px; margin:0 auto; padding:48px 30px 80px;}
.eyebrow{font-family:var(--ui); font-size:12px; font-weight:400; letter-spacing:.12em;
         text-transform:uppercase; color:var(--numeral); margin:0 0 14px;}
h1{font-family:var(--serif); font-weight:400; font-size:40px; line-height:1.05; margin:0 0 14px;
   letter-spacing:-.01em; text-wrap:balance; color:var(--ink);}
.lede{font-size:17px; color:var(--ink-soft); margin:40px 0 52px; line-height:1.6; text-align:justify;
      hyphens:auto;}
.lede b{color:var(--ink); font-weight:400;}
.assump-lead{font-size:17px; color:var(--ink-soft); margin:16px 0 0;}
.assump{margin:8px 0 0; padding-left:20px; color:var(--ink-soft); font-size:17px; line-height:1.6;}
.assump li{margin:5px 0;}
.ov-grid{display:grid; grid-template-columns:repeat(4,1fr); gap:12px;}
.ov-wide{grid-column:span 2; min-width:0; display:flex; align-self:start;}
.ov-wide > .panel{flex:1; min-width:0; margin-top:0; padding:16px 18px;}
.ov-wide > .panel:not([data-open="1"]) .card-head{margin-bottom:0;}
.ov-wide .card-title{font-size:13px; letter-spacing:.05em;}
.ov-wide .card-toggle{font-size:13px;}
.ov-wide .vlist{line-height:1.5; color:var(--muted);}
.attr{display:flex; align-items:baseline; justify-content:space-between; gap:14px; position:relative;
      background:var(--panel); border:1px solid var(--border); border-radius:12px;
      box-shadow:var(--shadow); padding:16px 18px; min-width:0;}
.attr .lab{font-size:13px; letter-spacing:.05em; text-transform:uppercase; color:var(--muted);
           white-space:nowrap;}
.attr .val{font-size:13px; color:var(--ink); font-variant-numeric:tabular-nums; text-align:right;
  word-break:break-word;}
.attr.pos .val{color:var(--pos);} .attr.neg .val{color:var(--neg);}
.attr [data-tip]{position:relative; cursor:help;}
.attr [data-tip]::after{content:attr(data-tip); position:absolute; left:50%; bottom:calc(100% + 8px);
  transform:translateX(-50%); z-index:20; width:max-content; max-width:220px; background:var(--panel);
  color:var(--ink); border:1px solid var(--border); border-radius:9px; box-shadow:var(--shadow);
  padding:8px 11px; font-size:12px; line-height:1.45; text-align:left; text-transform:none;
  letter-spacing:0; font-variant-numeric:normal; white-space:normal; opacity:0; pointer-events:none;
  transition:opacity .12s;}
.attr [data-tip]:hover::after{opacity:1;}
.grp{margin-top:24px;} .grp:first-child{margin-top:16px;}
.grp-lab{font-size:11px; letter-spacing:.08em; text-transform:uppercase; color:var(--muted);
         margin:0 0 8px;}
.vlist div{padding-left:12px;}
@media (max-width:720px){ .ov-grid{grid-template-columns:repeat(2,1fr);} .ov-wide{grid-column:span 2;} }
.chap{margin-top:52px;}
.chap-head{display:flex; align-items:baseline; gap:16px; border-bottom:2px solid var(--numeral);
           padding-bottom:8px; margin-bottom:6px;}
.chap-num{font-family:var(--serif); font-size:22px; font-weight:400; color:var(--numeral);
          font-variant-numeric:lining-nums;}
h2{font-family:var(--serif); font-weight:400; font-size:23px; margin:0; color:var(--ink);
   letter-spacing:-.01em;}
.panel{background:var(--panel); border:1px solid var(--border); border-radius:14px;
       box-shadow:var(--shadow); padding:20px 20px 14px; margin:16px 0 0; position:relative;}
.panel-title{font-family:var(--ui); font-size:12px; font-weight:400; letter-spacing:.04em;
             text-transform:uppercase; color:var(--muted); margin:0 0 6px; text-align:center;}
.chart-zoom{position:absolute; top:12px; right:14px; z-index:2; font-family:var(--ui);
  font-size:11px; letter-spacing:.04em; color:var(--muted); background:none; border:none;
  padding:0; cursor:pointer;}
.chart-zoom:hover{color:var(--accent); text-decoration:underline;}
.chart-zoom .z-out{display:none;}
.chart-zoom[aria-pressed="true"] .z-in{display:none;}
.chart-zoom[aria-pressed="true"] .z-out{display:inline;}
figure{margin:0;}
.panel svg{display:block; width:100%; height:auto; overflow:visible;}
.legend{display:flex; flex-wrap:wrap; gap:16px; padding:8px 4px 2px; margin-top:4px;
        font-family:var(--ui); font-size:12.5px;}
.legend span{display:inline-flex; align-items:center; gap:7px; color:var(--ink);}
.legend .swatch{width:20px; height:3px; border-radius:2px; display:inline-block;}
.legend.keyed{display:grid; gap:6px 12px; align-items:start;}
.legend.keyed span{color:var(--muted); align-items:baseline;}
.legend.keyed b{color:var(--ink); font-weight:400;}
.legend.keyed.refs{padding-top:8px;}
@media (max-width:600px){ .legend.keyed{grid-template-columns:1fr !important;} }
.swatch.dash{height:0; border-top:2px dashed; border-radius:0; position:relative; top:-3px;}
.caption{font-size:14px; color:var(--muted); margin:14px 2px 0;}
.caption.spaced{margin-bottom:52px;}
.caption.sec-intro{text-align:justify; hyphens:auto;}
.caption b{color:var(--ink); font-weight:400;}
.finding{border-left:2px solid var(--accent); padding-left:16px; margin:16px 2px 0; max-width:72ch;
         font-size:15px; color:var(--ink-soft); line-height:1.6;}
.finding-lab{display:block; font-family:var(--ui); font-size:11px; letter-spacing:.08em;
             text-transform:uppercase; color:var(--accent); margin-bottom:4px;}
.finding b{color:var(--ink); font-weight:400;}
.foot{font-size:12px; color:var(--muted); margin:12px 2px 0;}
.bets-bar{display:flex; flex-wrap:wrap; align-items:center; gap:12px 20px; margin:40px 2px 22px;}
.chip-group{display:flex; align-items:center; gap:8px;}
.bets-bar > .chip-group:first-child{order:1;}
.bets-bar > .chip-group:nth-child(2){order:3; flex-basis:100%;}
.chip-lab{font-family:var(--ui); font-size:11px; letter-spacing:.08em; text-transform:uppercase;
  color:var(--muted); display:inline-block; width:54px;}
.chips{display:flex; flex-wrap:wrap; gap:6px;}
.chip{font-family:var(--ui); font-size:12px; color:var(--ink-soft); background:none;
  border:1px solid var(--border); border-radius:999px; padding:3px 11px; cursor:pointer;
  font-variant-numeric:tabular-nums;}
.chip:hover{border-color:var(--muted);}
.chip.is-on{background:var(--numeral); color:#fff; border-color:var(--numeral);}
.bets-count-wrap{order:2; margin-left:auto; display:flex; flex-wrap:wrap; gap:6px 4px;
  font-family:var(--ui); font-size:12.5px; color:var(--muted);}
.agg-item{display:inline-flex; gap:9px; width:92px; white-space:nowrap; justify-content:flex-end;}
.bets-count, .agg{text-align:left; font-variant-numeric:tabular-nums; color:var(--ink);}
.bets-head{background:var(--panel); overflow-y:auto; scrollbar-gutter:stable;
  border:1px solid var(--border); border-radius:12px 12px 0 0;}
.bets-scroll{max-height:420px; overflow-y:auto; scrollbar-gutter:stable;
  border:1px solid var(--border); border-top:none; border-radius:0 0 12px 12px;}
.bets-table table{width:100%; border-collapse:collapse; table-layout:fixed; font-family:var(--ui);
  font-size:12.5px; font-variant-numeric:tabular-nums;}
.bets-table thead th{background:var(--panel); color:var(--muted); font-weight:400; font-size:11px;
  letter-spacing:.05em; text-transform:uppercase; padding:9px 12px; white-space:nowrap;
  cursor:pointer; user-select:none;}
.bets-table thead th:hover{color:var(--ink-soft);}
.bets-table thead th[data-sort]{color:var(--ink);}
.bets-table thead th[data-sort="asc"]::after{content:"↑"; margin-left:3px;}
.bets-table thead th[data-sort="desc"]::after{content:"↓"; margin-left:3px;}
.bets-table td{padding:7px 12px; border-bottom:1px solid var(--hl); color:var(--ink);
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;}
.bets-table tbody tr:hover td{background:var(--hl);}
.bets-table td.pos{color:var(--pos);} .bets-table td.neg{color:var(--neg);}
.bets-table td.won{color:var(--pos);} .bets-table td.lost{color:var(--muted);}
.cols-eq{display:flex; flex-wrap:wrap; align-items:stretch; gap:20px; margin-top:16px;}
.col-eq{flex:1; min-width:300px; display:flex;}
.col-eq > .panel{flex:1; min-width:0; margin-top:0;}
.card-head{display:flex; align-items:baseline; justify-content:space-between; gap:12px; margin:0 0 10px;}
.card-title{font-family:var(--ui); font-size:12px; font-weight:400; letter-spacing:.04em;
            text-transform:uppercase; color:var(--muted);}
.card-toggle{font-family:var(--ui); font-size:11.5px; color:var(--accent); background:none;
  border:none; padding:0; cursor:pointer; letter-spacing:.03em; font-weight:400;}
.card-toggle:hover{text-decoration:underline;}
.card-toggle .t-open{display:none;}
.panel[data-open="1"] .card-toggle .t-closed{display:none;}
.panel[data-open="1"] .card-toggle .t-open{display:inline;}
.panel[data-open="1"] .card-ph{display:none;}
.panel:not([data-open="1"]) .card-real{display:none;}
.kv{display:grid; grid-template-columns:auto 1fr; gap:3px 18px; font-family:var(--ui);
    font-size:13px; align-items:baseline;}
.kv .k{color:var(--muted); white-space:nowrap; padding-left:12px;}
.kv .v{color:var(--ink); text-align:right; font-variant-numeric:tabular-nums; font-weight:400;}
.vlist{font-family:var(--ui); font-size:13px; color:var(--ink-soft); line-height:1.75;}
.table-scroll{overflow-x:auto;}
table.stats{border-collapse:collapse; width:100%; font-family:var(--ui); font-size:13px;
            color:var(--ink-soft);}
table.stats caption{font-family:var(--ui); font-size:12px; font-weight:400; letter-spacing:.04em;
                    text-transform:uppercase; color:var(--muted); text-align:left;
                    caption-side:top; margin-bottom:10px;}
table.stats thead th{color:var(--muted); font-weight:400; text-align:right; padding:0 0 7px 16px;
                     border-bottom:1px solid var(--border); font-size:12px;}
table.stats tbody th{text-align:left; font-weight:400; color:var(--ink); padding:5px 0; white-space:nowrap;}
table.stats td{text-align:right; padding:5px 0 5px 16px; font-variant-numeric:tabular-nums;}
table.stats tr.tot th{font-weight:400;}
table.stats tr.tot th, table.stats tr.tot td{border-top:1px solid var(--border); padding-top:8px;
                                             color:var(--ink);}
table.stats td.pos{color:var(--pos);} table.stats td.neg{color:var(--neg);}
table.stats tr.hl th, table.stats tr.hl td{font-weight:600;}
.frame-tt{position:fixed; pointer-events:none; opacity:0; transform:translate(-50%,-115%);
  background:var(--panel); border:1px solid var(--border); border-radius:11px; padding:9px 12px;
  box-shadow:0 6px 24px rgba(0,0,0,.14); font-family:var(--ui); font-size:11.5px; color:var(--ink);
  white-space:nowrap; transition:opacity .09s; z-index:20;}
.frame-tt .row{display:flex; justify-content:space-between; gap:14px;}
.frame-tt .k{color:var(--muted);}
.num{font-variant-numeric:tabular-nums;}
@media print{
  @page{size:A4 portrait; margin:13mm;}
  body{background:#fff;}
  .wrap{max-width:none; padding:0;}
  .panel,.chap{break-inside:avoid;}
}
"""


def _attr(label, value, sign=None, tip=None, tip_val=None):
    """One Overview tile: an uppercase label and a value, each with an optional hover tip. A label tip
    describes the concept (fixed for the report type), a value tip describes this run's number."""
    cls = "attr"
    if sign is not None:
        cls += " pos" if sign > 0 else (" neg" if sign < 0 else "")
    lab_data = f' data-tip="{tip}"' if tip else ""
    val_data = f' data-tip="{tip_val}"' if tip_val else ""
    return (f'<div class="{cls}"><div class="lab"{lab_data}>{label}</div>'
            f'<div class="val"{val_data}>{value}</div></div>')


def _grp(label, inner):
    return f'<div class="grp"><div class="grp-lab">{label}</div>{inner}</div>'


def _chapter(num, question, body):
    return (f'<section class="chap"><div class="chap-head">'
            f'<div class="chap-num">{num}</div><h2>{question}</h2></div>{body}</section>')


# ---- prose (fixed for the report type) -------------------------------------------------------
# Season counts are read from the run, so the copy never lies about the data it summarizes: an intro
# built for five seasons says "five", not "four".
_NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
              8: "eight", 9: "nine", 10: "ten", 11: "eleven", 12: "twelve"}


def _num_word(n):
    """The English word for a small count, falling back to digits past twelve."""
    return _NUM_WORDS.get(n, str(n))


# Display phrasing for a league set. The set membership lives in 01_data/league_sets.yaml (the single
# source); this is only how the report names a set in prose. Unknown sets fall back to the raw name.
_LEAGUE_SETS = {
    "major": {"phrase": "top five European leagues",
              "members": "England, France, Germany, Italy, Spain"},
}


def _league_phrase(name):
    """The prose name for a league set, e.g. major -> "top five European leagues"."""
    return _LEAGUE_SETS.get(name, {}).get("phrase", name)


def _league_tip_val(name):
    """The Overview leagues-tile tooltip: the phrase and, when known, the member leagues."""
    info = _LEAGUE_SETS.get(name)
    if not info:
        return None
    members = info.get("members")
    return f"the {info['phrase']}" + (f": {members}." if members else ".")


def _intro(n_seasons, leagues_phrase):
    w = _num_word(n_seasons)
    return (
        '<p class="lede">This tearsheet report showcases promising, profitable football betting models. '
        "In simple terms, a model is profitable when betting on the match outcomes it rates higher than "
        'the market gives a positive return. The models are trained and assessed on the last '
        f'{w} full seasons from the {leagues_phrase}. Training and assessment use {w}-fold '
        'cross-validation, with each season serving as the test set once. All results are reported under '
        f'the following assumptions, unless noted otherwise: results are pooled across the {w} test sets; '
        'profitability is measured against margin-inclusive market prematch odds; stakes are proportional '
        "to the market-implied probability, weighting every match equally with no bankroll-dependent "
        'staking.</p>'
    )


def _profit_intro(n_dev):
    return (
        "Is the model profitable? "
        "A model is profitable when backing the match outcomes it rates higher than the market gives a "
        "positive return. Its value, however, lies not in any single threshold, but in its "
        "ability to consistently separate bets worth taking from those that are not. The buffer controls "
        "how selective the strategy is: lower values admit more marginal opportunities, while higher "
        "values require a larger model-market gap and therefore a stronger estimated edge. If that "
        "separation is real, profitability should remain broadly positive across a range of thresholds "
        "rather than depend on one finely tuned cutoff. To avoid in-sample optimization, the profit curve "
        f"pools the first {_num_word(n_dev)} seasons, and the selected buffer is then evaluated on the "
        "held-out season."
    )

_CAL_INTRO = (
    "Is the model calibrated? A model is calibrated when its probability estimates match the actual "
    "frequencies of the outcome. Perfect calibration is not necessary to find value bets, but it is "
    "crucial for sound staking and risk management. The market publishes its own estimate through "
    "the odds, a natural "
    "benchmark, so we read calibration through two lenses: one model-centric, one market-centric."
)

_EDGE_INTRO = (
    "Is the edge a true measure of the advantage? The edge on a match is the difference between the "
    "model's probability and the market's implied probability. Ordering matches by that difference "
    "shows whether a larger edge corresponds to a larger advantage over the price, and whether that "
    "advantage matches the edge in size."
)

def _bankroll_intro(n_seasons):
    return (
        "Is the profitability steady? Profitability measures the edge per unit staked, but a bankroll "
        "depends also on how much is risked on each bet and on the order in which wins and losses arrive. "
        "Staking each bet at a half-Kelly fraction of the running bankroll and compounding across the "
        f"{_num_word(n_seasons)} seasons shows whether that edge accumulates into real growth, and how "
        "steadily. For a more realistic estimate of the edge, the stakes use the portfolio's realized "
        "yield rather than the edge from the model's own probabilities."
    )

_BETS_INTRO = "How does the model bet on specific matches? Explore the list of all placed bets."


def _overview(run, variables=None, hyperparameters=None):
    model, m = run.model, run.metrics
    preds = run.predictions

    # The variables and hyperparameters slots default to the model's own. The caller can pass
    # replacement content, e.g. ["hidden"] / {"hidden": ""}, to withhold them from a public build. The
    # report does not interpret the content, it just renders what it is given.
    feats = list(model.features) if variables is None else variables
    hps = model.hyperparams if hyperparameters is None else hyperparameters
    vars_card = render.info_card("variables", render.var_list(feats), mode="toggle", placeholder="")
    hp_card = render.info_card("hyperparameters", render.param_list(hps), mode="toggle", placeholder="")
    settings = '<div class="ov-grid" data-equal-cards>' + "".join([
        _attr("method", model.method, tip="the algorithm the model is built on."),
        _attr("leagues", model.domain.leagues, tip="the leagues the model is trained and evaluated on.",
              tip_val=_league_tip_val(model.domain.leagues)),
        _attr("domain", f"gameweek &gt; {model.domain.gameweek_min}",
              tip="the slice of matches the model runs on.",
              tip_val="only matches once both teams are past their eighth game of the season, so "
                      "early-season noise is dropped."),
        _attr("matches", f"{len(preds)}", tip="matches left after the filters, that the model scored."),
        f'<div class="ov-wide">{vars_card}</div>',
        f'<div class="ov-wide">{hp_card}</div>',
    ]) + "</div>"

    results = '<div class="ov-grid">' + "".join([
        _attr("bets", f"{m['n_bets']}",
              tip="matches the model chose to back, its edge above the buffer."),
        _attr("breakeven", f"{m['breakeven'] * 100:.1f}%",
              tip="the draw rate the bets would need only to break even at these prices."),
        _attr("hit rate", f"{m['hit_rate'] * 100:.1f}%", tip="share of the bets that won."),
        _attr("p value", f"{m['p_value']:.3f}",
              tip="chance of a run this strong under the market devigged probabilities. small means "
                  "unlikely to be luck."),
        _attr("staked", f"{m['staked']:.1f}",
              tip="total staked, each bet sized proportional to its implied probability."),
        _attr("won", f"{int(m['wins'])}",
              tip="bets that won. each win returns 1, so this is also the money they returned."),
        _attr("profit", f"{m['profit']:+.1f}", sign=m["profit"], tip="won minus staked."),
        _attr("yield", f"{m['roi'] * 100:+.1f}%", sign=m["roi"],
              tip="return per unit staked (profit over staked), the edge on turnover."),
    ]) + "</div>"

    bank_profit = m["bank_final"] - run.bank_start
    financial = '<div class="ov-grid">' + "".join([
        _attr("investment", f"{run.bank_start:.0f}",
              tip="the example starting bankroll for the financial results."),
        _attr("profit", f"{bank_profit:+.1f}", sign=bank_profit,
              tip="growth of the bankroll, compounding each bet at half-Kelly."),
        _attr("MDD", f"{m['bank_maxdd'] * 100:.1f}%", sign=m["bank_maxdd"],
              tip="max drawdown: the deepest percentage fall from the running peak."),
        _attr("cagr", f"{m['bank_cagr'] * 100:+.1f}%", sign=m["bank_cagr"],
              tip="compound annual growth rate of the bankroll over the betting period."),
    ]) + "</div>"

    return (_grp("settings", settings) + _grp("model results", results)
            + _grp("financial results", financial)
            + '<p class="foot">hover a label or value for details</p>')


def _bet_rows(bets):
    rows = []
    for r in bets.itertuples(index=False):
        imp, mp, won = float(r.implied), float(r.model_p), int(r.y) == 1
        edge = mp - imp
        profit = (1 - imp) if won else (-imp)
        try:
            d = pd.Timestamp(r.date).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            d = str(r.date)
        rows.append({
            "season": str(r.season), "league": str(r.league),
            "date": d, "league_txt": str(r.league), "home": str(r.home), "away": str(r.away),
            "odds": f"{1 / imp:.2f}", "implied_p": f"{imp:.2f}", "model_p": f"{mp:.2f}",
            "edge": f"{edge:+.2f}",
            "result": {"text": "won" if won else "lost", "cls": "won" if won else "lost"},
            "profit": {"text": f"{profit:+.2f}", "cls": "pos" if profit > 0 else "neg"},
            "nums": {"staked": imp, "profit": profit},
        })
    return rows


_BET_COLS = [
    {"key": "date", "header": "date", "w": 108},
    {"key": "league_txt", "header": "league", "w": 92},
    {"key": "home", "header": "home team", "w": 116},
    {"key": "away", "header": "away team", "w": 116},
    {"key": "odds", "header": "odds", "align": "r", "w": 70},
    {"key": "implied_p", "header": "implied prob.", "align": "r", "w": 120},
    {"key": "model_p", "header": "model prob.", "align": "r", "w": 110},
    {"key": "edge", "header": "edge", "align": "r", "w": 82},
    {"key": "result", "header": "result", "align": "r", "w": 84},
    {"key": "profit", "header": "profit", "align": "r", "w": 82},
]
_BET_METRICS = [
    {"key": "staked", "label": "staked", "decimals": 1},
    {"key": "profit", "label": "profit", "decimals": 1},
    {"kind": "ratio", "num": "profit", "den": "staked", "label": "yield", "decimals": 1,
     "scale": 100, "suffix": "%"},
]


def render_tearsheet(run, variables=None, hyperparameters=None):
    """Render one Run into a self-contained tearsheet (an HTML string, no external assets).

    `variables` and `hyperparameters` default to the model's own; pass replacement content (for
    example ["hidden"] and {"hidden": ""}) to withhold them from a public build. Season counts in the
    prose are read from the run, so the copy matches the data it summarizes.
    """
    model, m = run.model, run.metrics
    preds = run.predictions
    buffer = run.buffer
    bets = preds[preds["model_p"] > preds["implied"] + buffer]

    n_seasons = len(run.seasons)
    n_dev = n_seasons - 1

    buffer_name = run.buffer_name
    if buffer_name == "operational":
        buffer_phrase = "the operational buffer"
    else:
        buffer_phrase = (f"the {buffer_name} buffer selected earlier using the "
                         f"{_num_word(n_dev)}-season data")

    overview = _overview(run, variables=variables, hyperparameters=hyperparameters)

    # Profitability: buffer chosen on the development seasons, scored across all.
    preds_dev = preds[preds["season"] != run.holdout]
    spec_buf, _ = panels.panel_buffer_profit(preds_dev, operational=buffer)
    season_df = panels.group_stats(preds, "season", buffer)
    league_df = panels.group_stats(preds, "league", buffer)
    ps_total = pd.Series({"n_matches": m["n_bets"], "breakeven": m["breakeven"],
                          "hit_rate": m["hit_rate"], "roi": m["roi"], "profit": m["profit"]})
    tables = render.cols(
        render.panel(render.stats_table("per season", season_df, total=("all seasons", ps_total))),
        render.panel(render.stats_table("per league", league_df)),
    )
    ch_profit = _chapter("03", "Profitability",
                         f'<p class="lede">{_profit_intro(n_dev)}</p>'
                         + render.line_chart(spec_buf)
                         + f'<p class="caption spaced">{buffer_name.capitalize()} buffer is chosen to '
                           'score the model.</p>'
                         + tables
                         + render.caption("Per-season and per-league model checks, scored at "
                                          f"{buffer_phrase}."))

    # Calibration: four panels on one shared scale.
    cal_m_all = panels.panel_calibration_by_model(preds, "all matches")
    cal_m_bet = panels.panel_calibration_by_model(bets, "bet matches")
    cal_k_all = panels.panel_calibration_by_market(preds, "all matches")
    cal_k_bet = panels.panel_calibration_by_market(bets, "bet matches")
    panels.unify_calibration_scale([cal_m_all, cal_m_bet, cal_k_all, cal_k_bet])
    ch_cal = _chapter("04", "Calibration",
                      f'<p class="lede">{_CAL_INTRO}</p>'
                      + render.cols(render.line_chart(cal_m_all), render.line_chart(cal_m_bet))
                      + '<p class="caption spaced">Bucketed by model probability: observed frequency '
                        'against the market. The dashed diagonal is perfect calibration of the model.'
                        '<br>Observed above the market line means the bucket is profitable.</p>'
                      + render.cols(render.line_chart(cal_k_all), render.line_chart(cal_k_bet))
                      + render.caption("Bucketed by market probability: observed frequency against the "
                                       "model. The dashed diagonal is perfect calibration of the market."
                                       "<br>Observed above the dashed diagonal means the bucket is "
                                       "profitable."))

    # Edge.
    spec_edge = panels.panel_edge_ranking(preds, window=EDGE_WINDOW, mark_buffer=buffer)
    ch_edge = _chapter("05", "Edge",
                       f'<p class="lede">{_EDGE_INTRO}</p>'
                       + render.line_chart(spec_edge)
                       + render.caption("Matches ranked by edge (model_p − implied), high to low. "
                                        f"Each rate is a rolling mean over {EDGE_WINDOW} matches."))

    # Bankroll.
    spec_bank = panels.panel_bankroll(bets.sort_values("date"), m["roi"], start=run.bank_start,
                                      kelly_fraction=run.kelly_fraction,
                                      season_label=lambda s: f"{str(s)[:2]}/{str(s)[2:]}")
    ch_bankroll = _chapter("06", "Bankroll",
                           f'<p class="lede">{_bankroll_intro(n_seasons)}</p>'
                           + render.line_chart(spec_bank)
                           + render.caption("Bankroll from a 100 unit start, compounding bet by bet "
                                            f"across the {_num_word(n_seasons)} seasons in date order. "
                                            "Stakes are half-Kelly on the market price lifted by the "
                                            "portfolio yield."))

    # Bets.
    bets_disp = bets.sort_values("date")
    bet_filters = [
        {"dim": "season", "label": "season",
         "values": [(s, f"{s[:2]}/{s[2:]}") for s in sorted(bets["season"].astype(str).unique())]},
        {"dim": "league", "label": "league",
         "values": [(lg, lg) for lg in sorted(bets["league"].astype(str).unique())]},
    ]
    ch_bets = _chapter("07", "Bets",
                       f'<p class="lede">{_BETS_INTRO}</p>'
                       + render.filter_table(_BET_COLS, _bet_rows(bets_disp), bet_filters,
                                             count_noun="bets", metrics=_BET_METRICS))

    engine = render._CHART_JS.replace(
        "'Libre Franklin','Helvetica Neue',Helvetica,Arial,sans-serif",
        "'Archivo','Helvetica Neue',Arial,sans-serif")

    body = f"""<div class="wrap" lang="en">
<p class="eyebrow">model · tearsheet</p>
<h1>{model.name} - {model.tagline}</h1>
{_chapter("01", "Intro", _intro(n_seasons, _league_phrase(model.domain.leagues)))}
{_chapter("02", "Overview", overview)}
{ch_profit}
{ch_cal}
{ch_edge}
{ch_bankroll}
{ch_bets}
</div>
<div class="frame-tt" id="frame-tt" aria-hidden="true"></div>"""

    return (f"<title>{model.name}</title>\n<style>{CSS}</style>\n{body}\n"
            f"<script>{engine}</script>\n")
