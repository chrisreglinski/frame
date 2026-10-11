"""Compact tearsheet: the internal summary of one run, metrics and charts only.

Same computation, panels and styling as tearsheet.render_technical, with no prose. Meant for reviewing
saved models side by side, not for readers outside the project. Sections run from the headline numbers
to the diagnostics: overview, bankroll, profitability, edge, calibration. The bets list is left out.
"""
import pandas as pd

from reporting import panels, render
from reporting import tearsheet as ts


def _forecast_tiles(m):
    """Forecast-quality tiles, when the run carries them (runs saved before BSS was added do not)."""
    if "bss" not in m:
        return ""
    tiles = '<div class="ov-grid">' + "".join([
        ts._attr("BSS model", f"{m['bss']:.4f}", sign=m["bss"],
                 tip="Brier skill score of the model on all predictions, vs the base rate."),
        ts._attr("BSS market", f"{m['bss_market']:.4f}", sign=m["bss_market"],
                 tip="the same for the devigged market probability."),
        ts._attr("BSS skill", f"{m['bss_skill']:+.4f}", sign=m["bss_skill"],
                 tip="model minus market. positive = the model forecasts better than the market."),
    ]) + "</div>"
    return ts._grp("forecast quality", tiles)


def render_compact(run):
    """Render one Run into the compact internal tearsheet (a self-contained HTML string)."""
    model, m = run.model, run.metrics
    preds = run.predictions
    buffer = run.buffer
    bets = preds[preds["model_p"] > preds["implied"] + buffer]
    title = model.name or model.slug

    meta = (f'<p class="caption">{model.slug} · seasons {run.seasons[0]}-{run.seasons[-1]} · '
            f'holdout {run.holdout} · buffer {buffer:.4f} ({run.buffer_name})</p>')

    overview = ts._overview(run).replace(
        '<p class="foot">hover a label or value for details</p>', "")
    overview += _forecast_tiles(m)

    spec_bank = panels.panel_bankroll(bets.sort_values("date"), m["roi"], start=run.bank_start,
                                      kelly_fraction=run.kelly_fraction,
                                      season_label=lambda s: f"{str(s)[:2]}/{str(s)[2:]}")

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

    spec_edge = panels.panel_edge_ranking(preds, window=ts.EDGE_WINDOW, mark_buffer=buffer)

    cal_m_all = panels.panel_calibration_by_model(preds, "all matches")
    cal_m_bet = panels.panel_calibration_by_model(bets, "bet matches")
    cal_k_all = panels.panel_calibration_by_market(preds, "all matches")
    cal_k_bet = panels.panel_calibration_by_market(bets, "bet matches")
    panels.unify_calibration_scale([cal_m_all, cal_m_bet, cal_k_all, cal_k_bet])

    engine = render._CHART_JS.replace(
        "'Libre Franklin','Helvetica Neue',Helvetica,Arial,sans-serif",
        "'Archivo','Helvetica Neue',Arial,sans-serif")

    body = f"""<div class="wrap" lang="en">
<p class="eyebrow">model · summary</p>
<h1>{title}</h1>
{meta}
{ts._chapter("01", "Overview", overview)}
{ts._chapter("02", "Bankroll", render.line_chart(spec_bank))}
{ts._chapter("03", "Profitability", render.line_chart(spec_buf) + tables)}
{ts._chapter("04", "Edge", render.line_chart(spec_edge))}
{ts._chapter("05", "Calibration",
             render.cols(render.line_chart(cal_m_all), render.line_chart(cal_m_bet))
             + render.cols(render.line_chart(cal_k_all), render.line_chart(cal_k_bet)))}
</div>
<div class="frame-tt" id="frame-tt" aria-hidden="true"></div>"""

    return (f"<title>{title}</title>\n<style>{ts.CSS}</style>\n{body}\n"
            f"<script>{engine}</script>\n")
