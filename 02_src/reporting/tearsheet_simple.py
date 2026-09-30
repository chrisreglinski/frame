"""Sybilla-simple: the plain-language variant of the tearsheet.

Same computation, panels and styling as tearsheet.render_technical. Only the prose is rewritten for a
non-specialist and the sections are reordered to read as a story. This module supplies the text and
the order, and reuses the shared machinery from reporting.tearsheet (the stylesheet, the chart engine,
the chapter and overview builders, the bets-table columns).

Explanations the technical report keeps in hover tooltips are moved into visible text here, and a
plain-words glossary of the scoreboard is spelled out under the Overview.
"""
import pandas as pd

from reporting import panels, render
from reporting import tearsheet as ts


def _para(text):
    return f'<p class="lede">{text}</p>'


def _cap(text):
    return render.caption(text)


def _overview_block(run, variables, hyperparameters):
    m = run.metrics
    tiles = ts._overview(run, variables=variables, hyperparameters=hyperparameters)
    tiles = tiles.replace('<p class="foot">hover a label or value for details</p>', "")
    tiles = tiles.replace(">cagr</div>", ">CAGR</div>")   # uppercase, to match MDD

    intro = _para(
        "First, the scoreboard. Three little clusters of numbers: what the model is, how its bets did, "
        "and what real money would have done. If any of the terms could use a plain-English "
        "translation, there is a short glossary tucked in just below.")

    glossary_list = (
        '<ul class="assump">'
        "<li><b>bets</b>: how many matches Sybilla actually put money on across the four seasons.</li>"
        "<li><b>hit rate</b>: how often those bets landed on a draw. It looks low, and it should, "
        "because draws are rare. You do not need to win often, because a draw pays back more than "
        "double, so being right well under half the time still turns a profit.</li>"
        "<li><b>staked</b>: the total money that passed through the model, larger stakes on the "
        "likelier draws and smaller ones on the long shots. Read it as turnover, not as money "
        "lost.</li>"
        "<li><b>yield</b>: the number that matters. Profit divided by everything staked. At "
        f"{m['roi'] * 100:+.1f}%, every 100 units run through Sybilla handed back about "
        f"{abs(m['roi'] * 100):.0f} in profit.</li>"
        "<li><b>breakeven</b>: the draw rate you would need just to not lose at these prices. Beat it "
        "and you are ahead.</li>"
        "<li><b>p value</b>: the odds that a run this good was dumb luck. Small means it probably was "
        "not.</li>"
        "<li><b>investment, profit, MDD, CAGR</b>: the real-money row. Start with 100, reinvest, and "
        "this is how it grows, how deep the worst fall gets (MDD, the maximum drawdown), and the pace "
        "per year (CAGR).</li>"
        "</ul>")
    glossary = ('<div class="grp" style="margin-top:26px">'
                '<div class="grp-lab">what the numbers mean</div>' + glossary_list + '</div>')
    return intro + tiles + glossary


def render_simple(run, variables=None, hyperparameters=None):
    """Render one Run into the plain-language tearsheet (a self-contained HTML string)."""
    m = run.metrics
    preds = run.predictions
    buffer = run.buffer
    bets = preds[preds["model_p"] > preds["implied"] + buffer]

    # ---- Overview -----------------------------------------------------------------------------
    ch_overview = ts._chapter("02", "The scoreboard",
                              _overview_block(run, variables, hyperparameters))

    # ---- Profitability ------------------------------------------------------------------------
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
    ch_profit = ts._chapter("03", "Does it actually make money",
                            _para(
                                "Sybilla is picky. It sits out most matches and only bets when its "
                                "own estimate beats the price by a clear margin. Think of that margin "
                                "as a dial. Crank it up and the model bets only on its strongest "
                                "disagreements. Ease it down and it bets more often, dipping into "
                                "games where it barely disagrees. The chart turns that dial across its "
                                "whole range, and it shows the thing that really matters: the profit "
                                "does not hang on one lucky setting.")
                            + render.line_chart(spec_buf)
                            + _cap("How to read it. Bottom axis: the dial, the disagreement the model "
                                   "demands before it bets. Side axis: total profit. Follow it from "
                                   "right to left. As you lower the bar and let more bets in, profit "
                                   "keeps climbing while those extra bets still carry value, then "
                                   "flattens out roughly where the model stops beating the price. Real "
                                   "skill looks like this smooth slope, not a lonely spike. The dashed "
                                   "line is the setting we actually use.")
                            + '<p class="lede" style="margin-top:44px">Now the same profit cut two '
                              "ways, by season and by league. If it only worked in one year, or only "
                              "in one country, you should be suspicious. It does not.</p>"
                            + tables
                            + _cap("matches is the number of bets in that slice, hit rate how many "
                                   "landed, and yield the profit per unit staked."))

    # ---- Bets ---------------------------------------------------------------------------------
    bets_disp = bets.sort_values("date")
    bet_filters = [
        {"dim": "season", "label": "season",
         "values": [(s, f"{s[:2]}/{s[2:]}") for s in sorted(bets["season"].astype(str).unique())]},
        {"dim": "league", "label": "league",
         "values": [(lg, lg) for lg in sorted(bets["league"].astype(str).unique())]},
    ]
    ch_bets = ts._chapter("04", "The bets themselves",
                          _para(
                              "Enough theory. Here is every bet, one match per row: the odds on "
                              "offer, Sybilla's own estimate, how far the two disagree (the edge), "
                              "whether the match ended level, and what you would have won or lost. "
                              "Filter by season or league, tap a header to sort. This is the model "
                              "with its cards face up. Not an equation, a list of calls you could have "
                              "made yourself.")
                          + render.filter_table(ts._BET_COLS, ts._bet_rows(bets_disp), bet_filters,
                                                count_noun="bets", metrics=ts._BET_METRICS))

    # ---- Bankroll -----------------------------------------------------------------------------
    spec_bank = panels.panel_bankroll(bets.sort_values("date"), m["roi"], start=run.bank_start,
                                      kelly_fraction=run.kelly_fraction,
                                      season_label=lambda s: f"{str(s)[:2]}/{str(s)[2:]}")
    ch_bank = ts._chapter("05", "What happens to the money",
                          _para(
                              "Everything above assumed you stake the same tiny amount every time. "
                              "That is clean for measuring skill, but nobody actually bets like that. "
                              "In real life you let it ride: win, and there is a little more for the "
                              "next one. So start with 100, reinvest as you go, bet in the true order "
                              "the matches happened, and each time risk only a careful slice of what "
                              "you are holding. Different question now. Not how good is the edge, but "
                              "what would the money have felt like.")
                          + render.line_chart(spec_bank)
                          + _cap("How to read it. Bottom axis: time, one step per bet. Side axis: the "
                                 "pot, starting at 100 (the dashed line). Up means it is winning, the "
                                 "dips are losing streaks. The deepest drop from a previous peak is "
                                 "the gut-check number, the maximum drawdown."))

    # ---- Calibration --------------------------------------------------------------------------
    cal_m_all = panels.panel_calibration_by_model(preds, "all matches")
    cal_m_bet = panels.panel_calibration_by_model(bets, "bet matches")
    cal_k_all = panels.panel_calibration_by_market(preds, "all matches")
    cal_k_bet = panels.panel_calibration_by_market(bets, "bet matches")
    panels.unify_calibration_scale([cal_m_all, cal_m_bet, cal_k_all, cal_k_bet])
    ch_cal = ts._chapter("06", "Can you trust its guesses",
                         _para(
                             "Profit is not the only thing worth trusting. You also want to know "
                             "whether the model is honest about its own confidence. Honest means that "
                             "among the games it calls 30% likely to draw, roughly 30% really do. "
                             "Sybilla is not perfectly honest here, and that is fine. It still wins, "
                             "because it is good at spotting which bets are worth taking even when the "
                             "exact percentage is a little off. These charts hold its guess up against "
                             "reality, and against the bookmaker.")
                         + render.cols(render.line_chart(cal_m_all), render.line_chart(cal_m_bet))
                         + _cap("How to read it. Each dot is a bunch of matches the model scored about "
                                "the same. Bottom axis: the model's probability. Side axis: how often "
                                "the draw truly happened in that bunch. The dashed diagonal is perfect "
                                "honesty, where the guess meets reality. A dot sitting above the other "
                                "line means that bunch of bets made money.")
                         + render.cols(render.line_chart(cal_k_all), render.line_chart(cal_k_bet))
                         + _cap("The same idea flipped to the bookmaker's side: matches grouped by the "
                                "market's price, with the model laid over the top. Above the diagonal, "
                                "that group paid off."))

    # ---- Edge ---------------------------------------------------------------------------------
    spec_edge = panels.panel_edge_ranking(preds, window=ts.EDGE_WINDOW, mark_buffer=buffer)
    ch_edge = ts._chapter("07", "Bigger disagreement, bigger edge",
                          _para(
                              "One last check. The edge is how far above the price Sybilla rates a "
                              "draw. The hope is that a bigger disagreement makes a better bet. So "
                              "line every match up, biggest disagreement on the left down to the "
                              "smallest on the right, and see whether the games it was surest about "
                              "really did throw up more draws.")
                          + render.line_chart(spec_edge)
                          + _cap("How to read it. Left to right, biggest edge to smallest. Each line "
                                 "is a rolling average over 300 matches: the real draw rate, the "
                                 "model's estimate, and the bookmaker's price. Wherever the real-draw "
                                 "line rides above the bookmaker's, that stretch of bets made money."))

    # ---- Intro + assembly ---------------------------------------------------------------------
    intro = _para(
        "You do not have to know who will win a football match to make money on it. You only need to "
        "disagree with the bookmaker in the right direction, a little more often than you are wrong. "
        "That is the whole idea behind Sybilla. Every match comes with odds, and those odds hide a "
        "prediction: the bookmaker's own guess at how likely each result is. Sybilla ignores the win "
        "and the loss and looks only at the draw. It makes its own guess from what was known before "
        "kickoff, holds it up against the price, and when it thinks a draw is being sold too cheap, it "
        "buys. What follows is the honest scorecard of that idea over the last four seasons of the top "
        "five European leagues, tested only on matches the model never got to see while it was "
        "learning.")

    engine = render._CHART_JS.replace(
        "'Libre Franklin','Helvetica Neue',Helvetica,Arial,sans-serif",
        "'Archivo','Helvetica Neue',Arial,sans-serif")

    body = f"""<div class="wrap" lang="en">
<p class="eyebrow">model · tearsheet · plain language</p>
<h1>{run.model.name} - finding value in football draw bets</h1>
{ts._chapter("01", "What this is", intro)}
{ch_overview}
{ch_profit}
{ch_bets}
{ch_bank}
{ch_cal}
{ch_edge}
</div>
<div class="frame-tt" id="frame-tt" aria-hidden="true"></div>"""

    return (f"<title>{run.model.name}, plainly</title>\n<style>{ts.CSS}</style>\n{body}\n"
            f"<script>{engine}</script>\n")
