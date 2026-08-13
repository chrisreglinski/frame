"""hours_since_* gap columns: the p95 cap and the cat2q / cat3q buckets.

The boundaries are cut on the pooled hmt + awt values rather than per side, which is what
makes the categories comparable across the two sides of a match — the tests below pin that
down with a fixture where the two sides have deliberately lopsided distributions.

The same pass also caps the underlying column, at the 95th percentile of its own pooled
distribution. That cut lands above p67 at every depth in the real data, so the capped tail
is all `high` and the boundaries below it are untouched — which is why the bucket
expectations here are unaffected by the cap.
"""
import json

import pandas as pd
import pytest

from features import match_team_fatigue
from features.match_team_fatigue import _cap_and_categorize_gaps, _gap_cols, _schema

_GAPS = _gap_cols(_schema())


@pytest.fixture
def features_dir(tmp_path, monkeypatch):
    """The pass writes thresholds.json next to the built tables; keep it in tmp."""
    monkeypatch.setattr(match_team_fatigue, "_features_dir", lambda group: tmp_path)
    return tmp_path


# One filler per gap column, so a test that varies one of them still hands the builder
# every column it categorizes. Driven off _GAP_COLS: adding a fourth gap to the contract
# without wiring it here fails loudly instead of silently skipping it.
_FILLER = {col: 100.0 * (i + 1) for i, col in enumerate(_GAPS)}


def _frame(**by_stem):
    """by_stem maps a gap column to its (hmt values, awt values); columns left out are
    filled with a constant."""
    n = len(next(iter(by_stem.values()))[0])
    data = {"match_id": [f"m{i}" for i in range(n)]}
    for col, filler in _FILLER.items():
        hmt, awt = by_stem.get(col, ([filler] * n, [filler] * n))
        data[f"hmt_{col}"], data[f"awt_{col}"] = hmt, awt
    return pd.DataFrame(data)


def _last(hmt, awt):
    return _frame(hours_since_last_match=(hmt, awt))


def _cap(df, group="major"):
    return _cap_and_categorize_gaps(df, group, _GAPS)


def test_the_cap_is_the_95th_percentile_of_the_pooled_values(features_dir):
    """Twenty values, 1..20 pooled over the two sides: p95 = 19.05, so only the top one is
    pulled down and everything below it passes through untouched."""
    hmt = [float(i) for i in range(1, 11)]
    awt = [float(i) for i in range(11, 21)]
    out = _cap(_last(hmt, awt))

    cap = json.loads((features_dir / "thresholds.json").read_text())[
        "hours_since_last_match_cap"]
    assert cap == pytest.approx(19.05)
    assert out["awt_hours_since_last_match"].tolist() == awt[:-1] + [19.05]
    assert out["hmt_hours_since_last_match"].tolist() == hmt


def test_tertiles_split_the_pooled_values_evenly(features_dir):
    # eight values pooled over the two sides: 1..8, so p33 = 3.33 and p67 = 5.67
    out = _cap(_last([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]), "major")
    assert list(out["hmt_hours_since_last_match_cat3q"]) == [
        "low", "low", "low", "medium"]
    assert list(out["awt_hours_since_last_match_cat3q"]) == [
        "medium", "high", "high", "high"]


def test_the_median_splits_cat2q(features_dir):
    # pooled 1..6, median 3.5
    out = _cap(_last([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), "major")
    assert list(out["hmt_hours_since_last_match_cat2q"]) == ["low", "low", "low"]
    assert list(out["awt_hours_since_last_match_cat2q"]) == ["high", "high", "high"]


def test_boundaries_belong_to_the_lower_bucket(features_dir):
    """A value sitting exactly on p33 or p67 is low / medium, never the bucket above."""
    out = _cap(_last([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), "major")
    p33 = json.loads((features_dir / "thresholds.json").read_text())[
        "hours_since_last_match_p33"]
    assert p33 == pytest.approx(2.667)

    on_boundary = _cap(_last([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
                                   .assign(hmt_hours_since_last_match=[p33, p33, p33]),
                                   "major")
    assert set(on_boundary["hmt_hours_since_last_match_cat3q"]) == {"low"}


def test_both_sides_share_one_set_of_boundaries(features_dir):
    """The away side here is uniformly slower-rested than the home side. Cutting per side
    would put an away value of 4 in the bottom bucket; pooling keeps it in the middle."""
    out = _cap(_last([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), "major")
    assert out.loc[0, "awt_hours_since_last_match_cat3q"] == "medium"
    assert out.loc[0, "hmt_hours_since_last_match_cat3q"] == "low"


def test_missing_gaps_stay_empty(features_dir):
    out = _cap(_last([1.0, None, 3.0], [4.0, 5.0, 6.0]), "major")
    assert pd.isna(out.loc[1, "hmt_hours_since_last_match_cat3q"])
    assert pd.isna(out.loc[1, "hmt_hours_since_last_match_cat2q"])
    # and the missing value does not shift the boundaries for anyone else
    assert out.loc[0, "hmt_hours_since_last_match_cat3q"] == "low"


def test_the_two_gap_columns_get_their_own_boundaries(features_dir):
    """They measure different spans, so one shared threshold would put every
    2nd-last value in the top bucket."""
    out = _cap(_frame(
        hours_since_last_match=([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
        hours_since_2nd_last_match=([10.0, 20.0, 30.0], [40.0, 50.0, 60.0])), "major")
    # pooled 10..60, so p33 = 26.7 and p67 = 43.3 — nothing like the 1..6 pool above
    assert list(out["hmt_hours_since_2nd_last_match_cat3q"]) == ["low", "low", "medium"]
    assert list(out["awt_hours_since_2nd_last_match_cat3q"]) == ["medium", "high", "high"]

    th = json.loads((features_dir / "thresholds.json").read_text())
    assert th["hours_since_last_match_p50"] != th["hours_since_2nd_last_match_p50"]


def test_thresholds_file_keeps_the_keys_earlier_steps_wrote(features_dir):
    (features_dir / "thresholds.json").write_text(json.dumps({"elo_mean": 1700.0}))
    _cap(_last([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), "major")

    th = json.loads((features_dir / "thresholds.json").read_text())
    assert th["elo_mean"] == 1700.0
    assert set(th) >= {f"hours_since_last_match_p{p}" for p in (33, 50, 67)}


def test_every_gap_column_gets_both_categories(features_dir):
    """Guards the wiring: a depth added to the contract but skipped downstream shows up
    here rather than as a column silently missing from the built table."""
    out = _cap(_last([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), "major")
    expected = {f"{side}_{col}_{suf}"
                for side in ("hmt", "awt") for col in _GAPS for suf in ("cat2q", "cat3q")}
    assert expected <= set(out.columns)


def test_an_empty_pool_leaves_the_categories_empty(features_dir):
    out = _cap(_last([None, None], [None, None]), "major")
    for side in ("hmt", "awt"):
        assert out[f"{side}_hours_since_last_match_cat3q"].isna().all()
        assert out[f"{side}_hours_since_last_match_cat2q"].isna().all()
