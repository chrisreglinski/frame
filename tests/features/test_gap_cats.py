"""hours_since_*_cat2q / cat3q: bucketing of the two gap columns.

The boundaries are cut on the pooled hmt + awt values rather than per side, which is what
makes the categories comparable across the two sides of a match — the tests below pin that
down with a fixture where the two sides have deliberately lopsided distributions.

The caps on the underlying columns (200h and 500h) sit above p67 in the real data, so the
capped tail is all `high`. Nothing here depends on that, but it is why the top bucket is
the one that swallows breaks and postponements.
"""
import json

import pandas as pd
import pytest

from features import match_team_fatigue
from features.match_team_fatigue import _attach_gap_cats


@pytest.fixture
def features_dir(tmp_path, monkeypatch):
    """_attach_gap_cats writes thresholds.json next to the built tables; keep it in tmp."""
    monkeypatch.setattr(match_team_fatigue, "_features_dir", lambda group: tmp_path)
    return tmp_path


def _frame(hmt_last, awt_last, hmt_prev=None, awt_prev=None):
    n = len(hmt_last)
    return pd.DataFrame({
        "match_id": [f"m{i}" for i in range(n)],
        "hmt_hours_since_last_match": hmt_last,
        "awt_hours_since_last_match": awt_last,
        "hmt_hours_since_2nd_last_match": hmt_prev if hmt_prev else [300.0] * n,
        "awt_hours_since_2nd_last_match": awt_prev if awt_prev else [300.0] * n,
    })


def test_tertiles_split_the_pooled_values_evenly(features_dir):
    # eight values pooled over the two sides: 1..8, so p33 = 3.33 and p67 = 5.67
    out = _attach_gap_cats(_frame([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]), "major")
    assert list(out["hmt_hours_since_last_match_cat3q"]) == [
        "low", "low", "low", "medium"]
    assert list(out["awt_hours_since_last_match_cat3q"]) == [
        "medium", "high", "high", "high"]


def test_the_median_splits_cat2q(features_dir):
    # pooled 1..6, median 3.5
    out = _attach_gap_cats(_frame([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), "major")
    assert list(out["hmt_hours_since_last_match_cat2q"]) == ["low", "low", "low"]
    assert list(out["awt_hours_since_last_match_cat2q"]) == ["high", "high", "high"]


def test_boundaries_belong_to_the_lower_bucket(features_dir):
    """A value sitting exactly on p33 or p67 is low / medium, never the bucket above."""
    out = _attach_gap_cats(_frame([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), "major")
    p33 = json.loads((features_dir / "thresholds.json").read_text())[
        "hours_since_last_match_p33"]
    assert p33 == pytest.approx(2.667)

    on_boundary = _attach_gap_cats(_frame([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
                                   .assign(hmt_hours_since_last_match=[p33, p33, p33]),
                                   "major")
    assert set(on_boundary["hmt_hours_since_last_match_cat3q"]) == {"low"}


def test_both_sides_share_one_set_of_boundaries(features_dir):
    """The away side here is uniformly slower-rested than the home side. Cutting per side
    would put an away value of 4 in the bottom bucket; pooling keeps it in the middle."""
    out = _attach_gap_cats(_frame([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), "major")
    assert out.loc[0, "awt_hours_since_last_match_cat3q"] == "medium"
    assert out.loc[0, "hmt_hours_since_last_match_cat3q"] == "low"


def test_missing_gaps_stay_empty(features_dir):
    out = _attach_gap_cats(_frame([1.0, None, 3.0], [4.0, 5.0, 6.0]), "major")
    assert pd.isna(out.loc[1, "hmt_hours_since_last_match_cat3q"])
    assert pd.isna(out.loc[1, "hmt_hours_since_last_match_cat2q"])
    # and the missing value does not shift the boundaries for anyone else
    assert out.loc[0, "hmt_hours_since_last_match_cat3q"] == "low"


def test_the_two_gap_columns_get_their_own_boundaries(features_dir):
    """They measure different spans, so one shared threshold would put every
    2nd-last value in the top bucket."""
    out = _attach_gap_cats(_frame([1.0, 2.0, 3.0], [4.0, 5.0, 6.0],
                                  hmt_prev=[10.0, 20.0, 30.0],
                                  awt_prev=[40.0, 50.0, 60.0]), "major")
    # pooled 10..60, so p33 = 26.7 and p67 = 43.3 — nothing like the 1..6 pool above
    assert list(out["hmt_hours_since_2nd_last_match_cat3q"]) == ["low", "low", "medium"]
    assert list(out["awt_hours_since_2nd_last_match_cat3q"]) == ["medium", "high", "high"]

    th = json.loads((features_dir / "thresholds.json").read_text())
    assert th["hours_since_last_match_p50"] != th["hours_since_2nd_last_match_p50"]


def test_thresholds_file_keeps_the_keys_earlier_steps_wrote(features_dir):
    (features_dir / "thresholds.json").write_text(json.dumps({"elo_mean": 1700.0}))
    _attach_gap_cats(_frame([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), "major")

    th = json.loads((features_dir / "thresholds.json").read_text())
    assert th["elo_mean"] == 1700.0
    assert set(th) >= {f"hours_since_last_match_p{p}" for p in (33, 50, 67)}


def test_an_empty_pool_leaves_the_categories_empty(features_dir):
    out = _attach_gap_cats(_frame([None, None], [None, None]), "major")
    for side in ("hmt", "awt"):
        assert out[f"{side}_hours_since_last_match_cat3q"].isna().all()
        assert out[f"{side}_hours_since_last_match_cat2q"].isna().all()
