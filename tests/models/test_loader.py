"""Unit tests for the params.yaml loader: fields copied, spaces and extras resolved, slug computed."""
import yaml

from models._spec import make_slug
from models.loader import load_model


def test_load_resolves_spaces_and_extra_to_features(tmp_path):
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump({
        "name": "Sybilla", "tagline": "a toy", "method": "XGBoost",
        "target": "t_draw_flg", "implied": "mrkt_draw_impl",
        "domain": {"gameweek_min": 8, "leagues": "major"},
        "spaces": ["goals_foragst", "points"], "extra": ["mrkt_draw_impl"],
        "estimator": "xgb", "hyperparams": {"max_depth": 50},
    }), encoding="utf-8")

    m = load_model(path)
    assert m.name == "Sybilla"
    assert m.estimator == "xgb"
    assert m.domain.gameweek_min == 8 and m.domain.leagues == "major"
    assert m.spaces == ["goals_foragst", "points"]
    assert m.extra == ["mrkt_draw_impl"]
    assert len(m.features) == 9                       # eight space columns plus the extra
    assert m.features[-1] == "mrkt_draw_impl"         # extras come after the spaces
    assert m.slug.startswith("draw-major-goals_foragst+points-xgb-")


def test_slug_without_extra_has_four_parts():
    assert make_slug("t_draw_flg", "major", ["goals_foragst", "points"], [], "xgb") == \
        "draw-major-goals_foragst+points-xgb"


def test_slug_hash_ignores_extra_order():
    a = make_slug("t_draw_flg", "major", ["xg"], ["b", "a"], "xgb")
    b = make_slug("t_draw_flg", "major", ["xg"], ["a", "b"], "xgb")
    assert a == b
    assert len(a.rpartition("-")[2]) == 6
