"""Unit tests for the params.yaml loader: fields copied, spaces and extras resolved, slug computed."""
import yaml

from models._spec import make_slug, problem_slug, solution_slug
from models.loader import load_model


def test_load_resolves_spaces_and_extra_to_features(tmp_path):
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump({
        "name": "Sybilla", "tagline": "a toy", "method": "XGBoost",
        "target": "t_draw_flg", "implied": "mrkt_draw_impl",
        "domain": {"leagues": "major", "name": "after_gw8"},
        "spaces": ["goals_foragst", "points", "impl_points"], "extra": ["mrkt_draw_impl"],
        "estimator": "xgb", "hyperparams": {"max_depth": 50},
    }), encoding="utf-8")

    m = load_model(path)
    assert m.name == "Sybilla"
    assert m.estimator == "xgb"
    assert m.domain.name == "after_gw8" and m.domain.leagues == "major"
    assert m.spaces == ["goals_foragst", "points", "impl_points"]
    assert m.extra == ["mrkt_draw_impl"]
    assert len(m.features) == 9                       # eight space columns plus the extra
    assert m.features[-1] == "mrkt_draw_impl"         # extras come after the spaces
    assert m.slug.startswith("draw-major-after_gw8-xgb-goals_foragst+points+impl_points-")


def test_slug_is_problem_then_solution():
    slug = make_slug("t_draw_flg", "major", "core", ["goals_foragst", "points"], [], "xgb")
    assert slug == "draw-major-core-xgb-goals_foragst+points"
    assert problem_slug("t_draw_flg", "major", "core") == "draw-major-core"
    assert solution_slug("xgb", ["goals_foragst", "points"], []) == "xgb-goals_foragst+points"


def test_slug_hash_ignores_extra_order():
    a = make_slug("t_draw_flg", "major", "core", ["xg"], ["b", "a"], "xgb")
    b = make_slug("t_draw_flg", "major", "core", ["xg"], ["a", "b"], "xgb")
    assert a == b
    assert len(a.rpartition("-")[2]) == 6
