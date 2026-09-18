"""Unit test for the params.yaml loader: fields copied, spaces resolved to features."""
import yaml

from models.loader import load_model


def test_load_resolves_spaces_to_features(tmp_path):
    path = tmp_path / "params.yaml"
    path.write_text(yaml.safe_dump({
        "name": "Sybilla", "slug": "draw-major-goals_foragst+points+anchor-xgb",
        "tagline": "a toy", "method": "XGBoost",
        "target": "t_draw_flg", "implied": "mrkt_draw_impl",
        "domain": {"gameweek_min": 8, "leagues": "major"},
        "spaces": ["goals_foragst", "points", "anchor"],
        "estimator": "xgb", "hyperparams": {"max_depth": 50},
    }), encoding="utf-8")

    m = load_model(path)
    assert m.name == "Sybilla"
    assert m.estimator == "xgb"
    assert m.domain.gameweek_min == 8 and m.domain.leagues == "major"
    assert m.spaces == ["goals_foragst", "points", "anchor"]
    assert len(m.features) == 9                       # resolved from the three spaces
    assert "mrkt_draw_impl" in m.features             # the anchor
    assert "hmt_season_goals_for_avg" in m.features   # a goals column
