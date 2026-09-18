"""Unit tests for the model/run store. Serialization only, so a Run is built by hand (no training)."""
import pandas as pd
import yaml

from models import store
from models._spec import Domain, Model
from models.run import Run


def a_model():
    return Model(
        name="Sybilla", slug="draw-major-goals_foragst+points+anchor-xgb",
        tagline="a toy", method="XGBoost", target="t_draw_flg", implied="mrkt_draw_impl",
        domain=Domain(gameweek_min=8, leagues="major"),
        spaces=["goals_foragst"], features=["hmt_season_goals_for_avg", "awt_season_goals_for_avg"],
        estimator="xgb", hyperparams={"max_depth": 50, "n_estimators": 150},
    )


def a_run(model):
    preds = pd.DataFrame({
        "match_id": [1, 2], "season": ["2223", "2526"], "league": ["E0", "D1"],
        "model_p": [0.3, 0.4], "implied": [0.28, 0.30], "fair": [0.26, 0.27], "y": [1, 0],
        "hmt_season_goals_for_avg": [1.5, 1.2], "awt_season_goals_for_avg": [1.1, 1.3],
    })
    return Run(model=model, predictions=preds, buffer=0.05, buffer_name="operational",
               holdout="2526", seasons=["2223", "2324", "2425", "2526"],
               bank_start=100.0, kelly_fraction=0.5,
               metrics={"n_bets": 2, "roi": 0.05, "profit": 0.1})


def test_save_params_writes_spaces_not_features(tmp_path):
    model = a_model()
    path = store.save_params(model, root=tmp_path)
    assert path == tmp_path / "sybilla" / "params.yaml"
    params = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert params["spaces"] == ["goals_foragst"]
    assert params["estimator"] == "xgb"
    assert params["domain"] == {"gameweek_min": 8, "leagues": "major"}
    assert "features" not in params            # features are derived from spaces on load


def test_run_id_is_span_plus_hash():
    rid = store.run_id(a_run(a_model()))
    span, _, h = rid.rpartition("-")
    assert span == "2223-2526"
    assert len(h) == 6


def test_save_run_writes_the_package(tmp_path):
    run = a_run(a_model())
    d = store.save_run(run, html="<title>x</title>", root=tmp_path)

    assert d.parent == tmp_path / "sybilla" / "runs"
    results = yaml.safe_load((d / "results.yaml").read_text(encoding="utf-8"))
    assert results["metrics"]["roi"] == 0.05
    assert results["model"]["features"] == ["hmt_season_goals_for_avg", "awt_season_goals_for_avg"]
    assert results["run"]["buffer"] == 0.05 and results["run"]["holdout"] == "2526"

    back = pd.read_parquet(d / "predictions.parquet")
    assert list(back["match_id"]) == [1, 2]
    assert (d / "report.html").read_text(encoding="utf-8") == "<title>x</title>"
