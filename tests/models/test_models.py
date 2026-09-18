"""Unit tests for the model layer: the contract helpers, the registry, and run plumbing.

run is exercised on a tiny synthetic ABT with a light LogisticRegression so the test stays fast and
does not touch the real data or xgboost. It checks the contract (columns, frozen metrics, buffer),
not model quality.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from models import factory, registry
from models._spec import Domain, Model, apply_domain, snapshot
from models.run import Run, run


@factory.register("test_lr")
def _test_lr(hp):
    """A light estimator registered only for these tests, so run stays off xgboost."""
    return LogisticRegression(**hp)


def tiny_model():
    return Model(
        name="Tiny", tagline="a toy model", method="LR",
        target="t_draw_flg", implied="mrkt_draw_impl",
        domain=Domain(gameweek_min=0, leagues="major"),
        features=["f1", "f2"], estimator="test_lr", hyperparams={"max_iter": 200},
    )


def tiny_abt(n=60):
    """Two seasons of made-up matches. The target splits each season at its f1 median, so both
    classes are always present per fold and f1 carries real signal."""
    rng = np.random.default_rng(0)
    seasons = np.repeat(["2223", "2324"], n // 2)
    f1, f2 = rng.normal(size=n), rng.normal(size=n)
    y = np.zeros(n, dtype=int)
    for s in np.unique(seasons):
        mask = seasons == s
        y[mask] = (f1[mask] > np.median(f1[mask])).astype(int)
    return pd.DataFrame({
        "hmt_game_number": 10, "awt_game_number": 10,
        "f1": f1, "f2": f2,
        "mrkt_draw_impl": np.clip(0.28 + 0.03 * rng.normal(size=n), 0.05, 0.6),
        "mrkt_home_impl": np.clip(0.40 + 0.05 * rng.normal(size=n), 0.1, 0.9),
        "mrkt_away_impl": np.clip(0.40 + 0.05 * rng.normal(size=n), 0.1, 0.9),
        "t_draw_flg": y,
        "league": rng.choice(["E0", "D1"], size=n),
        "season": seasons,
        "match_id": np.arange(n),
        "date": pd.date_range("2022-08-01", periods=n, freq="3D"),
        "hmt_name": "home", "awt_name": "away",
    })


# ---- contract helpers ------------------------------------------------------------------------

def test_apply_domain_filters_by_gameweek():
    abt = pd.DataFrame({"hmt_game_number": [5, 9, 12], "awt_game_number": [12, 3, 12]})
    kept = apply_domain(abt, Domain(gameweek_min=8, leagues="major"))
    assert list(kept.index) == [2]          # row 0 fails on home, row 1 on away, row 2 passes both


def test_snapshot_is_plain_data():
    snap = snapshot(tiny_model())
    assert "make_model" not in snap                 # no code fields anymore
    assert snap["domain"] == {"gameweek_min": 0, "leagues": "major"}
    assert snap["hyperparams"] == {"max_iter": 200}
    assert snap["estimator"] == "test_lr"
    assert snap["name"] == "Tiny"


# ---- registry --------------------------------------------------------------------------------

def _seed_model(root):
    """A minimal params.yaml under `root`, since real models live in gitignored 04_models."""
    import yaml
    d = root / "toy"
    d.mkdir(parents=True, exist_ok=True)
    (d / "params.yaml").write_text(yaml.safe_dump({
        "name": "Toy", "slug": "toy", "target": "t_draw_flg", "implied": "mrkt_draw_impl",
        "domain": {"gameweek_min": 8, "leagues": "major"},
        "spaces": ["anchor"], "estimator": "xgb", "hyperparams": {"max_depth": 3},
    }), encoding="utf-8")


def test_registry_get_and_names(tmp_path):
    _seed_model(tmp_path)
    assert registry.names(root=tmp_path) == ["Toy"]
    m = registry.get("Toy", root=tmp_path)
    assert m.name == "Toy"
    assert m.features == ["mrkt_draw_impl"]     # spaces resolved by the loader


def test_registry_unknown_name_raises(tmp_path):
    _seed_model(tmp_path)
    with pytest.raises(KeyError):
        registry.get("nobody", root=tmp_path)


# ---- run plumbing ----------------------------------------------------------------------------

def test_run_contract():
    """A permissive buffer makes every scored match a bet, so metrics are always well defined."""
    r = run(tiny_model(), abt=tiny_abt(), buffer=-1.0)
    assert isinstance(r, Run)

    for col in ("model_p", "implied", "y", "fair", "season", "date", "match_id"):
        assert col in r.predictions.columns
    assert len(r.predictions) == 60            # gameweek_min 0 keeps all, no NaN dropped

    assert r.buffer == -1.0
    assert r.buffer_name == "operational"
    assert r.metrics["n_bets"] == len(r.predictions)   # buffer -1 backs every match
    for key in ("roi", "profit", "p_value", "bank_final", "bank_maxdd", "bank_cagr"):
        assert key in r.metrics


def test_run_devig_is_a_probability():
    r = run(tiny_model(), abt=tiny_abt(), buffer=-1.0)
    fair = r.predictions["fair"]
    assert (fair > 0).all() and (fair < 1).all()


def test_run_explicit_buffer_selects_bets():
    r = run(tiny_model(), abt=tiny_abt(), buffer=0.02)
    bets = r.predictions[r.predictions["model_p"] > r.predictions["implied"] + 0.02]
    assert r.metrics["n_bets"] == len(bets)
