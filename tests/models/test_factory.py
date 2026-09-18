"""Unit tests for the estimator factory."""
import pytest

from models import factory


def test_names_has_xgb():
    assert "xgb" in factory.names()


def test_unknown_estimator_raises():
    with pytest.raises(KeyError):
        factory.build("nope", {})


def test_xgb_carries_hyperparams_and_operational_defaults():
    est = factory.build("xgb", {"max_depth": 3, "n_estimators": 10})
    p = est.get_params()
    assert p["max_depth"] == 3 and p["n_estimators"] == 10   # passed through
    assert p["eval_metric"] == "logloss" and p["random_state"] == 0   # operational, added here


def test_build_does_not_mutate_input():
    hp = {"max_depth": 3}
    factory.build("xgb", hp)
    assert hp == {"max_depth": 3}
