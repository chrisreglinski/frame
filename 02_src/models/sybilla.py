"""Sybilla: an XGBoost draw model on the major leagues.

The whole model is plain data: the feature spaces it is built on, the estimator name, and the
hyperparameters. Features are resolved from the spaces here, at definition time, so the run path only
ever sees a plain column list. It loads no data, trains nothing and knows nothing about the report.
"""
from models import spaces
from models._spec import Domain, Model

SPACES = ["goals_foragst", "points", "anchor"]

sybilla = Model(
    name="Sybilla",
    slug="draw-major-goals_foragst+points+anchor-xgb",
    tagline="an XGBoost model for predicting draws.",
    method="XGBoost",
    target="t_draw_flg",
    implied="mrkt_draw_impl",
    domain=Domain(gameweek_min=8, leagues="major"),
    spaces=SPACES,
    features=spaces.resolve(SPACES),
    estimator="xgb",
    hyperparams={
        "max_depth": 50, "n_estimators": 150, "learning_rate": 0.05,
        "subsample": 0.7, "colsample_bytree": 0.8, "min_child_weight": 20,
        "reg_lambda": 3.0, "reg_alpha": 0.5,
    },
)
