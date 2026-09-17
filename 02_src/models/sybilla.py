"""Sybilla: an XGBoost draw model on the major leagues.

This file is the whole model: its parameters (data, snapshotted into every run) and one thin hook
that builds the estimator. It loads no data, trains nothing and knows nothing about the report.
Change a hyperparameter here and the next run carries the new value in its manifest.
"""
from xgboost import XGBClassifier

from models._spec import Domain, Model


def make_model(hyperparams):
    """Build the estimator from the modeling hyperparameters, adding fixed operational settings."""
    return XGBClassifier(**hyperparams, eval_metric="logloss",
                         n_jobs=4, verbosity=0, enable_categorical=True)


sybilla = Model(
    name="Sybilla",
    tagline="an XGBoost model for predicting draws.",
    method="XGBoost",
    target="t_draw_flg",
    implied="mrkt_draw_impl",
    domain=Domain(gameweek_min=8, leagues="major"),
    features=[
        "hmt_season_goals_for_avg", "hmt_season_goals_agst_avg",
        "awt_season_goals_for_avg", "awt_season_goals_agst_avg",
        "hmt_season_points_avg", "awt_season_points_avg",
        "hmt_season_mp_impl_points_avg", "awt_season_mp_impl_points_avg",
        "mrkt_draw_impl",
    ],
    hyperparams={
        "max_depth": 50, "n_estimators": 5000, "learning_rate": 0.01,
        "subsample": 0.7, "colsample_bytree": 0.8, "min_child_weight": 10,
        "reg_lambda": 3.0, "reg_alpha": 0.5,
    },
    make_model=make_model,
)
