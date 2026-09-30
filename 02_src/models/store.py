"""Persist models and runs under 04_models, one folder per model.

    04_models/<name|slug>/
      params.yaml                    the authored definition (spaces, estimator, hyperparams, ...)
      runs/<run-id>/
        results.yaml                 frozen snapshot of the definition, the run, and the metrics
        predictions.parquet          the out-of-fold register plus the model's features (thin ABT)
        report.html                  the rendered tearsheet for this run

The folder is named by the human name if there is one, else the slug. The slug is a display label
only; every stored value is pulled from its own field, never parsed from the slug. A run-id is the
season span plus a short hash of the hyperparameters, a technical id distinct from the slug.
"""
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from models._spec import model_from_snapshot, snapshot

ROOT = Path(__file__).parents[2] / "04_models"


def _label(model):
    return (model.name or model.slug).lower()


def model_dir(model, root=ROOT):
    """The folder for a model: 04_models/<name|slug>."""
    return Path(root) / _label(model)


def run_id(run):
    """Season span plus a short hyperparameter hash, e.g. 2223-2526-a1b2c3."""
    seasons = [str(s) for s in run.seasons]
    digest = hashlib.sha1(json.dumps(run.model.hyperparams, sort_keys=True).encode()).hexdigest()
    return f"{seasons[0]}-{seasons[-1]}-{digest[:6]}"


def save_params(model, root=ROOT):
    """Write the model's authored definition to params.yaml. Stores the spaces and extra variables,
    not the resolved features or the slug, since both are derived on load."""
    params = {
        "name": model.name, "tagline": model.tagline, "method": model.method,
        "target": model.target, "implied": model.implied,
        "domain": {"gameweek_min": model.domain.gameweek_min, "leagues": model.domain.leagues},
        "spaces": list(model.spaces), "extra": list(model.extra), "estimator": model.estimator,
        "hyperparams": dict(model.hyperparams),
    }
    d = model_dir(model, root)
    d.mkdir(parents=True, exist_ok=True)
    path = d / "params.yaml"
    path.write_text(yaml.safe_dump(params, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def save_run(run, html=None, root=ROOT):
    """Write a run's folder: results.yaml, predictions.parquet, and report.html if given. Returns the
    run directory. results.yaml freezes the full model snapshot (features included) so the run is
    self-describing."""
    results = {
        "model": snapshot(run.model),
        "run": {
            "seasons": [str(s) for s in run.seasons],
            "holdout": str(run.holdout),
            "buffer": float(run.buffer),
            "buffer_name": run.buffer_name,
            "bank_start": float(run.bank_start),
            "kelly_fraction": float(run.kelly_fraction),
        },
        "metrics": dict(run.metrics),
    }
    d = model_dir(run.model, root) / "runs" / run_id(run)
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.yaml").write_text(
        yaml.safe_dump(results, sort_keys=False, allow_unicode=True), encoding="utf-8")
    run.predictions.to_parquet(d / "predictions.parquet")
    if html is not None:
        (d / "report.html").write_text(html, encoding="utf-8")
    return d


def load_run(run_dir):
    """Reconstruct a Run from a saved run folder (results.yaml plus predictions.parquet), so the
    report renders a persisted run without retraining. The model is rebuilt from the frozen snapshot,
    the predictions from the parquet."""
    from models.run import Run   # local import to avoid a module import cycle

    run_dir = Path(run_dir)
    results = yaml.safe_load((run_dir / "results.yaml").read_text(encoding="utf-8"))
    r = results["run"]
    return Run(
        model=model_from_snapshot(results["model"]),
        predictions=pd.read_parquet(run_dir / "predictions.parquet"),
        buffer=r["buffer"], buffer_name=r["buffer_name"], holdout=r["holdout"],
        seasons=r["seasons"], bank_start=r["bank_start"], kelly_fraction=r["kelly_fraction"],
        metrics=results["metrics"],
    )
