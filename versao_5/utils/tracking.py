"""utils/tracking.py — MLflow experiment tracking wrapper.

Provides a lightweight context manager that logs parameters, metrics,
and artefacts to MLflow when available, and falls back to a no-op logger
when MLflow is not installed or USE_MLFLOW=False.
"""
import os
import json
import datetime
from contextlib import contextmanager
from typing import Any

import config.pipeline as cfg


class _NoOpRun:
    """Silent fallback when MLflow is not used."""
    def log_param(self, k, v): pass
    def log_params(self, d): pass
    def log_metric(self, k, v, step=None): pass
    def log_metrics(self, d, step=None): pass
    def log_artifact(self, path): pass
    def set_tag(self, k, v): pass


@contextmanager
def experiment_run(run_name: str, tags: dict | None = None):
    """
    Context manager for a single experiment run.

    Usage:
        with experiment_run("RF_Agregado") as run:
            run.log_params({"n_estimators": 200})
            run.log_metric("RMSE", 2.31)
            run.log_artifact("figures/rmse_plot.png")
    """
    if not cfg.USE_MLFLOW:
        yield _NoOpRun()
        return

    try:
        import mlflow
        mlflow.set_tracking_uri(cfg.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(cfg.MLFLOW_EXP_NAME)
        with mlflow.start_run(run_name=run_name, tags=tags or {}) as active:
            yield active
    except ImportError:
        yield _NoOpRun()


def save_json_artefact(data: dict, path: str) -> None:
    """Save a dict as JSON and log it to the active MLflow run if any."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)


def log_metadata(step: str, params: dict, metrics: dict,
                 output_files: list[str]) -> None:
    """Write a lightweight JSON metadata record for a pipeline step."""
    from config.paths import METADATA_DIR
    record = {
        "step": step,
        "timestamp": datetime.datetime.now().isoformat(),
        "params": params,
        "metrics": metrics,
        "output_files": output_files,
    }
    path = os.path.join(METADATA_DIR, f"{step}_metadata.json")
    save_json_artefact(record, path)
    print(f"  [metadata] {path}")
