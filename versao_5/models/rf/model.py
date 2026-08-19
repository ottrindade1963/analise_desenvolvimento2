"""models/rf/model.py — Random Forest with Optuna hyperparameter search."""
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tuning.optuna_search import tune_random_forest
import config.model_params as mp


def train(X_tr, y_tr, X_val, y_val,
          run_name: str = "RF",
          priority_idx=None, seed=None) -> RandomForestRegressor:
    # priority_idx is accepted for interface parity with SARIMAX/Bayesian
    # (validation/walk_forward.py calls every trainer_fn uniformly) but is a
    # no-op here: Random Forest is fit on the full feature matrix, with no
    # internal top-K feature reduction, so it is never structurally blind to
    # the governance columns under ablation.
    seed = mp.RF["seed"] if seed is None else seed
    best = tune_random_forest(X_tr, y_tr, X_val, y_val, run_name=run_name, seed=seed)
    model = RandomForestRegressor(
        **best, random_state=seed, n_jobs=-1
    )
    model.fit(X_tr, y_tr)
    model._best_params    = best
    model._search_method  = "Optuna TPE"
    model._n_trials       = mp.RF["n_trials"]
    model._selection_criterion = "RMSE on inner validation slice"
    model._seed           = seed
    return model
