"""tuning/optuna_search.py — Optuna hyperparameter search (no MLflow dependency)."""
import os
import warnings
import numpy as np
import pandas as pd

import config.model_params as mp
import config.paths as paths

warnings.filterwarnings("ignore")


def _rmse(y_true, y_pred) -> float:
    m = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if m.sum() < 2:
        return np.nan
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2)))


def tune_random_forest(X_tr, y_tr, X_val, y_val,
                       run_name: str = "RF_tune", seed: int = None) -> dict:
    from sklearn.ensemble import RandomForestRegressor
    cfg = mp.RF
    seed = cfg["seed"] if seed is None else seed
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        return {"n_estimators": 200, "max_depth": 10,
                "min_samples_split": 5, "min_samples_leaf": 2,
                "max_features": "sqrt"}

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_categorical(
                "n_estimators", cfg["space"]["n_estimators"]),
            "max_depth": trial.suggest_categorical(
                "max_depth", cfg["space"]["max_depth"]),
            "min_samples_split": trial.suggest_categorical(
                "min_samples_split", cfg["space"]["min_samples_split"]),
            "min_samples_leaf": trial.suggest_categorical(
                "min_samples_leaf", cfg["space"]["min_samples_leaf"]),
            "max_features": trial.suggest_categorical(
                "max_features", cfg["space"]["max_features"]),
        }
        m = RandomForestRegressor(**params, random_state=seed, n_jobs=-1)
        m.fit(X_tr, y_tr)
        return _rmse(y_val, m.predict(X_val))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=cfg["n_trials"], show_progress_bar=False)
    return study.best_params


def tune_xgboost(X_tr, y_tr, X_val, y_val,
                 run_name: str = "XGB_tune", seed: int = None) -> dict:
    cfg = mp.XGB
    seed = cfg["seed"] if seed is None else seed
    try:
        import xgboost as xgb
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        return {"max_depth": 5, "learning_rate": 0.05,
                "subsample": 0.8, "colsample_bytree": 0.8,
                "reg_alpha": 0.5, "reg_lambda": 2.0}

    def objective(trial):
        params = {
            "max_depth":        trial.suggest_int(
                "max_depth", *cfg["space"]["max_depth"][:2]),
            "learning_rate":    trial.suggest_float(
                "learning_rate", *cfg["space"]["learning_rate"][:2], log=True),
            "subsample":        trial.suggest_float(
                "subsample", *cfg["space"]["subsample"]),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", *cfg["space"]["colsample_bytree"]),
            "reg_alpha":        trial.suggest_float(
                "reg_alpha", *cfg["space"]["reg_alpha"]),
            "reg_lambda":       trial.suggest_float(
                "reg_lambda", *cfg["space"]["reg_lambda"]),
            "n_estimators":     cfg["n_estimators"],
            "early_stopping_rounds": cfg["early_stopping_rounds"],
            "random_state":     seed,
            "n_jobs": -1,
        }
        m = xgb.XGBRegressor(**params)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        return _rmse(y_val, m.predict(X_val))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=cfg["n_trials"], show_progress_bar=False)
    return study.best_params


def tune_gbm(X_tr, y_tr, X_val, y_val,
             run_name: str = "GBM_tune", seed: int = None) -> dict:
    from sklearn.ensemble import HistGradientBoostingRegressor
    cfg = mp.GBM
    seed = cfg["seed"] if seed is None else seed
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        return {"max_depth": 6, "learning_rate": 0.05,
                "min_samples_leaf": 15, "l2_regularization": 1.0,
                "max_leaf_nodes": 31}

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int(
                "max_depth", *cfg["space"]["max_depth"][:2]),
            "learning_rate": trial.suggest_float(
                "learning_rate", *cfg["space"]["learning_rate"][:2], log=True),
            "min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", *cfg["space"]["min_samples_leaf"][:2]),
            "l2_regularization": trial.suggest_float(
                "l2_regularization", *cfg["space"]["l2_regularization"][:2]),
            "max_leaf_nodes": trial.suggest_int(
                "max_leaf_nodes", *cfg["space"]["max_leaf_nodes"][:2]),
        }
        m = HistGradientBoostingRegressor(
            **params, max_iter=cfg["max_iter"],
            early_stopping=cfg["early_stopping"],
            n_iter_no_change=cfg["patience"],
            random_state=seed,
        )
        m.fit(X_tr, y_tr)
        return _rmse(y_val, m.predict(X_val))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=cfg["n_trials"], show_progress_bar=False)
    return study.best_params


def export_hyperparameter_table(records: list[dict], out_dir: str = None,
                                 filename: str = "hyperparameter_table.csv") -> str:
    """
    out_dir : str, optional (NEW in versão_5 — user request: "os parâmetros
    e hiperparâmetros (em csv)... de forma independente" per horizon).
    Defaults to paths.TUNING_DIR for backward compatibility; callers that
    want a per-horizon table pass paths.horizon_dir(paths.TUNING_DIR, h).
    """
    out_dir = out_dir or paths.TUNING_DIR
    df = pd.DataFrame(records)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    df.to_csv(path, index=False)
    print(f"  Hyperparameter table → {path}")
    return path
