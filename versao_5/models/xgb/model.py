"""models/xgb/model.py — XGBoost with Optuna hyperparameter search."""
import numpy as np
from tuning.optuna_search import tune_xgboost
import config.model_params as mp


def train(X_tr, y_tr, X_val, y_val,
          run_name: str = "XGB",
          priority_idx=None, seed=None):
    # priority_idx: no-op here — see models/rf/model.py's train() docstring
    # for why (full feature matrix, no internal top-K reduction).
    seed = mp.XGB["seed"] if seed is None else seed
    best = tune_xgboost(X_tr, y_tr, X_val, y_val, run_name=run_name, seed=seed)

    try:
        import xgboost as xgb
        model = xgb.XGBRegressor(
            **best,
            n_estimators=mp.XGB["n_estimators"],
            early_stopping_rounds=mp.XGB["early_stopping_rounds"],
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    except ImportError:
        # CORRECTION (multi-seed requirement): this fallback used to hardcode
        # random_state=42 regardless of the requested seed, so every
        # xgboost-unavailable environment (like this one) would silently
        # report identical "XGBoost" results across all 3 seeds — and, unlike
        # SARIMAX/BayesianRidge, that would NOT be an honest reflection of
        # the underlying method (real XGBoost genuinely is seed-sensitive).
        # sklearn's GradientBoostingRegressor is only stochastic — and only
        # then responsive to random_state — when subsample<1.0 enables
        # per-tree row subsampling (its default subsample=1.0 is fully
        # deterministic regardless of seed). subsample=0.9 is a standard,
        # mild regularization setting (not tuned to produce a particular
        # result) that also makes the seed comparison meaningful here.
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(n_estimators=300, random_state=seed,
                                          subsample=0.9)
        model.fit(X_tr, y_tr)

    model._best_params         = best
    model._search_method       = "Optuna TPE"
    model._n_trials            = mp.XGB["n_trials"]
    model._selection_criterion = "RMSE on inner validation slice"
    model._seed                = seed
    return model
