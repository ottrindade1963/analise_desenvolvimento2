"""config/model_params.py — Hyperparameter search spaces, seeds, and training settings.

Every value that was previously hard-coded inside a training function now lives
here so that the dissertation can cite exact search spaces, seeds, and
selection criteria in one place.
"""

SEED = 42


SEEDS = [7, 42, 90]

# ── Random Forest ─────────────────────────────────────────────────────────────
RF = {
    "search": "optuna_tpe",
    "n_trials": 50,
    "seed": SEED,
    "criterion": "neg_mean_squared_error",
    "space": {
        "n_estimators":      [100, 200, 300, 500],
        "max_depth":         [3, 5, 10, 15, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "max_features":      ["sqrt", "log2", 0.5],
    },
}

# ── XGBoost ───────────────────────────────────────────────────────────────────
XGB = {
    "search": "optuna_tpe",
    "n_trials": 50,
    "seed": SEED,
    "criterion": "neg_mean_squared_error",
    "n_estimators": 500,
    "early_stopping_rounds": 50,
    "space": {
        "max_depth":        (3, 8),
        "learning_rate":    (0.005, 0.2, "log"),
        "subsample":        (0.5, 1.0),
        "colsample_bytree": (0.5, 1.0),
        "reg_alpha":        (0.0, 2.0),
        "reg_lambda":       (0.5, 5.0),
    },
}

# ── HistGradientBoosting ──────────────────────────────────────────────────────
GBM = {
    "search": "optuna_tpe",
    "n_trials": 50,
    "seed": SEED,
    "max_iter": 500,
    "early_stopping": True,
    "patience": 30,
    "space": {
        "max_depth":          (3, 9),
        "learning_rate":      (0.005, 0.2, "log"),
        "min_samples_leaf":   (5, 50),
        "l2_regularization":  (0.0, 5.0),
        "max_leaf_nodes":     (15, 63),
    },
}

# ── SARIMAX ───────────────────────────────────────────────────────────────────
SARIMAX = {
    "order": (1, 1, 1),          # default; AIC search enabled when auto=True
    "auto_order": True,          # try p,d,q ∈ {0,1,2} and pick lowest AIC
    "max_exog": 8,               # max exogenous regressors (stability)
    "maxiter": 500,
    "method": "lbfgs",
    "export_coefficients": True, # export coef, SE, CI95, p-value table
}

# ── LSTM ──────────────────────────────────────────────────────────────────────
LSTM = {
    "lookback": 3,               # sequence length (years) — must be >1
    "units": [64, 32],           # units per LSTM layer
    "dropout": 0.3,
    "l2": 0.01,
    "epochs": 200,
    "batch_size": 32,
    "patience": 15,              # early stopping
    "seed": SEED,
    "search": "fixed",           # architecture fixed; only regularisation tuned
}

# ── Bayesian (PyMC) ───────────────────────────────────────────────────────────
BAYESIAN = {
    "draws": 2000,
    "tune": 1000,
    "chains": 2,
    "cores": 1,
    "seed": SEED,
    "timeout_s": 120,
    "export_posterior": True,    # R-hat, ESS, HDI, trace plots via ArviZ
    "posterior_predictive": True,
    "max_features": 10,          # reduce before MCMC for stability
}

# ── ACI (Adaptive Conformal Inference) ────────────────────────────────────────
ACI = {
    "alpha": 0.10,               # miscoverage target → 90% coverage
    "default_gamma": 0.05,
    "default_window": 3,
    "gamma_grid":  [0.01, 0.05, 0.10, 0.20, 0.50, 1.00],
    "window_grid": [3, 5, 7, 10, 15],
}
