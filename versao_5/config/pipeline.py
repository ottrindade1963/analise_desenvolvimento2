"""config/pipeline.py — Pipeline-level settings: splits, validation, tracking."""

# ── Temporal splits ───────────────────────────────────────────────────────────
TRAIN_END_YEAR = 2017    # Training window: 1996–2017
VAL_END_YEAR   = 2020    # Validation window: 2018–2020
# Test window: 2021–2023 (held out)

FINAL_HOLDOUT_RATIO = 0.15   # Fraction reserved for final holdout evaluation

# ── Walk-forward cross-validation ─────────────────────────────────────────────
WF_N_FOLDS      = 5
WF_MIN_TRAIN    = 0.50    # Minimum training fraction in first fold

# ── Synthetic data ────────────────────────────────────────────────────────────
SYNTHETIC_YEARS = 500
SYNTHETIC_SEED  = 42

# ── Experiment tracking (MLflow) ──────────────────────────────────────────────
USE_MLFLOW      = False        # Disabled: MLflow filesystem backend discontinued
MLFLOW_EXP_NAME = "africa_mo_industrial_v4"
MLFLOW_TRACKING_URI = "mlruns"

# ── Project metadata ──────────────────────────────────────────────────────────
PROJECT_NAME    = "Análise Industrial — África e Médio Oriente"
PROJECT_VERSION = "4.0"
PROJECT_AUTHOR  = "Dissertação de Mestrado"
