# Industrial Analysis Pipeline — Africa & Middle East — versão_5

Dissertation ML pipeline for predicting industrial value-added (% GDP) across
37 countries in Africa and the Middle East, 1996–2023.

**versão_5** is derived from the original corrected project (**versão_4**,
GitHub repo `analise_desenvolvimento`) and adds genuine h-step-ahead
forecasting up to 5 years ahead (h = 1, 2, 3, 4, 5 — see
`validation/walk_forward.py`, `preprocessing/temporal.py`, and
`relatorio2` for the full methodology). Every model, hyperparameter table,
results table and figure is generated and saved independently per horizon
(see `config/paths.py::horizon_dir`). It is intended to be pushed to its
own GitHub repository, `analise_desenvolvimento2` — see `config/paths.py`
for the Colab clone-path convention this relies on.

## Directory structure

```
versao_5/
│
├── config/              # All configuration — one file per concern
│   ├── paths.py         # Filesystem paths
│   ├── variables.py     # WDI/WGI indicators, countries, target variable
│   ├── features.py      # Feature engineering settings + ablation specs
│   ├── model_params.py  # Hyperparameter search spaces, seeds, LSTM lookback
│   └── pipeline.py      # Splits, walk-forward CV, MLflow, project metadata
│
├── data/                # Data files (created at runtime)
│   ├── raw/             # WDI + WGI from World Bank API
│   ├── clean/           # After per-country MICE imputation
│   ├── aggregated/      # INNER JOIN of WDI + WGI
│   ├── synthetic/       # 500-year extrapolation for robustness
│   └── features/        # Feature-engineered datasets (one per ablation spec)
│
├── preprocessing/
│   ├── imputer.py       # PanelMICEImputer (sklearn TransformerMixin)
│   └── scaler.py        # PanelScaler      (sklearn TransformerMixin)
│
├── features/
│   └── engineer.py      # FoldFeatureEngineer (sklearn TransformerMixin)
│                        # Lags + rolling means + PCA — all fit on train only
│
├── validation/
│   └── walk_forward.py  # WalkForwardCV — expanding-window, fold-level preprocessing
│
├── models/
│   ├── rf/model.py      # Random Forest + Optuna TPE
│   ├── xgb/model.py     # XGBoost + Optuna TPE
│   ├── sarimax/model.py # SARIMAX + AIC order selection + coefficient table
│   ├── lstm/model.py    # LSTM with lookback sequences (not single time-step)
│   └── bayesian/model.py # PyMC hierarchical + PPCs + R-hat + ESS
│
├── tuning/
│   └── optuna_search.py # Unified Optuna TPE search + hyperparameter table export
│
├── explainability/
│   ├── shap_analysis.py # SHAP: bar, beeswarm, waterfall, dependence, WGI% pie
│   ├── permutation.py   # Temporal permutation importance (test-window only)
│   └── ablation.py      # 5-spec ablation + DM cluster-bootstrap + Wilcoxon
│
├── reports/
│   └── report_generator.py  # Auto LaTeX/CSV tables for dissertation
│
├── figures/             # All generated plots
├── utils/
│   └── tracking.py      # MLflow experiment tracking wrapper
│
├── pipeline.py                                # Main orchestrator — run this
├── build_multi_horizon_reference_datasets.py  # Optional documentation-only
│                                               # pre-step (see relatorio2)
└── requirements.txt
```

All artifacts (models .pkl, hyperparameter tables .csv, results .csv,
figures .png) are generated independently per forecast horizon, under
`h0/` .. `h5/` subdirectories of their respective base folder (e.g.
`models/artefacts/h3/`, `reports/h3/`, `tuning/results/h3/`,
`explainability/results/h3/`) — see `config/paths.py::horizon_dir()`.

## How to run

```bash
pip install -r requirements.txt

# 1. Extract data (run once — calls World Bank API)
python -c "
import sys; sys.path.insert(0,'.')
from data.extraction import run_extraction
run_extraction()
"

# 2. (Optional, documentation only) Build the wide multi-horizon reference
#    dataset — inputs (WDI+WGI) at year t plus alvo_t+1..alvo_t+5, with the
#    trailing empty target cells filled by each country's own historical
#    median and explicitly flagged. NOT used by step 3 below — see
#    preprocessing/multi_horizon_reference.py and relatorio2 for why this
#    is a separate, deliberately disconnected script.
python build_multi_horizon_reference_datasets.py

# 3. Run the full pipeline (genuine forecasts, h=1..5 years ahead)
python pipeline.py
```

## Methodological guarantees

| Problem (review)         | Solution                                  | File                         |
|--------------------------|-------------------------------------------|------------------------------|
| Look-ahead bias (MICE)   | MICE fitted inside each fold              | `validation/walk_forward.py` |
| PCA leakage              | PCA fitted on training slice only         | `features/engineer.py`       |
| Scaling leakage          | StandardScaler fitted inside each fold    | `validation/walk_forward.py` |
| Script-only architecture | sklearn TransformerMixin + Pipeline       | `preprocessing/`, `features/`|
| Fixed hyperparameters    | Optuna TPE, 50 trials, documented table   | `tuning/optuna_search.py`    |
| LSTM seq_len=1 (MLP)     | lookback=3, rolling-window sequences      | `models/lstm/model.py`       |
| Limited SHAP             | Bar + beeswarm + waterfall + dependence   | `explainability/shap_analysis.py` |
| No research hypothesis   | 5-spec ablation + DM + Wilcoxon           | `explainability/ablation.py` |
| No ACI sensitivity       | gamma×window grid, heatmaps               | `pipeline.py` → passo10      |
| Bayesian diagnostics     | PPCs + R-hat + ESS + trace plots          | `models/bayesian/model.py`   |
| SARIMAX coefficients     | coef, SE, CI95%, p-value table            | `models/sarimax/model.py`    |
| Monolithic config        | Split into 5 focused modules              | `config/`                    |

## Ablation specifications

| Specification   | Governance channel          | Research question answered              |
|-----------------|-----------------------------|-----------------------------------------|
| WDI_only        | None (baseline)             | What is the no-governance baseline?     |
| WDI_plus_PCA1   | Single latent factor (PC1)  | Does a governance index help?           |
| WDI_plus_6WGI   | Six raw WGI indicators      | Are individual WGI dimensions useful?   |
| WDI_plus_inter  | Governance × economic moder.| Does governance moderate econ channels? |
| WDI_PCA_inter   | PCA + interaction terms     | Combined channel — best spec?           |
