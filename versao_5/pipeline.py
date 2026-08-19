"""pipeline.py — Main pipeline orchestrator.

Architecture (per the document recommendation):

    pipeline.py
        │
        ├── config/          paths · variables · features · model_params · pipeline
        ├── data/            extraction (WDI + WGI via World Bank API)
        ├── preprocessing/   PanelMICEImputer · PanelScaler  (sklearn Transformers)
        ├── features/        FoldFeatureEngineer              (sklearn Transformer)
        ├── validation/      WalkForwardCV
        ├── models/          rf · xgb · sarimax · lstm · bayesian
        ├── tuning/          Optuna TPE search for all tree models
        ├── explainability/  SHAP · Permutation · Ablation
        ├── reports/         auto-generated dissertation tables + Markdown summary
        ├── figures/         all plots
        └── utils/           MLflow tracking · metadata

Key methodological guarantees
──────────────────────────────
1. MICE imputation fitted ONLY on training data inside each fold   → no look-ahead
2. PCA fitted ONLY on training data inside each fold               → no look-ahead
3. StandardScaler fitted ONLY on training data inside each fold    → no leakage
4. Lag/rolling features computed with pandas shift/rolling         → backward-only
5. Optuna TPE search with inner validation split inside each fold  → correct CV
6. LSTM lookback = config.LSTM['lookback'] ≥ 3                    → true sequences
7. Full hyperparameter table exported                              → reproducibility
8. Five ablation specifications                                    → research hypothesis
9. Bayesian PPCs + R-hat + ESS exported                           → diagnostic
10. SARIMAX coefficient table (SE, CI95, p-val) exported          → interpretability
"""
import os
import sys
import time
import glob
import pickle
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Ensure project root is on sys.path ───────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import config.paths     as paths
import config.variables as var
import config.features  as feat
import config.pipeline  as cfg_pipe
import config.model_params as mp

from preprocessing.imputer    import PanelMICEImputer
from preprocessing.scaler     import PanelScaler
from features.engineer        import FoldFeatureEngineer
from validation.walk_forward  import WalkForwardCV, build_horizon_target
from tuning.optuna_search     import export_hyperparameter_table
from explainability.shap_analysis import shap_tree_analysis, shap_kernel_analysis
from explainability.permutation   import permutation_importance
from explainability.ablation      import run_ablation
from reports.report_generator     import run_all_reports
from utils.tracking               import log_metadata


# ── Model trainers ────────────────────────────────────────────────────────────
from models.rf.model       import train as train_rf
from models.xgb.model      import train as train_xgb
from models.sarimax.model  import train as train_sarimax
from models.lstm.model     import train as train_lstm
from models.bayesian.model import train as train_bayesian

MODEL_TRAINERS = {
    "RandomForest":         train_rf,
    "XGBoost":              train_xgb,
    "SARIMAX":              train_sarimax,
    "LSTM":                 train_lstm,
    # CORRECTION (multi-seed + ablation-blindness requirements): every
    # trainer_fn is now called uniformly with priority_idx= and seed=
    # kwargs by validation/walk_forward.py — these lambdas must accept and
    # forward both (previously they only accepted the 4 positional args,
    # which would TypeError as soon as evaluate()/evaluate_multi_seed()
    # started passing the new kwargs).
    "Bayes_Partial":        lambda X_tr,y_tr,X_va,y_va,priority_idx=None,seed=None: train_bayesian(X_tr,y_tr,X_va,y_va,"partial",priority_idx=priority_idx,seed=seed),
    "Bayes_Complete":       lambda X_tr,y_tr,X_va,y_va,priority_idx=None,seed=None: train_bayesian(X_tr,y_tr,X_va,y_va,"complete",priority_idx=priority_idx,seed=seed),
}

# NEW in versão_5 (user request — genuine forecasts up to 5 years ahead,
# h=1..5, NOT including contemporaneous/nowcasting h=0 — see
# config/features.py::FORECAST_HORIZONS): the horizon that
# explainability/ablation treat as the reference/default. Written as an
# explicit constant (1), not feat.FORECAST_HORIZONS[0], purely so it stays
# correct and self-documenting even if the horizon list is ever reordered.
# All horizons (1..5) are still fully evaluated, trained and explained
# independently (see run_pipeline) — this constant only decides which
# single horizon's outputs sit at the flat legacy locations some older
# tooling might still look at.
PRIMARY_HORIZON = 1


# ── Step helpers ──────────────────────────────────────────────────────────────

def _load_clean_data() -> pd.DataFrame:
    """Load the INNER JOIN of clean WDI + WGI."""
    path = os.path.join(paths.AGGREGATED_DIR, "agregado_inner_join.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Run data extraction first. Expected: {path}"
        )
    df = pd.read_csv(path)
    print(f"  Data loaded: {df.shape}  "
          f"({df['country_code'].nunique()} countries, "
          f"{df['year'].min()}–{df['year'].max()})")
    return df


def _build_spec_datasets(df_raw: pd.DataFrame, horizon: int = None) -> dict:
    """
    Build one feature-engineered dataset per ablation specification, labeled
    for a single forecast horizon.

    CORRECTION (found in a later review pass, same family as the imputation
    look-ahead fix in the data-loading step): this used to call
    fe.fit(df_raw) on the ENTIRE panel — including the final holdout years —
    before transforming. That means the PCA governance factor and the lag
    structure feeding SHAP, permutation importance and the ablation study
    were partly learned from "future" data relative to the holdout split.
    _train_and_evaluate() already fits FoldFeatureEngineer on the pre-holdout
    training years only (see below) — this function now mirrors that same
    split, so SHAP/permutation/ablation see features built the identical,
    non-leaky way the models were actually trained.

    horizon : int, optional — CORRECTION (user request — "previsão para ano
    t, t+1, ..., t+5... cada combinação de inputs + ano de previsão deve
    gerar também um dataset independente"): this used to hard-default to
    feat.FORECAST_HORIZONS[0] (a single, implicit "primary" horizon,
    originally h=1). Now the caller (run_pipeline) invokes this once PER
    horizon, and each horizon's feature-engineered datasets are written to
    their own subdirectory (data/features/h{horizon}/) — independent
    datasets per (spec, horizon) combination, never overwriting each other.
    Defaults to PRIMARY_HORIZON only if not given, for any other caller.
    """
    horizon = PRIMARY_HORIZON if horizon is None else horizon

    years      = sorted(df_raw["year"].unique())
    split_idx  = int(len(years) * (1 - cfg_pipe.FINAL_HOLDOUT_RATIO))
    train_yr   = years[:split_idx]

    # permutation_importance() needs the TRUE label the model was actually
    # scored against to compute a meaningful RMSE increase, so the target
    # column attached here must be the SAME horizon-shifted label the model
    # for THIS horizon was trained on — not the contemporaneous target
    # var.TARGET (unless horizon==0, in which case they coincide).
    h_features_dir = paths.horizon_dir(paths.FEATURES_DIR, horizon)

    datasets = {}
    for spec_name, spec_cfg in feat.ABLATION_SPECS.items():
        print(f"  Building features for spec: {spec_name}  [h={horizon}]")
        fe = FoldFeatureEngineer(spec=spec_name)
        df_tr_only = df_raw[df_raw["year"].isin(train_yr)]
        fe.fit(df_tr_only)                   # fit only on pre-holdout years
        df_fe = fe.transform(df_raw)
        df_fe["__y_h__"] = build_horizon_target(df_fe, horizon, source_df=df_raw)
        out_path = os.path.join(h_features_dir, f"{spec_name}_features.csv")
        df_fe.to_csv(out_path, index=False)
        datasets[spec_name] = df_fe
    return datasets


def _train_and_evaluate(
    df_raw: pd.DataFrame,
    spec_name: str,
    wf: WalkForwardCV,
    horizon: int = 1,
) -> tuple[list, list]:
    """Train all models for one spec using walk-forward CV.

    horizon : int, optional (NEW in versão_5) — forecast horizon in calendar
    years, forwarded to WalkForwardCV.evaluate_multi_seed() and reproduced
    below for the final (post-CV) model trained on the full pre-holdout
    window. See validation/walk_forward.py::build_horizon_target for the
    full rationale (this is the genuine h-step-ahead forecast the original
    project's relatorio1 found was configured but never actually wired in).
    """
    all_fold_results = []
    hp_records       = []

    # CORRECTION (multi-seed requirement): the walk-forward CV metrics now
    # come from evaluate_multi_seed() (config.model_params.SEEDS, default
    # [7, 42, 90]) instead of a single evaluate() call, so
    # walkforward_results.csv carries one row per (fold, seed) and callers
    # can report mean±std per spec×model instead of a single-initialization
    # number. The FINAL saved model per spec×model (used by
    # explainability/ablation.py) still trains once, on the primary seed
    # (SEEDS[0] = 42), for continuity with every artifact already built
    # around a single reference model per spec×model.
    primary_seed = mp.SEEDS[0]

    for mod_name, trainer_fn in MODEL_TRAINERS.items():
        print(f"\n  [{spec_name}] [{mod_name}] [h={horizon}]")
        fold_results = wf.evaluate_multi_seed(df_raw, spec_name, trainer_fn, mod_name,
                                               horizon=horizon)
        all_fold_results.extend(fold_results)

        # ── Train final model on full train window → save ──────────────────
        years      = sorted(df_raw["year"].unique())
        split_idx  = int(len(years) * (1 - cfg_pipe.FINAL_HOLDOUT_RATIO))
        train_yr   = years[:split_idx]
        test_yr    = years[split_idx:]

        df_tr = df_raw[df_raw["year"].isin(train_yr)].copy()
        df_te = df_raw[df_raw["year"].isin(test_yr)].copy()

        # CORRECTION (target-imputation leak — see preprocessing/imputer.py
        # and validation/walk_forward.py for the full rationale): exclude
        # the target from MICE so it is never fabricated from the same
        # covariates the model is trained on.
        imputer = PanelMICEImputer(random_state=primary_seed, exclude_cols=[var.TARGET])
        imputer.fit(df_tr)
        df_tr_imp = imputer.transform(df_tr)
        df_te_imp = imputer.transform(df_te)

        fe     = FoldFeatureEngineer(spec=spec_name)
        combined = pd.concat([df_tr_imp, df_te_imp], ignore_index=True)
        fe.fit(df_tr_imp)
        df_combined_fe = fe.transform(combined)

        # CORRECTION (versão_5 — same genuine-horizon construction as
        # validation/walk_forward.py::evaluate(); see that file for the
        # full rationale). The label is the target `horizon` calendar years
        # after each row's own year, matched exactly by (country, year).
        df_combined_fe["__y_h__"] = build_horizon_target(df_combined_fe, horizon, source_df=df_raw)

        df_tr_fe = df_combined_fe[df_combined_fe["year"].isin(train_yr)]
        df_te_fe = df_combined_fe[df_combined_fe["year"].isin(test_yr)]

        feat_cols = [
            c for c in df_combined_fe.select_dtypes(include=[np.number]).columns
            if c not in {"year", var.TARGET, "__y_h__"} and "country" not in c.lower()
        ]
        if not feat_cols:
            continue

        from sklearn.preprocessing import StandardScaler
        tr_years_raw = df_tr_fe["year"].values
        X_tr_raw  = df_tr_fe[feat_cols].fillna(0).values
        y_tr      = df_tr_fe["__y_h__"].values
        X_te_raw  = df_te_fe[feat_cols].fillna(0).values
        y_te      = df_te_fe["__y_h__"].values

        # CORRECTION (leakage guard, versão_5 — same reasoning as
        # validation/walk_forward.py::evaluate()): a training row at year t
        # needs target[t+horizon]; if that falls at or after the holdout
        # (test_yr[0]), keeping it as a training example would use outcomes
        # from the final holdout window during training.
        if len(train_yr) > 0 and len(tr_years_raw) > 0:
            train_yr_max = max(train_yr)
            mask_leak = (tr_years_raw + horizon) <= train_yr_max
            X_tr_raw, y_tr = X_tr_raw[mask_leak], y_tr[mask_leak]

        # CORRECTION (target-imputation leak, cont.): drop rows whose true
        # (horizon-shifted) target was never observed — same reasoning as
        # validation/walk_forward.py::evaluate().
        mask_tr_y = ~pd.isna(y_tr)
        X_tr_raw, y_tr = X_tr_raw[mask_tr_y], y_tr[mask_tr_y]
        mask_te_y = ~pd.isna(y_te)
        X_te_raw, y_te = X_te_raw[mask_te_y], y_te[mask_te_y]
        if len(X_tr_raw) < 5 or len(X_te_raw) == 0:
            continue

        priority_idx = [i for i, c in enumerate(feat_cols)
                        if c.startswith("wgi_") or c.startswith("inter_")]

        scaler    = StandardScaler()
        X_tr_s    = scaler.fit_transform(X_tr_raw)
        X_te_s    = scaler.transform(X_te_raw)

        n_val   = max(1, int(len(X_tr_s) * 0.15))
        X_va_s  = X_tr_s[-n_val:]
        y_va    = y_tr[-n_val:]
        X_tr2   = X_tr_s[:-n_val]
        y_tr2   = y_tr[:-n_val]

        try:
            final_model = trainer_fn(X_tr2, y_tr2, X_va_s, y_va,
                                     priority_idx=priority_idx, seed=primary_seed)

            # CORRECTION (user request — "em cada ano de previsão tem que
            # gerar os artefactos (em pkl)... de forma independente e
            # salvar tudo"): every horizon now gets its own subdirectory
            # (models/artefacts/h{horizon}/) with an UNSUFFIXED filename
            # inside it — the directory itself disambiguates the horizon,
            # so ablation.py's model_dir-based lookup
            # ("{model_dir}/modelo_{spec}_{model}.pkl") keeps working
            # unmodified as long as it is pointed at the right horizon's
            # subdirectory (see run_pipeline's ablation loop). This
            # replaces the old flat-file "_h2"-suffix convention.
            h_models_dir = paths.horizon_dir(paths.MODELS_DIR, horizon)
            model_path = os.path.join(h_models_dir, f"modelo_{spec_name}_{mod_name}.pkl")
            # CORRECTION (root-cause diagnostic report, Secções 6/7/8, recomendação #3):
            # persist scaler + feat_cols alongside the model (this is the file that
            # explainability/ablation.py, explainability/innovations.py, and
            # _run_explainability() below actually load — it overwrites the one
            # written inside wf.evaluate()).
            from utils.model_io import save_model_bundle
            save_model_bundle(model_path, final_model, scaler, feat_cols)

            # Export SARIMAX coefficient table — independent per horizon.
            if mod_name == "SARIMAX" and hasattr(final_model, "export_coef_table"):
                coef_path = os.path.join(paths.horizon_dir(paths.REPORTS_DIR, horizon),
                                         f"sarimax_{spec_name}_coef.csv")
                final_model.export_coef_table(coef_path)

            # Export Bayesian diagnostics — independent per horizon.
            if "Bayes" in mod_name and hasattr(final_model, "export_diagnostics"):
                diag_dir = os.path.join(paths.horizon_dir(paths.EXPLAINABILITY_DIR, horizon), "bayesian")
                final_model.export_diagnostics(diag_dir)

            hp_records.append({
                "Specification":      spec_name,
                "Model":              mod_name,
                "Horizon":            horizon,
                "Search_Method":      getattr(final_model, "_search_method", "—"),
                "N_Trials":           mp.RF["n_trials"] if "Forest" in mod_name else "—",
                "Selection_Criterion":getattr(final_model, "_selection_criterion", "—"),
                "Seed":               getattr(final_model, "_seed", 42),
                "Best_Params":        str(getattr(final_model, "_best_params", "—")),
            })

        except Exception as exc:
            print(f"    Final model failed: {exc}")

    return all_fold_results, hp_records


# CORRECTION (root-cause diagnostic report, Secção 10.2 / recomendação #4):
# the old name "WDI_plus_PCA1" predates the current ABLATION_SPECS naming
# convention (A1_WDI_only, A2_WDI_PCA1, ...) and never matched any real key,
# so the KernelExplainer branch below never ran for ANY specification. The
# real equivalent spec is A2_WDI_PCA1 (WDI + single PCA governance factor);
# fixing the name restored KernelExplainer for that one spec only.
#
# CORRECTION (user request, sessão de 18/08/2026 — relatorio2 secção 10):
# that single-spec scope was never a deliberate cost/coverage trade-off —
# it is what was left over from the naming-bug fix above, not a considered
# decision. KernelExplainer now runs for every specification in
# ABLATION_SPECS (A1-A5), matching the tree-model path below. This constant
# is kept only as the historical/default reference spec for callers that
# still want a single-spec smoke test; _run_explainability() itself no
# longer reads it.
REFERENCE_SPEC_FOR_KERNEL_SHAP = "A2_WDI_PCA1"


def _run_explainability(spec_datasets: dict, horizon: int = None) -> None:
    """
    SHAP + permutation importance for all tree models on main datasets.

    horizon : int, optional — CORRECTION (user request — "em cada ano de
    previsão tem que gerar... as figuras (png)... de forma independente"):
    this used to run once, implicitly against whatever the caller's single
    spec_datasets happened to be labeled for. Now the caller (run_pipeline)
    invokes this once PER horizon, and every figure/CSV is written to its
    own explainability/results/h{horizon}/ subdirectory — never overwriting
    another horizon's explainability artifacts.
    """
    horizon = PRIMARY_HORIZON if horizon is None else horizon
    print("\n" + "=" * 60)
    print(f"  EXPLAINABILITY  [h={horizon}]")
    print("=" * 60)
    from utils.model_io import load_model_bundle
    exp_dir = paths.horizon_dir(paths.EXPLAINABILITY_DIR, horizon)
    h_models_dir = paths.horizon_dir(paths.MODELS_DIR, horizon)
    tree_models = ["RandomForest", "XGBoost"]

    for spec_name, df in spec_datasets.items():
        if "Sintetico" in spec_name:
            continue

        feat_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in {"year", var.TARGET, "__y_h__"} and "country" not in c.lower()
        ]
        years     = sorted(df["year"].unique())
        split_idx = int(len(years) * (1 - cfg_pipe.FINAL_HOLDOUT_RATIO))
        test_yr   = years[split_idx:]

        for mod_name in tree_models:
            pkl = os.path.join(h_models_dir, f"modelo_{spec_name}_{mod_name}.pkl")
            if not os.path.exists(pkl):
                continue
            model, scaler, trained_feat_cols = load_model_bundle(pkl)
            cols = trained_feat_cols if trained_feat_cols else feat_cols
            cols = [c for c in cols if c in df.columns]

            X_all_raw  = df[cols].fillna(0)
            X_test_raw = df[df["year"].isin(test_yr)][cols].fillna(0)
            # CORRECTION (versão_5): use the horizon-shifted label ("__y_h__",
            # built by _build_spec_datasets against the same primary horizon
            # this loaded model was trained on) — the contemporaneous
            # var.TARGET is a DIFFERENT quantity here and would silently
            # make permutation importance's RMSE-increase numbers meaningless.
            y_col  = "__y_h__" if "__y_h__" in df.columns else var.TARGET
            y_test = df[df["year"].isin(test_yr)][y_col].values

            # CORRECTION (root-cause diagnostic report, Secção 8, achado nº7 /
            # recomendação #3): the model was trained on StandardScaler-scaled
            # data (see _train_and_evaluate above). X_all/X_test used to be
            # passed to SHAP/permutation in raw, unscaled units — apply the
            # persisted scaler here before calling either function.
            if scaler is not None:
                X_all  = pd.DataFrame(scaler.transform(X_all_raw.values),  columns=cols, index=X_all_raw.index)
                X_test = pd.DataFrame(scaler.transform(X_test_raw.values), columns=cols, index=X_test_raw.index)
            else:
                print(f"    [aviso] {spec_name}_{mod_name}: pickle sem scaler persistido "
                      f"(formato anterior à correcção) — SHAP/permutação usam dados brutos.")
                X_all, X_test = X_all_raw, X_test_raw

            label = f"{spec_name}_{mod_name}"
            print(f"  SHAP + Permutation → {label}")
            try:
                shap_tree_analysis(model, X_all, label, exp_dir)
            except Exception as exc:
                print(f"    SHAP failed: {exc}")
            try:
                permutation_importance(model, X_test, y_test, label, exp_dir)
            except Exception as exc:
                print(f"    Permutation failed: {exc}")

        # KernelExplainer for non-tree models — CORRECTION (user request,
        # sessão de 18/08/2026): now runs for EVERY specification (A1-A5),
        # not just REFERENCE_SPEC_FOR_KERNEL_SHAP, and covers all 4
        # non-tree models (Bayes_Complete added — it had no technical
        # reason to be excluded: its predict() has the same one-row-in/
        # one-row-out shape as Bayes_Partial's, see models/bayesian/model.py).
        # Combined with the 2 tree models above, this is 6 models × 5 specs.
        cols_ref  = [c for c in feat_cols if c in df.columns]
        X_all_ref = df[cols_ref].fillna(0)
        for mod_name in ["SARIMAX", "LSTM", "Bayes_Partial", "Bayes_Complete"]:
            pkl = os.path.join(h_models_dir, f"modelo_{spec_name}_{mod_name}.pkl")
            if not os.path.exists(pkl):
                continue
            model, scaler, trained_feat_cols = load_model_bundle(pkl)
            cols = [c for c in (trained_feat_cols or cols_ref) if c in df.columns]
            X_bg = df[cols].fillna(0)
            if scaler is not None:
                X_bg = pd.DataFrame(scaler.transform(X_bg.values), columns=cols, index=X_bg.index)
            label = f"{spec_name}_{mod_name}"
            # CORRECTION: LSTMModel.predict() has sliding-window semantics
            # (one continuous series in, len(X)-lookback+1 rows out) that
            # broke KernelExplainer's one-row-in/one-row-out assumption —
            # this is what made LSTM fail silently before (see
            # models/lstm/model.py::predict_independent_rows docstring).
            predict_fn = (model.predict_independent_rows
                          if mod_name == "LSTM" and hasattr(model, "predict_independent_rows")
                          else None)
            try:
                shap_kernel_analysis(model, X_bg,
                                     X_bg.sample(min(200, len(X_bg)), random_state=42),
                                     label, exp_dir, predict_fn=predict_fn)
            except Exception as exc:
                print(f"    KernelExplainer failed ({label}): {exc}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_pipeline():
    print("\n" + "═" * 70)
    print("  AFRICA & MIDDLE EAST INDUSTRIAL ANALYSIS PIPELINE — versão_5")
    print("  Professional ML Pipeline | Walk-Forward CV | Optuna | MLflow")
    print("═" * 70)

    t_global = time.time()
    all_fold_results = []
    all_hp_records   = []

    # ── 1. Load clean aggregated data ─────────────────────────────────────────
    print("\n[1/6] Loading data...")
    df_raw = _load_clean_data()

    # ── 2. Walk-forward CV for all specs, models AND horizons ─────────────────
    # NEW in versão_5: config.features.FORECAST_HORIZONS is now actually
    # wired in — the original project (relatorio1) declared it but never
    # used it, so every prediction there was contemporaneous rather than a
    # genuine forward forecast. Every spec×model combination is now
    # evaluated once per GENUINE horizon (h=1..5 — no contemporaneous/h=0
    # case here, by explicit user request), so walkforward_results.csv lets
    # you compare every horizon's performance directly, in addition to the
    # spec/model/seed comparisons that already existed.
    print(f"\n[2/6] Walk-forward cross-validation + model training "
          f"(horizontes: {feat.FORECAST_HORIZONS})...")
    wf = WalkForwardCV()

    for horizon in feat.FORECAST_HORIZONS:
        for spec_name in feat.ABLATION_SPECS:
            print(f"\n{'─'*60}")
            print(f"  Specification: {spec_name}  |  Horizonte: {horizon} ano(s)")
            fold_res, hp_recs = _train_and_evaluate(df_raw, spec_name, wf, horizon=horizon)
            all_fold_results.extend(fold_res)
            all_hp_records.extend(hp_recs)

    # ── 3. Save walk-forward results ──────────────────────────────────────────
    print("\n[3/6] Saving results...")
    # CORRECTION (multi-seed + computational-cost requirements): added
    # "seed", "fit_time_s", "predict_time_s", "peak_mem_mb" (see
    # validation/walk_forward.py::FoldResult / _peak_rss_mb) and
    # "n_dropped_missing_target" (target-imputation-leak fix — how many
    # rows had a genuinely missing label and were excluded from that fold's
    # training/evaluation, rather than fabricated by MICE).
    # NEW in versão_5: "horizon" (forecast horizon in calendar years — see
    # validation/walk_forward.py::build_horizon_target) and
    # "n_dropped_leak_guard" (training rows dropped per fold because their
    # horizon-shifted label would have required already-known future data —
    # see evaluate()'s leakage-guard comment) are now part of every row.
    # CORRECTION (user request — "incluir as métricas MAPE também nos
    # logs"): MAPE and n_mape_excluded were computed per fold (FoldResult)
    # but missing from this CSV — only RMSE/MAE/R2/MASE were exported here.
    df_wf = pd.DataFrame([
        {"fold": r.fold, "spec": r.spec, "model": r.model, "seed": r.seed,
         "horizon": r.horizon,
         "RMSE": r.RMSE, "MAE": r.MAE, "R2": r.R2, "MASE": r.MASE, "MAPE": r.MAPE,
         "n_train": r.n_train, "n_test": r.n_test,
         "fit_time_s": r.fit_time_s, "predict_time_s": r.predict_time_s,
         "peak_mem_mb": r.peak_mem_mb,
         "n_dropped_missing_target": r.n_dropped_missing_target,
         "n_dropped_leak_guard": r.n_dropped_leak_guard,
         "n_mape_excluded": r.n_mape_excluded}
        for r in all_fold_results
    ])
    wf_path = os.path.join(paths.REPORTS_DIR, "walkforward_results.csv")
    df_wf.to_csv(wf_path, index=False)

    # CORRECTION (user request — "em cada ano de previsão tem que gerar...
    # os resultados do modelo (em csv)... de forma independente e salvar
    # tudo"): in addition to the combined walkforward_results.csv above
    # (kept because report_generator.py's cross-horizon comparison tables
    # need every horizon's rows together), each horizon's own subset is
    # ALSO written out independently, under reports/h{horizon}/.
    for _h in feat.FORECAST_HORIZONS:
        df_wf_h = df_wf[df_wf["horizon"] == _h]
        if not df_wf_h.empty:
            df_wf_h.to_csv(
                os.path.join(paths.horizon_dir(paths.REPORTS_DIR, _h), "walkforward_results.csv"),
                index=False,
            )

    # Combined hyperparameter table (all horizons together — used by the
    # dissertation-wide table_hyperparameters.csv/.tex).
    hp_path = export_hyperparameter_table(all_hp_records)

    # CORRECTION (user request — "em cada ano de previsão tem que gerar...
    # os parâmetros e hiperparâmetros (em csv)... de forma independente"):
    # per-horizon hyperparameter table, alongside the combined one above.
    for _h in feat.FORECAST_HORIZONS:
        recs_h = [r for r in all_hp_records if r.get("Horizon") == _h]
        if recs_h:
            export_hyperparameter_table(recs_h, out_dir=paths.horizon_dir(paths.TUNING_DIR, _h))

    # ── 4. Explainability ─────────────────────────────────────────────────────
    # CORRECTION (user request — "em cada ano de previsão tem que gerar...
    # as figuras (png)... de forma independente e salvar tudo"): this used
    # to run ONCE, implicitly against a single "primary" horizon's
    # datasets. It now runs once PER horizon (t, t+1, ..., t+5), each with
    # its own feature datasets (_build_spec_datasets(df_raw, horizon=h))
    # and its own explainability/results/h{h}/ output directory — SHAP,
    # permutation importance and (below) ablation are therefore produced
    # independently for every forecast year, not only for h=1.
    print(f"\n[4/6] Explainability (SHAP + Permutation) — todos os horizontes "
          f"{feat.FORECAST_HORIZONS}...")
    spec_datasets_by_horizon = {}
    for _h in feat.FORECAST_HORIZONS:
        spec_datasets_h = _build_spec_datasets(df_raw, horizon=_h)
        spec_datasets_by_horizon[_h] = spec_datasets_h
        _run_explainability(spec_datasets_h, horizon=_h)
    # Kept for anything downstream that still expects a single dict shaped
    # like the pre-versão_5 API (the primary/reference horizon's datasets).
    spec_datasets = spec_datasets_by_horizon[PRIMARY_HORIZON]

    # ── 5. Ablation study ─────────────────────────────────────────────────────
    # Same per-horizon independence as step 4: model_dir and out_dir both
    # point at that horizon's own subdirectory, so ablation.py's existing
    # "{model_dir}/modelo_{spec}_{model}.pkl" lookup needs no changes — it
    # simply resolves within whichever horizon's directory it is pointed at.
    print(f"\n[5/6] Ablation study — todos os horizontes {feat.FORECAST_HORIZONS}...")
    years      = sorted(df_raw["year"].unique())
    split_idx  = int(len(years) * (1 - cfg_pipe.FINAL_HOLDOUT_RATIO))
    test_cutoff = years[split_idx]

    for _h in feat.FORECAST_HORIZONS:
        abl_dir_h = paths.horizon_dir(os.path.join(paths.EXPLAINABILITY_DIR, "ablation"), _h)
        run_ablation(
            spec_datasets=spec_datasets_by_horizon[_h],
            model_names=list(MODEL_TRAINERS.keys()),
            model_dir=paths.horizon_dir(paths.MODELS_DIR, _h),
            out_dir=abl_dir_h,
            test_year_cutoff=test_cutoff,
        )
    # Primary horizon's ablation output feeds the single combined
    # dissertation-wide ablation table below (run_all_reports); every
    # horizon's own ablation_results.csv/ablation_dm_tests.csv still exists
    # independently under explainability/results/ablation/h{h}/.
    abl_dir = paths.horizon_dir(os.path.join(paths.EXPLAINABILITY_DIR, "ablation"), PRIMARY_HORIZON)

    # ── 6. Reports ────────────────────────────────────────────────────────────
    print("\n[6/6] Generating dissertation reports...")

    # CORRECTION (root-cause diagnostic report, Secção 10.1 / recomendação #5):
    # df_wf has one row per (fold, spec, model). "best_RMSE" used to report only
    # the single minimum fold-level RMSE, with no label distinguishing it from
    # the mean-per-spec×model figure in table_performance.csv — the two are
    # different statistics of the same data and were easy to misread as
    # inconsistent. Both are now reported, explicitly labelled.
    best_rmse_single_fold = float(df_wf["RMSE"].dropna().min()) if not df_wf.empty else np.nan
    best_row_single_fold  = df_wf.loc[df_wf["RMSE"].idxmin()] if not df_wf.empty else {}

    # CORRECTION (versão_5): "horizon" must be part of this groupby too —
    # otherwise a spec×model's h=1 and h=2 fold RMSEs get silently averaged
    # together into one number that describes neither horizon honestly.
    if not df_wf.empty:
        df_mean_grp = df_wf.groupby(["spec", "model", "horizon"])["RMSE"].mean().reset_index()
        best_row_mean = df_mean_grp.loc[df_mean_grp["RMSE"].idxmin()]
        best_rmse_mean = float(best_row_mean["RMSE"])
    else:
        best_row_mean  = {}
        best_rmse_mean = np.nan

    # CORRECTION (root-cause diagnostic report, Secção 10.2 / recomendação #4):
    # "sarimax_WDI_plus_PCA1_coef.csv" never matched any file actually written
    # by _train_and_evaluate() (real files are sarimax_{spec_name}_coef.csv).
    # Aggregate every per-spec SARIMAX coefficient file that was produced into
    # one combined table instead of pointing at a name that never existed.
    # CORRECTION (versão_5): these files now live under per-horizon
    # subdirectories (reports/h{horizon}/sarimax_{spec}_coef.csv) — the glob
    # must look inside every "h*" subdirectory, and the aggregated table
    # gets an explicit Horizon column so per-spec rows from different
    # horizons are never confused with each other.
    sarimax_coef_files = sorted(glob.glob(os.path.join(paths.REPORTS_DIR, "h*", "sarimax_*_coef.csv")))
    sarimax_coef_csv = None
    if sarimax_coef_files:
        frames = []
        for fp in sarimax_coef_files:
            spec_from_name = os.path.basename(fp)[len("sarimax_"):-len("_coef.csv")]
            horizon_from_dir = os.path.basename(os.path.dirname(fp))  # "h0", "h1", ...
            d = pd.read_csv(fp)
            d.insert(0, "Horizon", horizon_from_dir.lstrip("h"))
            d.insert(0, "Specification", spec_from_name)
            frames.append(d)
        combined = pd.concat(frames, ignore_index=True)
        sarimax_coef_csv = os.path.join(paths.REPORTS_DIR, "sarimax_all_specs_coef.csv")
        combined.to_csv(sarimax_coef_csv, index=False)
        print(f"  [correction] SARIMAX coef table aggregated from {len(sarimax_coef_files)} "
              f"spec×horizon files → {sarimax_coef_csv}")

    run_all_reports(
        results_csv=wf_path,
        hp_csv=hp_path,
        sarimax_coef_csv=sarimax_coef_csv,
        ablation_csv=os.path.join(abl_dir, "ablation_results.csv"),
        ablation_dm_csv=os.path.join(abl_dir, "ablation_dm_tests.csv"),
        summary_kv={
            "best_RMSE_single_fold (mínimo entre as linhas de fold individuais)": best_rmse_single_fold,
            "best_model_single_fold":  getattr(best_row_single_fold, "model", "—"),
            "best_spec_single_fold":   getattr(best_row_single_fold, "spec",  "—"),
            "best_horizon_single_fold": getattr(best_row_single_fold, "horizon", "—"),
            "best_RMSE_mean_per_group (média por especificação×modelo×horizonte — comparável à tabela de desempenho)": best_rmse_mean,
            "best_model_mean_per_group": best_row_mean.get("model", "—") if isinstance(best_row_mean, dict) else best_row_mean["model"],
            "best_spec_mean_per_group":  best_row_mean.get("spec",  "—") if isinstance(best_row_mean, dict) else best_row_mean["spec"],
            "best_horizon_mean_per_group": best_row_mean.get("horizon", "—") if isinstance(best_row_mean, dict) else best_row_mean["horizon"],
            "n_models":   len(MODEL_TRAINERS),
            "n_specs":    len(feat.ABLATION_SPECS),
            "n_horizons": len(feat.FORECAST_HORIZONS),
            "n_folds":    cfg_pipe.WF_N_FOLDS,
        },
    )
    best_rmse = best_rmse_single_fold

    # ── Metadata ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t_global
    log_metadata(
        step="pipeline_v4",
        params={
            "n_models": len(MODEL_TRAINERS),
            "n_specs":  len(feat.ABLATION_SPECS),
            "n_folds":  cfg_pipe.WF_N_FOLDS,
            "lookback": mp.LSTM["lookback"],
            "optuna_trials": mp.RF["n_trials"],
        },
        metrics={"total_time_s": round(elapsed, 1), "best_RMSE": best_rmse},
        output_files=[wf_path, hp_path],
    )

    print(f"\n{'═'*70}")
    print(f"  PIPELINE COMPLETE — {elapsed:.1f}s")
    print(f"{'═'*70}")
    print(f"  Walk-forward results (combinado, todos os horizontes) → {wf_path}")
    print(f"  Hyperparameter table (combinado)                     → {hp_path}")
    print(f"  Reports                                              → {paths.REPORTS_DIR}/")
    print(f"  Figures / explicabilidade (por horizonte)            → {paths.EXPLAINABILITY_DIR}/h{{1..5}}/")
    print(f"  Modelos (.pkl, por horizonte)                        → {paths.MODELS_DIR}/h{{1..5}}/")
    print(f"  Resultados e hiperparâmetros independentes por horizonte:")
    for _h in feat.FORECAST_HORIZONS:
        print(f"    h={_h}: {paths.horizon_dir(paths.REPORTS_DIR, _h)}/, "
              f"{paths.horizon_dir(paths.TUNING_DIR, _h)}/")


if __name__ == "__main__":
    run_pipeline()
