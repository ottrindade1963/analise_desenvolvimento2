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
   
    "Bayes_Partial":        lambda X_tr,y_tr,X_va,y_va,priority_idx=None,seed=None: train_bayesian(X_tr,y_tr,X_va,y_va,"partial",priority_idx=priority_idx,seed=seed),
    "Bayes_Complete":       lambda X_tr,y_tr,X_va,y_va,priority_idx=None,seed=None: train_bayesian(X_tr,y_tr,X_va,y_va,"complete",priority_idx=priority_idx,seed=seed),
}


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
   
    horizon = PRIMARY_HORIZON if horizon is None else horizon

    years      = sorted(df_raw["year"].unique())
    split_idx  = int(len(years) * (1 - cfg_pipe.FINAL_HOLDOUT_RATIO))
    train_yr   = years[:split_idx]
   
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
   
    all_fold_results = []
    hp_records       = []

   
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

       
        imputer = PanelMICEImputer(random_state=primary_seed, exclude_cols=[var.TARGET])
        imputer.fit(df_tr)
        df_tr_imp = imputer.transform(df_tr)
        df_te_imp = imputer.transform(df_te)

        fe     = FoldFeatureEngineer(spec=spec_name)
        combined = pd.concat([df_tr_imp, df_te_imp], ignore_index=True)
        fe.fit(df_tr_imp)
        df_combined_fe = fe.transform(combined)

       
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

        
        if len(train_yr) > 0 and len(tr_years_raw) > 0:
            train_yr_max = max(train_yr)
            mask_leak = (tr_years_raw + horizon) <= train_yr_max
            X_tr_raw, y_tr = X_tr_raw[mask_leak], y_tr[mask_leak]

       
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

          
            h_models_dir = paths.horizon_dir(paths.MODELS_DIR, horizon)
            model_path = os.path.join(h_models_dir, f"modelo_{spec_name}_{mod_name}.pkl")
           
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


REFERENCE_SPEC_FOR_KERNEL_SHAP = "A2_WDI_PCA1"


def _run_explainability(spec_datasets: dict, horizon: int = None) -> None:
  
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
           
            y_col  = "__y_h__" if "__y_h__" in df.columns else var.TARGET
            y_test = df[df["year"].isin(test_yr)][y_col].values

            
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

   
    for _h in feat.FORECAST_HORIZONS:
        df_wf_h = df_wf[df_wf["horizon"] == _h]
        if not df_wf_h.empty:
            df_wf_h.to_csv(
                os.path.join(paths.horizon_dir(paths.REPORTS_DIR, _h), "walkforward_results.csv"),
                index=False,
            )
   
    hp_path = export_hyperparameter_table(all_hp_records)
   
    for _h in feat.FORECAST_HORIZONS:
        recs_h = [r for r in all_hp_records if r.get("Horizon") == _h]
        if recs_h:
            export_hyperparameter_table(recs_h, out_dir=paths.horizon_dir(paths.TUNING_DIR, _h))

   
    print(f"\n[4/6] Explainability (SHAP + Permutation) — todos os horizontes "
          f"{feat.FORECAST_HORIZONS}...")
    spec_datasets_by_horizon = {}
    for _h in feat.FORECAST_HORIZONS:
        spec_datasets_h = _build_spec_datasets(df_raw, horizon=_h)
        spec_datasets_by_horizon[_h] = spec_datasets_h
        _run_explainability(spec_datasets_h, horizon=_h)
   
    spec_datasets = spec_datasets_by_horizon[PRIMARY_HORIZON]

    
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
   
    abl_dir = paths.horizon_dir(os.path.join(paths.EXPLAINABILITY_DIR, "ablation"), PRIMARY_HORIZON)

    # ── 6. Reports ────────────────────────────────────────────────────────────
    print("\n[6/6] Generating dissertation reports...")

   
    best_rmse_single_fold = float(df_wf["RMSE"].dropna().min()) if not df_wf.empty else np.nan
    best_row_single_fold  = df_wf.loc[df_wf["RMSE"].idxmin()] if not df_wf.empty else {}

  
    if not df_wf.empty:
        df_mean_grp = df_wf.groupby(["spec", "model", "horizon"])["RMSE"].mean().reset_index()
        best_row_mean = df_mean_grp.loc[df_mean_grp["RMSE"].idxmin()]
        best_rmse_mean = float(best_row_mean["RMSE"])
    else:
        best_row_mean  = {}
        best_rmse_mean = np.nan

   
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
