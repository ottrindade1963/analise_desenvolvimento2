"""validation/walk_forward.py — Walk-forward cross-validation with shape-safe prediction."""
import os
import pickle
import time
import resource
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from sklearn.preprocessing import StandardScaler

def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # KB → MB (Linux)

import config.pipeline     as cfg_pipe
import config.variables    as var
import config.paths        as paths
import config.model_params as mp
import config.features     as feat_cfg
from preprocessing.imputer  import PanelMICEImputer
from preprocessing.temporal import year_exact_shift
from features.engineer      import FoldFeatureEngineer


def build_horizon_target(df: pd.DataFrame, horizon: int,
                          source_df: pd.DataFrame = None) -> np.ndarray:
   
    lookup_source = source_df if source_df is not None else df
    return year_exact_shift(df, var.TARGET, -horizon, source=lookup_source)


def _governance_priority_idx(feat_cols: list):
    target_lag_idx = [i for i, c in enumerate(feat_cols)
                       if c.startswith(f"{var.TARGET}_lag")]
    governance_idx = [i for i, c in enumerate(feat_cols)
                       if c.startswith("wgi_") or c.startswith("inter_")]
    return target_lag_idx, governance_idx


@dataclass
class FoldResult:
    fold:        int
    spec:        str
    model:       str
    n_train:     int
    n_test:      int
    train_years: list
    test_years:  list
    RMSE:  float = np.nan
    MAE:   float = np.nan
    R2:    float = np.nan
    MAPE:  float = np.nan
    MASE:  float = np.nan
    
    seed:          int   = None
    fit_time_s:    float = np.nan
    predict_time_s: float = np.nan
    peak_mem_mb:   float = np.nan
    n_dropped_missing_target: int = 0
    # NEW in versão_5:
    horizon:                int = 1
    n_dropped_leak_guard:   int = 0
    n_mape_excluded:        int = 0


def _metrics(y_true, y_pred, y_train=None) -> dict:
    yt = np.asarray(y_true,  float)
    yp = np.asarray(y_pred, float)

    # FIX: align lengths — LSTM returns fewer rows due to lookback
    min_len = min(len(yt), len(yp))
    yt = yt[-min_len:]
    yp = yp[-min_len:]

    m  = ~np.isnan(yt) & ~np.isnan(yp)
    yt, yp = yt[m], yp[m]
    n = len(yt)
    if n < 3:
        return dict(RMSE=np.nan, MAE=np.nan, R2=np.nan, MAPE=np.nan, MASE=np.nan, n_mape_excluded=0)

    res    = yt - yp
    ss_res = np.sum(res ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2) + 1e-10
    mae    = float(np.mean(np.abs(res)))


    MAPE_MIN_ABS_TARGET = 0.5  # percentage points
    mape_mask = np.abs(yt) >= MAPE_MIN_ABS_TARGET
    n_mape_excluded = int((~mape_mask).sum())
    if mape_mask.any():
        mape = float(np.mean(np.abs(res[mape_mask] / yt[mape_mask])) * 100)
    else:
        mape = np.nan

    if y_train is not None and len(y_train) > 1:
        naive_mae = float(np.mean(np.abs(np.diff(np.asarray(y_train, float)))))
    else:
        naive_mae = float(np.mean(np.abs(np.diff(yt)))) if n > 1 else 1e-10

    return dict(
        RMSE=float(np.sqrt(np.mean(res ** 2))),
        MAE=mae,
        R2=float(1 - ss_res / ss_tot),
        MAPE=mape,
        MASE=mae / (naive_mae + 1e-10),
        n_mape_excluded=n_mape_excluded,
    )


def _print_seed_comparison_table(spec: str, model_name: str, results: list,
                                  horizon: int = None) -> None:
   
    if not results:
        return
    by_seed: dict = {}
    for r in results:
        by_seed.setdefault(r.seed, []).append(r)

    h_label = f"  h={horizon}" if horizon is not None else ""
    print(f"      ┌─ Comparação RMSE/MAPE por seed — [{spec}] [{model_name}]{h_label} ─")
    header = f"      │ {'seed':>6} │ {'RMSE (média)':>13} │ {'MAPE (média)':>13} │ {'n_folds':>7} │"
    print(header)
    print(f"      │{'-'*8}│{'-'*15}│{'-'*15}│{'-'*9}│")
    for seed in sorted(by_seed):
        rows = by_seed[seed]
        rmse_vals = [r.RMSE for r in rows if not np.isnan(r.RMSE)]
        mape_vals = [r.MAPE for r in rows if not np.isnan(r.MAPE)]
        rmse_m = f"{np.mean(rmse_vals):.3f}" if rmse_vals else "nan"
        mape_m = f"{np.mean(mape_vals):.2f}%" if mape_vals else "nan"
        print(f"      │ {seed:>6} │ {rmse_m:>13} │ {mape_m:>13} │ {len(rows):>7} │")
    print(f"      └{'─'*8}┴{'─'*15}┴{'─'*15}┴{'─'*9}┘")


class WalkForwardCV:
    def __init__(self,
                 n_folds: int         = cfg_pipe.WF_N_FOLDS,
                 min_train_frac: float = cfg_pipe.WF_MIN_TRAIN):
        self.n_folds        = n_folds
        self.min_train_frac = min_train_frac

    def split(self, years: list) -> list:
        n       = len(years)
        min_tr  = max(5, int(n * self.min_train_frac))
        avail   = n - min_tr
        n_folds = min(self.n_folds, max(1, avail))
        fold_sz = max(1, avail // n_folds)
        splits  = []
        for f in range(n_folds):
            te_start = min_tr + f * fold_sz
            te_end   = min(n, te_start + fold_sz)
            if te_start >= n:
                break
            splits.append((years[:te_start], years[te_start:te_end]))
        return splits

    def evaluate(self, df_raw: pd.DataFrame, spec: str,
                 trainer_fn, model_name: str,
                 save_model: bool = True,
                 seed: int = None,
                 model_suffix: str = "",
                 horizon: int = 1) -> list:
       
        seed = mp.SEED if seed is None else seed

        years   = sorted(df_raw["year"].unique())
        splits  = self.split(years)
        results = []
        best_model      = None
        best_scaler     = None   # CORRECTION: persist alongside the model (see utils/model_io.py)
        best_feat_cols  = None

        for fold_idx, (train_yr, test_yr) in enumerate(splits, 1):
            df_tr = df_raw[df_raw["year"].isin(train_yr)].copy()
            df_te = df_raw[df_raw["year"].isin(test_yr)].copy()

          
            imputer = PanelMICEImputer(max_iter=20, random_state=seed,
                                       exclude_cols=[var.TARGET])
            imputer.fit(df_tr)
            df_tr_imp = imputer.transform(df_tr)
            df_te_imp = imputer.transform(df_te)

            # Step 2: Feature engineering fitted on train only
            fe = FoldFeatureEngineer(spec=spec)
            fe.fit(df_tr_imp)
            df_combined    = pd.concat([df_tr_imp, df_te_imp], ignore_index=True)
            df_combined_fe = fe.transform(df_combined)

          
            df_combined_fe["__y_h__"] = build_horizon_target(
                df_combined_fe, horizon, source_df=df_raw
            )

            df_tr_fe = df_combined_fe[df_combined_fe["year"].isin(train_yr)]
            df_te_fe = df_combined_fe[df_combined_fe["year"].isin(test_yr)]

            feat_cols = [
                c for c in df_combined_fe.select_dtypes(include=[np.number]).columns
                if c not in {"year", var.TARGET, "__y_h__"} and "country" not in c.lower()
            ]
            if not feat_cols or "__y_h__" not in df_tr_fe.columns:
                continue

            tr_years_arr = df_tr_fe["year"].values
            X_tr = df_tr_fe[feat_cols].fillna(0).values
            y_tr = df_tr_fe["__y_h__"].values
            X_te = df_te_fe[feat_cols].fillna(0).values
            y_te = df_te_fe["__y_h__"].values

      
            n_leak_guard = 0
            if len(train_yr) > 0 and len(tr_years_arr) > 0:
                train_yr_max = max(train_yr)
                mask_leak = (tr_years_arr + horizon) <= train_yr_max
                n_leak_guard = int((~mask_leak).sum())
                if n_leak_guard:
                    X_tr, y_tr = X_tr[mask_leak], y_tr[mask_leak]
           
            n_dropped = 0
            mask_tr_y = ~pd.isna(y_tr)
            if not mask_tr_y.all():
                n_dropped += int((~mask_tr_y).sum())
                X_tr, y_tr = X_tr[mask_tr_y], y_tr[mask_tr_y]
            mask_te_y = ~pd.isna(y_te)
            if not mask_te_y.all():
                n_dropped += int((~mask_te_y).sum())
                X_te, y_te = X_te[mask_te_y], y_te[mask_te_y]

            if len(X_tr) < 5 or len(X_te) == 0:
                continue

        
            priority_idx = _governance_priority_idx(feat_cols)

            # Step 3: Scaling fitted on train only
            scaler  = StandardScaler()
            X_tr_s  = scaler.fit_transform(X_tr)
            X_te_s  = scaler.transform(X_te)

            # Step 4: Inner validation split
            n_val   = max(1, int(len(X_tr_s) * 0.15))
            X_val_s = X_tr_s[-n_val:]
            y_val   = y_tr[-n_val:]
            X_tr2   = X_tr_s[:-n_val]
            y_tr2   = y_tr[:-n_val]

            # Step 5: Train and predict, with computational-cost instrumentation
            fit_time_s = predict_time_s = np.nan
            peak_mem_mb = np.nan
            try:
                t0 = time.perf_counter()
                model = trainer_fn(X_tr2, y_tr2, X_val_s, y_val,
                                   priority_idx=priority_idx, seed=seed)
                fit_time_s = time.perf_counter() - t0
                peak_mem_mb = _peak_rss_mb()

                t1 = time.perf_counter()
                y_pred = np.asarray(model.predict(X_te_s), dtype=float)
                predict_time_s = time.perf_counter() - t1

                # FIX: length alignment handled inside _metrics
                m      = _metrics(y_te, y_pred, y_tr)
                best_model     = model
                best_scaler    = scaler
                best_feat_cols = feat_cols
            except Exception as exc:
                print(f"      Fold {fold_idx} [{model_name}] failed: {exc}")
                m = dict(RMSE=np.nan, MAE=np.nan, R2=np.nan, MAPE=np.nan, MASE=np.nan, n_mape_excluded=0)

            results.append(FoldResult(
                fold=fold_idx, spec=spec, model=model_name,
                n_train=len(X_tr2), n_test=len(X_te),
                train_years=list(train_yr), test_years=list(test_yr),
                seed=seed, fit_time_s=fit_time_s, predict_time_s=predict_time_s,
                peak_mem_mb=peak_mem_mb, n_dropped_missing_target=n_dropped,
                horizon=horizon, n_dropped_leak_guard=n_leak_guard,
                **m,
            ))

            # CORRECTION (user request — 4 decimal places are noise for RMSE/R2
            # at this scale): reduced from .4f to .3f.
            rmse_s = f"{m['RMSE']:.3f}" if not np.isnan(m['RMSE']) else "nan"
            r2_s   = f"{m['R2']:.3f}"   if not np.isnan(m['R2'])   else "nan"
           
            mape_s = f"{m['MAPE']:.2f}%" if not np.isnan(m['MAPE']) else "nan"
            cost_s = (f"fit={fit_time_s:.3f}s  predict={predict_time_s:.4f}s  "
                      f"peak_mem={peak_mem_mb:.1f}MB")
            extra = []
            if n_leak_guard:
                extra.append(f"{n_leak_guard} treino descartadas: rótulo t+{horizon} exigiria dados ainda "
                              f"não conhecidos nesta janela")
            if n_dropped:
                extra.append(f"{n_dropped} linhas descartadas: alvo em t+{horizon} indisponível")
            if m.get("n_mape_excluded"):
                extra.append(f"{m['n_mape_excluded']} linhas excluídas do MAPE: |alvo|<0.5")
            print(f"      Fold {fold_idx}/{len(splits)} — RMSE={rmse_s}  R²={r2_s}  MAPE={mape_s}"
                  f"  |  {cost_s}"
                  f"  |  seed={seed}  h={horizon}"
                  + ("  [" + "; ".join(extra) + "]" if extra else ""))

      
        if save_model and best_model is not None:
           
            h_models_dir = paths.horizon_dir(paths.MODELS_DIR, horizon)
            pkl_path = os.path.join(
                h_models_dir, f"modelo_{spec}_{model_name}{model_suffix}.pkl"
            )
            from utils.model_io import save_model_bundle
            save_model_bundle(pkl_path, best_model, best_scaler, best_feat_cols)

        return results

    def evaluate_multi_seed(self, df_raw: pd.DataFrame, spec: str,
                             trainer_fn, model_name: str,
                             seeds: list = None,
                             save_model: bool = True,
                             horizon: int = 1) -> list:
      
        seeds = list(mp.SEEDS) if seeds is None else list(seeds)
        try:
            from joblib import Parallel, delayed
            n_jobs = min(len(seeds), max(1, (os.cpu_count() or 2) - 1))
        except ImportError:
            Parallel = None

        def _run(i, s):
            suffix = "" if i == 0 else f"_seed{s}"
            print(f"    ── seed {s} ({'primary' if i == 0 else 'secondary'}) — horizon={horizon} ano(s) ──")
            return self.evaluate(df_raw, spec, trainer_fn, model_name,
                                 save_model=save_model, seed=s, model_suffix=suffix,
                                 horizon=horizon)

        if Parallel is not None and len(seeds) > 1:
            
            nested = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_run)(i, s) for i, s in enumerate(seeds)
            )
        else:
            nested = [_run(i, s) for i, s in enumerate(seeds)]

        flat = [r for sub in nested for r in sub]
       
        _print_seed_comparison_table(spec, model_name, flat, horizon=horizon)

        return flat
