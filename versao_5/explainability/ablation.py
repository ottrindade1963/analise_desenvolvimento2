"""explainability/ablation.py — Ablation study across governance specifications.

Directly answers the research hypothesis:
  "Do WGI governance indicators improve industrial value-added forecasting,
   and through which channel?"

NOTE: models were trained on StandardScaler-transformed data inside each
walk-forward fold. This module replicates that scaling (fit on train,
transform on test) before calling predict(), so RMSE values are comparable
to the walk-forward results.
"""
import os
import pickle
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

import config.paths    as paths
import config.features as feat
import config.variables as var
from utils.model_io import load_model_bundle


def _rmse(y_true, y_pred) -> float:
    m = ~np.isnan(y_true) & ~np.isnan(y_pred)
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2))) if m.sum() >= 3 else np.nan


def _r2(y_true, y_pred) -> float:
    m = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt, yp = y_true[m], y_pred[m]
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2) + 1e-10
    return float(1 - ss_res / ss_tot)


def _dm_cluster_bootstrap(e1, e2, cluster_ids, n_boot=1000) -> dict:
    d       = e1**2 - e2**2
    d_obs   = float(np.mean(d))
    clusters = np.unique(cluster_ids)
    nc      = len(clusters)
    if nc < 3:
        return {"dm_stat": np.nan, "p_value": np.nan, "n_clusters": nc}
    d_by_c = {c: d[cluster_ids == c] for c in clusters}
    boot   = np.array([
        np.mean(np.concatenate([d_by_c[clusters[i]]
                                for i in np.random.choice(nc, nc, replace=True)]))
        for _ in range(n_boot)
    ])
    se  = float(np.std(boot, ddof=1))
    dm  = d_obs / (se + 1e-10)
    pv  = float(min(2 * min(np.mean(boot > d_obs), np.mean(boot < d_obs)), 1.0))
    return {"dm_stat": dm, "p_value": pv, "d_mean": d_obs,
            "ci_lower": float(np.percentile(boot, 2.5)),
            "ci_upper": float(np.percentile(boot, 97.5)),
            "n_clusters": nc}


def run_ablation(
    spec_datasets: dict,
    model_names: list,
    model_dir: str,
    out_dir: str,
    test_year_cutoff: int,
) -> pd.DataFrame:
    """
    Compare model performance across governance specifications.

    Scaling fix: fits StandardScaler on the training rows (year < cutoff)
    and applies it to test rows before calling model.predict(). This
    replicates exactly what happened inside the walk-forward folds.
    """
    os.makedirs(out_dir, exist_ok=True)
    rows       = []
    pred_store = {}

    for spec_name, df in spec_datasets.items():
        feat_cols = [
            c for c in df.select_dtypes(include=[np.number]).columns
            if c not in {"year", var.TARGET, "__y_h__"} and "country" not in c.lower()
        ]
        if not feat_cols:
            continue

        # Split train / test
        mask_tr = df["year"] <  test_year_cutoff
        mask_te = df["year"] >= test_year_cutoff

        # CORRECTION (versão_5): use the horizon-shifted label attached by
        # pipeline.py::_build_spec_datasets ("__y_h__") — the loaded models
        # (modelo_{spec}_{model}.pkl, unsuffixed) were trained against the
        # primary forecast horizon, not the contemporaneous var.TARGET;
        # scoring against the wrong quantity would silently invalidate
        # every RMSE/R2 this function reports.
        y_col = "__y_h__" if "__y_h__" in df.columns else var.TARGET

        # CORRECTION (h=5 ablation crash — root cause fix): rows whose
        # shifted target falls beyond the last available year have a NaN
        # y_col — there is no genuine future value to compare against.
        # walk_forward.py already excludes these rows from every fold
        # ("linhas descartadas: alvo em t+H indisponível"); ablation.py
        # used a fixed test_year_cutoff shared across all horizons and did
        # NOT apply the same exclusion, so at large horizons (fewer years
        # left after the +h shift) the test window could end up 100% NaN
        # target for a given spec — every model's RMSE became NaN, and the
        # resulting all-NaN heatmap crashed seaborn's vmin/vmax computation
        # (np.nanmin on a zero-size array after dropping NaNs). Applying the
        # same "no genuine label → not evaluated" rule here keeps ablation
        # consistent with walk-forward and prevents silently scoring against
        # missing data.
        n_te_before  = int(mask_te.sum())
        mask_te      = mask_te & df[y_col].notna()
        n_dropped_te = n_te_before - int(mask_te.sum())

        X_tr = df.loc[mask_tr, feat_cols].fillna(0).values
        X_te = df.loc[mask_te, feat_cols].fillna(0).values
        y_te  = df.loc[mask_te, y_col].values
        cc   = (df.loc[mask_te, "country_code"].values
                if "country_code" in df.columns else np.zeros(len(y_te)))

        if len(X_tr) == 0 or len(X_te) == 0:
            if n_dropped_te:
                print(f"    {spec_name}: sem linhas de teste com alvo genuíno "
                      f"disponível nesta janela ({n_dropped_te} linhas "
                      f"descartadas: alvo em t+h indisponível) — spec ignorada "
                      f"neste horizonte.")
            continue

        # Fit scaler on TRAIN, apply to TEST — same as walk-forward
        scaler   = StandardScaler()
        scaler.fit(X_tr)
        X_te_s   = scaler.transform(X_te)

        for mod_name in model_names:
            pkl = os.path.join(model_dir, f"modelo_{spec_name}_{mod_name}.pkl")
            if not os.path.exists(pkl):
                continue
            try:
                # CORRECTION: models are now persisted as {"model","scaler","feat_cols"}
                # bundles (utils/model_io.py) — unwrap the model object. This module
                # already fits its own train/test-consistent scaler above (see
                # docstring), so the persisted scaler is not needed here.
                model, _persisted_scaler, _persisted_cols = load_model_bundle(pkl)
                y_pred = np.asarray(model.predict(X_te_s), dtype=float)

                # Align lengths (LSTM lookback may reduce output)
                min_len = min(len(y_te), len(y_pred))
                yt = y_te[-min_len:]
                yp = y_pred[-min_len:]

                rmse = _rmse(yt, yp)
                r2   = _r2(yt, yp)

                rows.append({
                    "Specification": spec_name,
                    "Model":         mod_name,
                    "RMSE":          rmse,
                    "R2":            r2,
                    "N_test":        min_len,
                })
                pred_store[(spec_name, mod_name)] = (yt, yp, cc[-min_len:])

            except Exception as exc:
                print(f"    {spec_name}/{mod_name}: {exc}")

    df_abl = pd.DataFrame(rows)

    if df_abl.empty:
        print("  No results — check model files and spec names.")
        return df_abl

    # ── DM tests vs WDI baseline ──────────────────────────────────────────────
    dm_rows      = []
    # CORRECTION (root-cause diagnostic report, Secção 10.4 / recomendação #6):
    # the baseline used to be implicit ("first key of the dict"), which happened
    # to always be A1_WDI_only only because of ABLATION_SPECS's current
    # insertion order — silently fragile to any future reordering. Now explicit,
    # falling back to the first dict key only as a defensive last resort.
    baseline_spec = "A1_WDI_only" if "A1_WDI_only" in spec_datasets else next(iter(spec_datasets))

    for mod_name in model_names:
        key_base = (baseline_spec, mod_name)
        if key_base not in pred_store:
            continue
        yt_b, yp_b, cc_b = pred_store[key_base]
        e_base = yt_b - yp_b

        for spec_name in spec_datasets:
            if spec_name == baseline_spec:
                continue
            key_alt = (spec_name, mod_name)
            if key_alt not in pred_store:
                continue
            _, yp_a, _ = pred_store[key_alt]
            e_alt = yt_b - yp_a

            if len(e_base) != len(e_alt):
                continue

            dm = _dm_cluster_bootstrap(e_base, e_alt, cc_b)
            try:
                _, wil_p = stats.wilcoxon(np.abs(e_base), np.abs(e_alt),
                                          alternative="greater", zero_method="zsplit")
            except Exception:
                wil_p = np.nan

            dm_rows.append({
                "Baseline":    baseline_spec,
                "Alternative": spec_name,
                "Model":       mod_name,
                "DM_stat":     dm["dm_stat"],
                "DM_p_value":  dm["p_value"],
                "Wilcoxon_p":  wil_p,
                "Significant": "Yes" if dm["p_value"] < 0.05 else "No",
            })

    df_dm = pd.DataFrame(dm_rows)

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    abl_path = os.path.join(out_dir, "ablation_results.csv")
    dm_path  = os.path.join(out_dir, "ablation_dm_tests.csv")
    df_abl.to_csv(abl_path, index=False)
    df_dm.to_csv(dm_path,   index=False)
    print(f"  Saved: {abl_path}")
    print(f"  Saved: {dm_path}")

    # ── Charts ────────────────────────────────────────────────────────────────

    # 1. RMSE heatmap
    pivot = df_abl.pivot_table(values="RMSE", index="Specification",
                                columns="Model", aggfunc="mean")
    # DEFENSIVE GUARD (belt-and-suspenders alongside the NaN-target filter
    # above): if every cell is still NaN — e.g. a model×spec combination
    # with fewer than 3 valid test points survives the filter but _rmse's
    # own m.sum()>=3 guard returns NaN — seaborn's vmin/vmax computation
    # (np.nanmin) crashes on a zero-size array after dropping NaNs. Skip
    # the figure with a clear log message instead of crashing the run.
    if pivot.notna().to_numpy().any():
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn_r",
                    ax=ax, linewidths=0.5)
        ax.set_title("Ablation: RMSE by governance specification × model\n"
                     "(lower = better; first row = no-governance baseline)")
        plt.tight_layout()
        p = os.path.join(out_dir, "ablation_rmse_heatmap.png")
        plt.savefig(p, dpi=130); plt.close()
        print(f"  Saved: {p}")
    else:
        print("  RMSE heatmap ignorado: nenhum valor de RMSE válido neste "
              "horizonte (dados de teste insuficientes após o filtro de "
              "alvo genuíno).")

    # 2. RMSE gain vs baseline
    if baseline_spec in df_abl["Specification"].values:
        base_rmse = (df_abl[df_abl["Specification"] == baseline_spec]
                     .set_index("Model")["RMSE"])
        gain_rows = []
        for spec in df_abl["Specification"].unique():
            if spec == baseline_spec:
                continue
            spec_rmse = df_abl[df_abl["Specification"] == spec].set_index("Model")["RMSE"]
            for mod in base_rmse.index.intersection(spec_rmse.index):
                gain_rows.append({
                    "Specification": spec, "Model": mod,
                    "RMSE_gain_pct": (base_rmse[mod] - spec_rmse[mod]) / base_rmse[mod] * 100
                })
        if gain_rows:
            df_gain = pd.DataFrame(gain_rows)
            fig, ax = plt.subplots(figsize=(12, 5))
            for spec in df_gain["Specification"].unique():
                sub = df_gain[df_gain["Specification"] == spec]
                ax.plot(sub["Model"], sub["RMSE_gain_pct"], "o-",
                        label=spec, markersize=7)
            ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_title("RMSE gain (%) vs WDI-only baseline\n"
                         "(positive = governance improves prediction)")
            ax.set_ylabel("% RMSE reduction")
            ax.legend(fontsize=9)
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            p = os.path.join(out_dir, "ablation_rmse_gain.png")
            plt.savefig(p, dpi=130); plt.close()
            print(f"  Saved: {p}")

    # 3. DM test heatmap
    if not df_dm.empty:
        for mod in df_dm["Model"].unique():
            sub = df_dm[df_dm["Model"] == mod]
            pivot_dm = sub.pivot(index="Alternative", columns="Baseline",
                                 values="DM_p_value")
            if pivot_dm.empty or not pivot_dm.notna().to_numpy().any():
                continue
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.heatmap(pivot_dm, annot=True, fmt=".3f", cmap="RdYlGn_r",
                        vmin=0, vmax=0.10, ax=ax, linewidths=0.4)
            ax.set_title(f"DM test p-values vs baseline — {mod}\n"
                         "(<0.05 = governance significantly improves accuracy)")
            plt.tight_layout()
            p = os.path.join(out_dir, f"ablation_dm_{mod}.png")
            plt.savefig(p, dpi=120); plt.close()

    n_sig = len(df_dm[df_dm["Significant"] == "Yes"]) if not df_dm.empty else 0
    print(f"\n  Ablation: {len(df_abl)} model×spec combinations")
    print(f"  DM tests significant at 5%: {n_sig}/{len(df_dm)}")

    return df_abl
