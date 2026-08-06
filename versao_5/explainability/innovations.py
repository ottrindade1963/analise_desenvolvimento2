"""explainability/innovations.py — Three methodological innovations.

1. Structural break detection (PELT algorithm via ruptures, CUSUM fallback)
   → CSV: structural_breaks.csv, regimes_by_country.csv
   → PNG: structural_breaks_histogram.png

2. Localised counterfactual simulation (ceteris paribus WGI perturbation)
   → CSV: counterfactual_results.csv
   → PNG: counterfactual_dose_response.png

3. Adaptive Conformal Inference (ACI) — sensitivity grid gamma × window
   → CSV: aci_sensitivity.csv
   → PNG: aci_sensitivity_coverage.png, aci_sensitivity_interval_width.png
"""
import os
import pickle
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

import config.paths       as paths
import config.variables   as var
import config.model_params as mp
import config.pipeline    as cfg_pipe
from utils.model_io import load_model_bundle

OUT_DIR = os.path.join(paths.EXPLAINABILITY_DIR, "innovations")


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_model(spec: str, model_name: str):
    """Returns (model, scaler_or_None, feat_cols_or_None), or None if missing."""
    path = os.path.join(paths.MODELS_DIR, f"modelo_{spec}_{model_name}.pkl")
    if not os.path.exists(path):
        return None
    try:
        return load_model_bundle(path)
    except Exception:
        return None


def _feat_cols(df: pd.DataFrame) -> list:
    excl = {"country_code", "year", var.TARGET, "pais"}
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in excl]


# ── Innovation 1: Structural breaks ───────────────────────────────────────────

def run_structural_breaks(df: pd.DataFrame) -> tuple:
    """
    Detect structural breaks in the industrial value-added series per country.

    Algorithm: PELT (Pruned Exact Linear Time) from the ruptures library,
    with L2 cost function and penalty calibrated to series length.
    Fallback: CUSUM-based detection when ruptures is not installed.

    Returns two DataFrames:
      df_breaks  — one row per detected break (country, year, magnitude)
      df_regimes — panel with regime classification (CRISIS/STABLE/EXPANSION)
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    target = var.TARGET

    try:
        import ruptures as rpt
        use_pelt = True
        print("  ruptures disponível — usando PELT")
    except ImportError:
        use_pelt = False
        print("  ruptures não instalado — usando CUSUM (pip install ruptures)")

    rows_breaks, rows_regimes = [], []

    for pais in sorted(df["country_code"].unique()):
        sub   = df[df["country_code"] == pais].sort_values("year")
        serie = sub[target].dropna().values
        anos  = sub.loc[sub[target].notna(), "year"].values

        if len(serie) < 8:
            continue

        # Detect break points
        bps = []
        if use_pelt:
            try:
                pen  = np.log(len(serie)) * np.var(serie) * (0.5 if len(serie) < 30 else 2.0)
                algo = rpt.Pelt(model="l2", min_size=3, jump=1).fit(serie)
                bps  = [b for b in algo.predict(pen=pen) if b < len(serie)]
                if not bps:
                    algo2 = rpt.Binseg(model="l2", min_size=3, jump=1).fit(serie)
                    bps   = [b for b in algo2.predict(n_bkps=min(2, max(1, len(serie)//8)))
                             if b < len(serie)]
            except Exception:
                bps = []
        else:
            mu = np.mean(serie)
            for i in range(2, len(serie) - 2):
                se = np.sqrt(np.var(serie[:i]) / i + np.var(serie[i:]) / (len(serie) - i))
                if se > 0 and abs(np.mean(serie[i:]) - np.mean(serie[:i])) / se > 2.0:
                    bps.append(i)
            bps = sorted(bps)[:3]

        anos_bp = [int(anos[b]) for b in bps if b < len(anos)]

        for idx, (b, ano_q) in enumerate(zip(bps, anos_bp)):
            m_before = np.mean(serie[max(0, b - 3): b])
            m_after  = np.mean(serie[b: min(len(serie), b + 3)])
            rows_breaks.append({
                "Country":      pais,
                "Country_Name": var.COUNTRIES.get(pais, pais),
                "Break_Num":    idx + 1,
                "Year":         ano_q,
                "Mean_Before":  round(m_before, 3),
                "Mean_After":   round(m_after,  3),
                "Magnitude":    round(m_after - m_before, 3),
                "Method":       "PELT" if use_pelt else "CUSUM",
            })

        # Regime classification by segment
        segments = [0] + bps + [len(serie)]
        mu_g, sd_g = np.mean(serie), np.std(serie)

        for s in range(len(segments) - 1):
            seg  = serie[segments[s]: segments[s + 1]]
            mu_s = np.mean(seg)
            regime = ("CRISIS"    if mu_s < mu_g - 0.5 * sd_g else
                      "EXPANSION" if mu_s > mu_g + 0.5 * sd_g else
                      "STABLE")
            for idx in range(segments[s], min(segments[s + 1], len(anos))):
                rows_regimes.append({
                    "Country":       pais,
                    "Country_Name":  var.COUNTRIES.get(pais, pais),
                    "Year":          int(anos[idx]),
                    "Regime":        regime,
                    "Segment_Mean":  round(mu_s, 3),
                })

    df_breaks  = pd.DataFrame(rows_breaks)
    df_regimes = pd.DataFrame(rows_regimes)

    df_breaks.to_csv( os.path.join(OUT_DIR, "structural_breaks.csv"),   index=False)
    df_regimes.to_csv(os.path.join(OUT_DIR, "regimes_by_country.csv"),  index=False)

    if not df_breaks.empty:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram of break years
        df_breaks["Year"].hist(
            bins=range(int(df_breaks["Year"].min()),
                       int(df_breaks["Year"].max()) + 2),
            ax=axes[0], color="steelblue", edgecolor="white"
        )
        axes[0].set_title("Distribution of Structural Breaks by Year")
        axes[0].set_xlabel("Year"); axes[0].set_ylabel("Number of breaks")
        for yr, label in [(2008, "GFC"), (2014, "Commodities"), (2020, "COVID")]:
            axes[0].axvline(yr, color="red", linestyle="--", alpha=0.6)
            axes[0].text(yr + 0.2, axes[0].get_ylim()[1] * 0.85,
                         label, fontsize=9, color="red")

        # Regime distribution
        if not df_regimes.empty:
            regime_counts = df_regimes["Regime"].value_counts()
            colors = {"CRISIS": "#c62828", "STABLE": "#1976d2", "EXPANSION": "#2e7d32"}
            axes[1].bar(regime_counts.index,
                        regime_counts.values,
                        color=[colors.get(r, "grey") for r in regime_counts.index])
            axes[1].set_title("Regime Distribution across Countries & Years")
            axes[1].set_ylabel("Observations (country × year)")

        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "structural_breaks_histogram.png"), dpi=130)
        plt.close()

    print(f"  Breaks detected: {len(df_breaks)}")
    print(f"  Regimes classified: {len(df_regimes)} country×year observations")
    return df_breaks, df_regimes


# ── Innovation 2: Counterfactual simulation ───────────────────────────────────

def run_counterfactual(df: pd.DataFrame,
                       spec: str = "A2_WDI_PCA1") -> pd.DataFrame:
    """
    Localised counterfactual: ceteris paribus WGI perturbation.

    For each country's last available observation, perturbs each
    WGI-related feature by ±0.5 and ±1.0 standard deviations while
    holding all other features constant (ceteris paribus), then records
    the predicted change in industrial value added.

    This reveals: "if governance had been better/worse by X std devs,
    what would the model predict for industrial value added?"
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load best tree model for this spec
    loaded = (_load_model(spec, "RandomForest") or
              _load_model(spec, "XGBoost"))
    if loaded is None:
        print(f"  No model found for {spec} — skipping counterfactual.")
        return pd.DataFrame()
    model, scaler, trained_feat_cols = loaded

    # CORRECTION (root-cause diagnostic report, Secção 6, achado nº5 /
    # recomendação #3): use the exact column set/order the model was trained
    # on, and scale every vector with the persisted scaler before predict() —
    # previously raw, unscaled feature values were fed straight into a model
    # trained on standardized data, collapsing Y_Base into a narrow band.
    feat_cols = [c for c in (trained_feat_cols or _feat_cols(df)) if c in df.columns]
    if scaler is None:
        print(f"    [aviso] {spec}: pickle sem scaler persistido (formato anterior à "
              f"correcção) — contrafactual usa dados brutos, tal como antes da correcção.")

    def _predict(X_raw_row):
        X_in = scaler.transform(X_raw_row) if scaler is not None else X_raw_row
        return float(model.predict(X_in)[0])

    wgi_feat  = [c for c in feat_cols
                 if "wgi" in c.lower() or "pca" in c.lower() or "inter_pca" in c.lower()]
    magnitudes = [-1.0, -0.5, 0.0, 0.5, 1.0]

    rows = []
    for pais in sorted(df["country_code"].unique()):
        sub      = df[df["country_code"] == pais].sort_values("year")
        if sub.empty or var.TARGET not in sub.columns:
            continue
        last_obs = sub.iloc[-1]
        X_base   = np.nan_to_num(
            last_obs[feat_cols].values.astype(float)
        ).reshape(1, -1)
        try:
            y_base = _predict(X_base)
        except Exception:
            continue

        for wgi_col in wgi_feat[:4]:          # limit to 4 governance features
            if wgi_col not in feat_cols:
                continue
            idx     = feat_cols.index(wgi_col)
            std_wgi = df[wgi_col].std() if df[wgi_col].std() > 0 else 1.0

            for mag in magnitudes:
                X_cf = X_base.copy()
                X_cf[0, idx] = X_base[0, idx] + mag * std_wgi
                try:
                    y_cf = _predict(X_cf)
                except Exception:
                    continue
                rows.append({
                    "Country":        pais,
                    "Country_Name":   var.COUNTRIES.get(pais, pais),
                    "WGI_Feature":    wgi_col,
                    "Magnitude_Std":  mag,
                    "Y_Base":         round(y_base, 4),
                    "Y_Counterfactual": round(y_cf, 4),
                    "Delta_Abs":      round(y_cf - y_base, 4),
                    "Delta_Pct":      round((y_cf - y_base) / (abs(y_base) + 1e-10) * 100, 2),
                })

    df_cf = pd.DataFrame(rows)
    if df_cf.empty:
        return df_cf

    df_cf.to_csv(os.path.join(OUT_DIR, "counterfactual_results.csv"), index=False)

    # Dose-response chart per WGI feature
    for wgi_col in df_cf["WGI_Feature"].unique():
        sub_plot = df_cf[df_cf["WGI_Feature"] == wgi_col]
        paises_plot = list(sub_plot["Country"].unique())[:8]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: predicted values
        for pais in paises_plot:
            p = sub_plot[sub_plot["Country"] == pais].sort_values("Magnitude_Std")
            axes[0].plot(p["Magnitude_Std"], p["Y_Counterfactual"], "o-",
                         label=var.COUNTRIES.get(pais, pais), markersize=5)
        axes[0].axvline(0, color="gray", linestyle="--", alpha=0.5)
        axes[0].set_xlabel("Perturbation (std devs)")
        axes[0].set_ylabel("Predicted industrial VA (% GDP)")
        axes[0].set_title(f"Dose-response: {wgi_col}")
        axes[0].legend(fontsize=7, ncol=2)

        # Right: delta_pct distribution at +1 std
        delta_1std = sub_plot[sub_plot["Magnitude_Std"] == 1.0].copy()
        delta_1std = delta_1std.sort_values("Delta_Pct")
        colors = ["#2e7d32" if v > 0 else "#c62828" for v in delta_1std["Delta_Pct"]]
        axes[1].barh(delta_1std["Country_Name"], delta_1std["Delta_Pct"],
                     color=colors, alpha=0.8)
        axes[1].axvline(0, color="black", linewidth=0.8)
        axes[1].set_xlabel("Predicted change in industrial VA (%) at +1 std governance")
        axes[1].set_title(f"Counterfactual effect: {wgi_col} +1 std")

        plt.tight_layout()
        safe_name = wgi_col.replace("/", "_").replace(" ", "_")
        plt.savefig(os.path.join(OUT_DIR,
                    f"counterfactual_dose_response_{safe_name}.png"), dpi=130)
        plt.close()

    print(f"  Counterfactual simulations: {len(df_cf)}")
    return df_cf


# ── Innovation 3: ACI — Adaptive Conformal Inference ─────────────────────────

def _aci_single(y_cal, y_cal_pred, y_test, y_test_pred,
                gamma: float, window: int, alpha: float = 0.10) -> dict:
    """
    One ACI run for a given (gamma, window, alpha) combination.

    The ACI algorithm adaptively updates the quantile of non-conformity
    scores at each test step, targeting empirical coverage of (1-alpha).

    Returns: coverage (%), mean interval width.
    """
    residuos = list(np.abs(y_cal - y_cal_pred))
    q_hat    = np.quantile(residuos, 1 - alpha)
    lower_l, upper_l, covered = [], [], []

    for i in range(len(y_test)):
        lo = y_test_pred[i] - q_hat
        hi = y_test_pred[i] + q_hat
        lower_l.append(lo); upper_l.append(hi)
        inside = bool(lo <= y_test[i] <= hi)
        covered.append(inside)

        residuos.append(abs(y_test[i] - y_test_pred[i]))
        if len(residuos) > window:
            residuos = residuos[-window:]

        cov_acc = np.mean(covered)
        alpha_t = np.clip(alpha - gamma * (cov_acc - (1 - alpha)), 0.01, 0.99)
        q_hat   = np.quantile(residuos, 1 - alpha_t)

    return {
        "coverage":       round(float(np.mean(covered) * 100), 2),
        "interval_width": round(float(np.mean(
            np.array(upper_l) - np.array(lower_l))), 4),
    }


def run_aci_sensitivity(df: pd.DataFrame,
                        spec: str = "A2_WDI_PCA1") -> pd.DataFrame:
    """
    ACI sensitivity analysis over gamma × window grid.

    Exports heatmaps of:
      - Empirical coverage (%) — target: 90%
      - Mean interval width

    Gamma controls how fast the quantile adapts to miscoverage.
    Window controls the size of the calibration set.
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    loaded = (_load_model(spec, "RandomForest") or
              _load_model(spec, "XGBoost"))
    if loaded is None:
        print(f"  No model found for {spec} — skipping ACI.")
        return pd.DataFrame()
    model, scaler, trained_feat_cols = loaded

    # CORRECTION (root-cause diagnostic report, Secção 7, achado nº6 /
    # recomendação #3): use the model's own trained column set/order and
    # apply the persisted scaler before predict() — previously X_cal/X_te
    # were raw, unscaled values fed to a model trained on standardized data.
    feat_cols = [c for c in (trained_feat_cols or _feat_cols(df)) if c in df.columns]
    if scaler is None:
        print(f"    [aviso] {spec}: pickle sem scaler persistido (formato anterior à "
              f"correcção) — ACI usa dados brutos, tal como antes da correcção.")
    if var.TARGET not in df.columns or not feat_cols:
        print("  Insufficient data — skipping ACI.")
        return pd.DataFrame()

    df_s = df.sort_values(["country_code", "year"])
    n    = len(df_s)
    n_tr = int(n * 0.60); n_cal = int(n * 0.20)

    X_tr  = df_s.iloc[:n_tr][feat_cols].fillna(0).values
    X_cal = df_s.iloc[n_tr: n_tr + n_cal][feat_cols].fillna(0).values
    y_cal = df_s.iloc[n_tr: n_tr + n_cal][var.TARGET].fillna(0).values
    X_te  = df_s.iloc[n_tr + n_cal:][feat_cols].fillna(0).values
    y_te  = df_s.iloc[n_tr + n_cal:][var.TARGET].fillna(0).values

    if len(X_te) < 5:
        print("  Test set too small — skipping ACI.")
        return pd.DataFrame()

    if scaler is not None:
        X_cal_in = scaler.transform(X_cal)
        X_te_in  = scaler.transform(X_te)
    else:
        X_cal_in, X_te_in = X_cal, X_te

    try:
        y_cal_pred  = model.predict(X_cal_in).ravel()
        y_test_pred = model.predict(X_te_in).ravel()
    except Exception as exc:
        print(f"  Prediction failed: {exc}")
        return pd.DataFrame()

    cfg          = mp.ACI
    gamma_grid   = cfg["gamma_grid"]
    window_grid  = cfg["window_grid"]
    alpha        = cfg["alpha"]

    rows = []
    for g in gamma_grid:
        for w in window_grid:
            r = _aci_single(y_cal, y_cal_pred, y_te, y_test_pred,
                            gamma=g, window=w, alpha=alpha)
            rows.append({"gamma": g, "window": w, **r})
            print(f"    γ={g:<5}  w={w:<4}  coverage={r['coverage']:.1f}%  "
                  f"width={r['interval_width']:.3f}")

    df_aci = pd.DataFrame(rows)
    df_aci.to_csv(os.path.join(OUT_DIR, "aci_sensitivity.csv"), index=False)

    # Heatmaps
    for metric, title, fmt, cmap in [
        ("coverage",       f"Empirical Coverage (%) — target: {(1-alpha)*100:.0f}%",
         ".1f", "RdYlGn"),
        ("interval_width", "Mean Prediction Interval Width (pp GDP)",
         ".3f", "RdYlGn_r"),
    ]:
        pivot = df_aci.pivot(index="gamma", columns="window", values=metric)
        fig, ax = plt.subplots(figsize=(9, 5))
        vmin = (80 if metric == "coverage" else None)
        vmax = (100 if metric == "coverage" else None)
        sns.heatmap(pivot, annot=True, fmt=fmt, cmap=cmap,
                    ax=ax, linewidths=0.5, vmin=vmin, vmax=vmax)
        ax.set_title(f"ACI Sensitivity: {title}\n"
                     f"Spec={spec}  α={alpha}")
        ax.set_xlabel("Window size (calibration years)")
        ax.set_ylabel("Gamma (adaptation rate)")
        plt.tight_layout()
        fname = f"aci_sensitivity_{metric}.png"
        plt.savefig(os.path.join(OUT_DIR, fname), dpi=130)
        plt.close()
        print(f"  Saved: {fname}")

    # Default ACI result
    default_r = _aci_single(y_cal, y_cal_pred, y_te, y_test_pred,
                             gamma=cfg["default_gamma"],
                             window=cfg["default_window"],
                             alpha=alpha)
    print(f"\n  Default ACI (γ={cfg['default_gamma']}, w={cfg['default_window']}): "
          f"coverage={default_r['coverage']:.1f}%  "
          f"width={default_r['interval_width']:.3f}")

    return df_aci


# ── Entry point ───────────────────────────────────────────────────────────────

def run_all_innovations(df: pd.DataFrame,
                        spec: str = "A2_WDI_PCA1") -> dict:
    """
    Run all three innovations and return results dict.

    Internally applies FoldFeatureEngineer so the model receives the
    same feature-engineered input it was trained on (53 features).
    Innovation 1 uses raw df — breaks are on the target series directly.
    Innovations 2 and 3 use the feature-engineered dataset.
    """
    import shutil
    from features.engineer import FoldFeatureEngineer

    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    results = {}

    # Build feature-engineered dataset for innovations 2 & 3
    #
    # CORRECTION (found in a later review pass, same family as the
    # imputation look-ahead fix): this used to call fe.fit(df) on the
    # ENTIRE panel, including the years that run_counterfactual()/
    # run_aci_sensitivity() later treat as "test" via their own
    # train/cal/test row split \u2014 so the PCA governance factor and lag
    # structure were partly learned from data those two functions treat as
    # future/held-out. Fit only on the same pre-holdout training years that
    # pipeline.py._build_spec_datasets() and _train_and_evaluate() use, then
    # transform the full panel \u2014 consistent, non-leaky feature construction.
    print(f"  A construir features para spec={spec}...")
    years      = sorted(df["year"].unique())
    split_idx  = int(len(years) * (1 - cfg_pipe.FINAL_HOLDOUT_RATIO))
    train_yr   = years[:split_idx]

    fe = FoldFeatureEngineer(spec=spec)
    fe.fit(df[df["year"].isin(train_yr)])
    df_feat = fe.transform(df)
    print(f"  \u2713 Dataset com features: {df_feat.shape}")

    print("\n" + "=" * 60)
    print("  INNOVATION 1: STRUCTURAL BREAKS + REGIMES")
    print("=" * 60)
    df_breaks, df_regimes = run_structural_breaks(df)
    results["breaks"]  = df_breaks
    results["regimes"] = df_regimes

    print("\n" + "=" * 60)
    print("  INNOVATION 2: COUNTERFACTUAL SIMULATION")
    print("=" * 60)
    df_cf = run_counterfactual(df_feat, spec=spec)
    results["counterfactual"] = df_cf

    print("\n" + "=" * 60)
    print("  INNOVATION 3: ACI SENSITIVITY ANALYSIS")
    print("=" * 60)
    df_aci = run_aci_sensitivity(df_feat, spec=spec)
    results["aci"] = df_aci

    # Backup to Drive
    if paths.DRIVE_DIR:
        drive_innov = os.path.join(paths.DRIVE_DIR,
                                   "explainability", "innovations")
        shutil.copytree(OUT_DIR, drive_innov, dirs_exist_ok=True)
        print(f"\n✓ Drive backup: {drive_innov}")

    print(f"\n✓ All innovations complete ({time.time()-t0:.1f}s)")
    print(f"  Output: {OUT_DIR}")
    for f in sorted(os.listdir(OUT_DIR)):
        size = os.path.getsize(os.path.join(OUT_DIR, f)) / 1024
        print(f"  {f}  ({size:.0f} KB)")

    return results
