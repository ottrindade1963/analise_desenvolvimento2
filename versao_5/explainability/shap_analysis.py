"""explainability/shap_analysis.py — SHAP analysis with all recommended plot types.

Addresses Problem 6 from the review: this module produces:
  - SHAP summary bar plot (feature importance ranking)
  - SHAP beeswarm plot (distribution of SHAP values per feature)
  - SHAP waterfall plots (individual prediction explanation)
  - SHAP dependence plots (feature interaction effects)
  - WGI vs WDI contribution pie chart

And the ablation comparison answering the research hypothesis:
  "Do WGI indicators improve industrial value-added forecasting?"
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

import config.paths    as paths
import config.variables as var


def _load_shap():
    try:
        import shap
        return shap
    except ImportError:
        raise ImportError("pip install shap")


# ── SHAP for tree models ───────────────────────────────────────────────────────

def shap_tree_analysis(model, X: pd.DataFrame,
                       label: str, out_dir: str) -> dict:
    """
    Full SHAP analysis for tree-based models (RF, XGBoost, GBM).

    Produces:
      1. Summary bar (importance ranking)
      2. Beeswarm (value distribution per feature)
      3. Waterfall (first 3 observations)
      4. Dependence plot (top governance feature vs top economic feature)
    """
    shap = _load_shap()
    os.makedirs(out_dir, exist_ok=True)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    ev          = explainer.expected_value

    importance = pd.DataFrame({
        "Feature":       X.columns,
        "SHAP_Mean_Abs": np.abs(shap_values).mean(axis=0),
    }).sort_values("SHAP_Mean_Abs", ascending=False).reset_index(drop=True)

    # 1. Summary bar
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X, plot_type="bar",
                      max_display=20, show=False)
    plt.title(f"SHAP feature importance — {label}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{label}_shap_bar.png"), dpi=130, bbox_inches="tight")
    plt.close()

    # 2. Beeswarm
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X, max_display=15, show=False)
    plt.title(f"SHAP beeswarm — {label}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{label}_shap_beeswarm.png"), dpi=130, bbox_inches="tight")
    plt.close()

    # 3. Waterfall plots (3 individual observations)
    try:
        expl = shap.Explanation(
            values=shap_values[:3],
            base_values=np.full(3, ev),
            data=X.values[:3],
            feature_names=list(X.columns),
        )
        for i in range(min(3, len(X))):
            plt.figure(figsize=(9, 6))
            shap.plots.waterfall(expl[i], show=False)
            plt.title(f"SHAP waterfall — {label} — obs {i+1}")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{label}_waterfall_{i+1}.png"),
                        dpi=120, bbox_inches="tight")
            plt.close()
    except Exception as exc:
        print(f"    Waterfall failed: {exc}")

    # 4. Dependence plot (top governance feature)
    gov_cols = [c for c in importance["Feature"] if "wgi" in c.lower() or "pca" in c.lower()]
    eco_cols = [c for c in importance["Feature"] if c not in gov_cols and c in X.columns]
    if gov_cols and eco_cols:
        top_gov = gov_cols[0]
        top_eco = eco_cols[0]
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(top_gov, shap_values, X,
                             interaction_index=top_eco, show=False)
        plt.title(f"SHAP dependence: {top_gov} × {top_eco} — {label}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{label}_dependence.png"), dpi=120, bbox_inches="tight")
        plt.close()

    # 5. WGI vs WDI contribution
    gov_mask = importance["Feature"].str.contains("wgi|pca|inter_pca", case=False)
    gov_shap = importance.loc[gov_mask,  "SHAP_Mean_Abs"].sum()
    eco_shap = importance.loc[~gov_mask, "SHAP_Mean_Abs"].sum()
    total    = gov_shap + eco_shap + 1e-10
    gov_pct  = gov_shap / total * 100

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        [gov_shap, eco_shap],
        labels=[f"Governance\n{gov_pct:.1f}%",
                f"Economic\n{100 - gov_pct:.1f}%"],
        colors=["#ff6f00", "#1976d2"],
        startangle=90,
    )
    ax.set_title(f"SHAP: governance vs economic contribution\n{label}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{label}_gov_vs_eco.png"), dpi=120)
    plt.close()

    importance.to_csv(os.path.join(out_dir, f"{label}_shap_importance.csv"), index=False)

    return {
        "shap_values":  shap_values,
        "importance":   importance,
        "gov_pct":      gov_pct,
        "method":       "TreeExplainer",
    }


# ── SHAP Kernel for non-tree models ───────────────────────────────────────────

def shap_kernel_analysis(model, X_bg: pd.DataFrame,
                         X_explain: pd.DataFrame,
                         label: str, out_dir: str) -> dict:
    shap = _load_shap()
    os.makedirs(out_dir, exist_ok=True)

    n_bg = min(50, len(X_bg))
    bg   = shap.kmeans(X_bg.values[:n_bg], min(10, n_bg))
    Xe   = X_explain.iloc[:min(100, len(X_explain))]

    try:
        explainer   = shap.KernelExplainer(model.predict, bg)
        shap_values = explainer.shap_values(Xe.values, nsamples=100)

        importance = pd.DataFrame({
            "Feature":       Xe.columns,
            "SHAP_Mean_Abs": np.abs(shap_values).mean(axis=0),
        }).sort_values("SHAP_Mean_Abs", ascending=False)

        importance.to_csv(os.path.join(out_dir, f"{label}_shap_importance.csv"), index=False)

        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_values, Xe, plot_type="bar", show=False)
        plt.title(f"SHAP KernelExplainer — {label}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{label}_shap_bar.png"), dpi=120, bbox_inches="tight")
        plt.close()

        return {"shap_values": shap_values, "importance": importance, "method": "KernelExplainer"}
    except Exception as exc:
        print(f"    KernelExplainer failed ({label}): {exc}")
        return {}
