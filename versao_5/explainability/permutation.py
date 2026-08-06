"""explainability/permutation.py — Temporal permutation importance."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def permutation_importance(model, X_test: pd.DataFrame,
                           y_test: np.ndarray, label: str,
                           out_dir: str, n_repeats: int = 20) -> pd.DataFrame:
    """
    Temporal permutation importance.

    Features are shuffled independently within the test window only.
    RMSE increase after shuffling measures reliance on each feature.
    Governance vs economic contributions are highlighted separately.
    """
    os.makedirs(out_dir, exist_ok=True)
    Xv = X_test.values.copy()
    yv = np.asarray(y_test, float)

    ok = ~np.isnan(yv)
    baseline = np.sqrt(np.mean((yv[ok] - model.predict(Xv)[ok]) ** 2))

    rows = []
    for i, feat in enumerate(X_test.columns):
        deltas = []
        for _ in range(n_repeats):
            X_perm       = Xv.copy()
            X_perm[:, i] = np.random.permutation(Xv[:, i])
            preds        = model.predict(X_perm)
            rmse_perm    = np.sqrt(np.mean((yv[ok] - preds[ok]) ** 2))
            deltas.append(rmse_perm - baseline)
        rows.append({
            "Feature":            feat,
            "RMSE_increase_mean": float(np.mean(deltas)),
            "RMSE_increase_std":  float(np.std(deltas)),
            "Is_Governance":      bool("wgi" in feat.lower() or "pca" in feat.lower()),
        })

    df = pd.DataFrame(rows).sort_values("RMSE_increase_mean", ascending=False)
    df.to_csv(os.path.join(out_dir, f"{label}_permutation_importance.csv"), index=False)

    # Chart
    top = df.head(20)
    colors = ["#ff6f00" if g else "#1976d2" for g in top["Is_Governance"]]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["Feature"][::-1], top["RMSE_increase_mean"][::-1],
            color=colors[::-1], alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean RMSE increase after permutation")
    ax.set_title(f"Temporal permutation importance — {label}\n"
                 "(orange = governance  ·  blue = economic)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{label}_permutation_importance.png"), dpi=120)
    plt.close()

    return df
