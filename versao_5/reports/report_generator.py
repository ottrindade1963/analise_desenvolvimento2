"""reports/report_generator.py — Automatic dissertation-ready report generation."""
import os
import datetime
import numpy as np
import pandas as pd
import config.paths    as paths
import config.pipeline as cfg


def _to_latex(df: pd.DataFrame, caption: str, label: str,
              float_fmt: str = "%.3f") -> str:
    return df.to_latex(
        index=False, caption=caption, label=label,
        float_format=float_fmt, na_rep="—", escape=True,
    )


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names from walk-forward CSV to report format."""
    df = df.rename(columns={
        "spec":    "Specification",
        "model":   "Model",
        "Dataset": "Specification",
        "horizon": "Horizon",   # NEW in versão_5
    })
    return df


def generate_performance_table(results_csv: str) -> dict:
    if not os.path.exists(results_csv):
        return {}

    df = _normalise(pd.read_csv(results_csv))

    metric_cols = [c for c in ["RMSE","MAE","R2","MASE","MAPE"] if c in df.columns]
    # NEW in versão_5: "Horizon" joins the grouping key so h=1 and h=2 are
    # never silently averaged together into one meaningless RMSE.
    id_cols     = [c for c in ["Specification","Model","Horizon"] if c in df.columns]

    # Aggregate folds → mean per Specification × Model
    if id_cols and metric_cols:
        df = df.groupby(id_cols)[metric_cols].mean().reset_index()

    cols   = [c for c in id_cols + metric_cols if c in df.columns]
    # CORRECTION (user request — 4 decimal places read as false precision for
    # RMSE/R2 at this scale): reduced from .round(4) to .round(3).
    df_tbl = df[cols].round(3)

    csv_path = os.path.join(paths.REPORTS_DIR, "table_performance.csv")
    tex_path = os.path.join(paths.REPORTS_DIR, "table_performance.tex")
    df_tbl.to_csv(csv_path, index=False)
    with open(tex_path, "w") as f:
        f.write(_to_latex(df_tbl,
                          "Walk-forward CV — mean RMSE/MAE/R² per model and specification",
                          "tab:performance"))
    print(f"  [report] Performance table → {csv_path}  ({len(df_tbl)} rows)")
    return {"csv": csv_path, "tex": tex_path}


def generate_hyperparameter_table(hp_csv: str) -> dict:
    if not hp_csv or not os.path.exists(hp_csv):
        return {}
    df = pd.read_csv(hp_csv)
    if df.empty:
        return {}
    csv_path = os.path.join(paths.REPORTS_DIR, "table_hyperparameters.csv")
    tex_path = os.path.join(paths.REPORTS_DIR, "table_hyperparameters.tex")
    df.to_csv(csv_path, index=False)
    with open(tex_path, "w") as f:
        f.write(_to_latex(df, "Hyperparameter search: method, space, and selected values",
                          "tab:hyperparameters"))
    print(f"  [report] Hyperparameter table → {csv_path}")
    return {"csv": csv_path, "tex": tex_path}


def generate_sarimax_coef_table(coef_csv: str) -> dict:
    if not coef_csv or not os.path.exists(coef_csv):
        return {}
    df = pd.read_csv(coef_csv)
    df_fmt = df.copy()
    for col in ["Coefficient","Std_Error","CI_lower_95","CI_upper_95"]:
        if col in df_fmt.columns:
            df_fmt[col] = df_fmt[col].apply(lambda x: f"{x:.4f}")
    if "p_value" in df_fmt.columns:
        df_fmt["p_value"] = df_fmt["p_value"].apply(
            lambda x: f"{x:.4f}{'***' if x<.01 else '**' if x<.05 else '*' if x<.1 else ''}"
        )
    csv_path = os.path.join(paths.REPORTS_DIR, "table_sarimax_coef.csv")
    tex_path = os.path.join(paths.REPORTS_DIR, "table_sarimax_coef.tex")
    df_fmt.to_csv(csv_path, index=False)
    with open(tex_path, "w") as f:
        f.write(_to_latex(df_fmt,
                          "SARIMAX coefficient estimates (*** p<0.01; ** p<0.05; * p<0.10)",
                          "tab:sarimax_coef"))
    print(f"  [report] SARIMAX coefficient table → {csv_path}")
    return {"csv": csv_path, "tex": tex_path}


def generate_ablation_table(ablation_csv: str, dm_csv: str) -> dict:
    if (not ablation_csv or not os.path.exists(ablation_csv)
            or os.path.getsize(ablation_csv) == 0):
        return {}
    df_abl = pd.read_csv(ablation_csv)
    cols   = ["Specification","Model","RMSE","R2"]
    df_out = df_abl[[c for c in cols if c in df_abl.columns]].copy()

    # DEFENSIVE GUARD (caught in sandbox testing): when spec_datasets has
    # only one specification (e.g. a reduced ABLATION_SPECS, or a horizon
    # where every other spec was skipped for lack of genuine test data —
    # see explainability/ablation.py's target-NaN filter), run_ablation()
    # has nothing to compare the baseline against, so ablation_dm_tests.csv
    # is written with zero rows AND zero columns — pd.DataFrame([]).to_csv()
    # still emits a trailing newline, so the file is 1 byte, not 0; a plain
    # size check does not catch it. pd.read_csv() raises EmptyDataError on
    # that (no header row to parse) rather than returning an empty frame,
    # which would otherwise crash report generation. Try the read and treat
    # "genuinely no columns" the same as "file absent".
    df_dm = None
    if dm_csv and os.path.exists(dm_csv):
        try:
            df_dm = pd.read_csv(dm_csv)
        except pd.errors.EmptyDataError:
            df_dm = None
    if df_dm is not None:
        # Handle both column name variants
        alt_col = "Alternative" if "Alternative" in df_dm.columns else "Specification"
        sig_col = next((c for c in ["Significant_5pct","Significant"] if c in df_dm.columns), None)
        merge_cols = [alt_col, "Model", "DM_p_value"] + ([sig_col] if sig_col else [])
        merge_cols = [c for c in merge_cols if c in df_dm.columns]
        dm_pv = df_dm[merge_cols].rename(columns={
            alt_col:  "Specification",
            "DM_p_value": "DM_p(vs baseline)",
        })
        df_out = df_out.merge(dm_pv, on=["Specification","Model"], how="left")

    csv_path = os.path.join(paths.REPORTS_DIR, "table_ablation.csv")
    tex_path = os.path.join(paths.REPORTS_DIR, "table_ablation.tex")
    df_out.to_csv(csv_path, index=False)
    with open(tex_path, "w") as f:
        f.write(_to_latex(df_out,
                          "Ablation study: impact of governance specifications on RMSE",
                          "tab:ablation"))
    print(f"  [report] Ablation table → {csv_path}  ({len(df_out)} rows)")
    return {"csv": csv_path, "tex": tex_path}


def generate_seed_robustness_table(results_csv: str) -> dict:
    """
    CORRECTION (user request — "implemente 3 seeds... para que no fim
    possamos fazer comparações de desempenho"): walkforward_results.csv now
    has one row per (fold, spec, model, seed) instead of one per
    (fold, spec, model). This aggregates across BOTH folds and seeds to
    report mean ± std per spec×model, so the reader can see how sensitive
    each model's RMSE/R2 is to its random initialization — a model whose
    std is a large fraction of its mean is not "well tuned" in a
    seed-robust sense, whatever its best single-seed number looks like.
    """
    if not results_csv or not os.path.exists(results_csv):
        return {}
    df = _normalise(pd.read_csv(results_csv))
    if "seed" not in df.columns:
        return {}

    # CORRECTION (user request — "incluir as métricas MAPE também nos logs"):
    # MAPE joins RMSE/R2 in the seed-robustness aggregation.
    metric_cols = [c for c in ["RMSE", "MAPE", "R2"] if c in df.columns]
    # NEW in versão_5: group by Horizon too, for the same reason as
    # generate_performance_table above.
    id_cols     = [c for c in ["Specification", "Model", "Horizon"] if c in df.columns]
    if not id_cols or not metric_cols:
        return {}

    agg = df.groupby(id_cols)[metric_cols].agg(["mean", "std"]).round(3)
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    agg = agg.reset_index()

    # Number of distinct seeds actually observed per group, and whether the
    # model's outcome is a deterministic no-op across seeds (SARIMAX, and
    # the BayesianRidge/Ridge fallbacks when PyMC/TensorFlow are
    # unavailable) — reported explicitly instead of silently showing
    # std=0 as if that were evidence of "well tuned" stability.
    n_seeds = df.groupby(id_cols)["seed"].nunique().rename("n_seeds")
    agg = agg.merge(n_seeds.reset_index(), on=id_cols)

    csv_path = os.path.join(paths.REPORTS_DIR, "table_seed_robustness.csv")
    agg.to_csv(csv_path, index=False)
    print(f"  [report] Seed-robustness table → {csv_path}  ({len(agg)} rows)")
    return {"csv": csv_path}


def generate_seed_comparison_table(results_csv: str) -> dict:
    """
    NEW (user request — "os logs têm que mostrar tabelas de comparação de
    RMSE e MAPE em relação aos datasets, modelos e seeds"): pivots so each
    seed's own RMSE/MAPE sits in its own column, side by side, per
    Specification × Model × Horizon — a direct, literal dataset × model ×
    seed comparison (Horizon included since versão_5 re-evaluates every
    spec×model once per genuine forecast horizon, h=1..5).
    """
    if not results_csv or not os.path.exists(results_csv):
        return {}
    df = _normalise(pd.read_csv(results_csv))
    if "seed" not in df.columns:
        return {}

    id_cols     = [c for c in ["Specification", "Model", "Horizon"] if c in df.columns]
    metric_cols = [c for c in ["RMSE", "MAPE"] if c in df.columns]
    if not id_cols or not metric_cols:
        return {}

    agg = df.groupby(id_cols + ["seed"])[metric_cols].mean().reset_index()
    piv = agg.pivot(index=id_cols, columns="seed", values=metric_cols)
    piv.columns = [f"{m}_seed{s}" for m, s in piv.columns]
    piv = piv.reset_index().round(3)

    csv_path = os.path.join(paths.REPORTS_DIR, "table_rmse_mape_seed_comparison.csv")
    piv.to_csv(csv_path, index=False)
    print(f"  [report] RMSE/MAPE × dataset × modelo × horizonte × seed comparison table → {csv_path}  ({len(piv)} rows)")
    return {"csv": csv_path}


def generate_cost_table(results_csv: str) -> dict:
    """
    CORRECTION (user request — "implemente funções para medição do custo
    computacional para cada modelo e estratégia" / later refined to
    "mostrar nos logs o custo computacional de cada dataset (estratégia)
    de cada modelo"): aggregates the per-fold fit_time_s / predict_time_s
    / peak_mem_mb instrumentation. Three granularities are produced: per
    Model alone (across all specs/horizons), per Specification×Model
    (dataset-level breakdown), and — NEW in versão_5, since cost can also
    vary with the forecast horizon (different leak-guard drop fractions
    change effective training-set size per horizon) — per
    Specification×Model×Horizon.
    peak_mem_mb is a process-wide high-water mark (see
    validation/walk_forward.py::_peak_rss_mb docstring) — reported as a
    max across rows, not summed/averaged, since it is not an isolated
    per-call measurement.
    """
    if not results_csv or not os.path.exists(results_csv):
        return {}
    df = _normalise(pd.read_csv(results_csv))
    cost_cols = [c for c in ["fit_time_s", "predict_time_s"] if c in df.columns]
    if "Model" not in df.columns or not cost_cols:
        return {}

    agg = df.groupby("Model")[cost_cols].mean().round(3)
    if "peak_mem_mb" in df.columns:
        agg["peak_mem_mb_max"] = df.groupby("Model")["peak_mem_mb"].max().round(1)
    agg["n_runs"] = df.groupby("Model").size()
    agg = agg.reset_index()

    csv_path = os.path.join(paths.REPORTS_DIR, "table_computational_cost.csv")
    agg.to_csv(csv_path, index=False)
    print(f"  [report] Computational-cost table (por modelo) → {csv_path}  ({len(agg)} rows)")

    result = {"csv": csv_path}

    group_cols = [c for c in ["Specification", "Model", "Horizon"] if c in df.columns]
    if "Specification" in df.columns and len(group_cols) >= 2:
        agg2 = df.groupby(group_cols)[cost_cols].mean().round(3)
        if "peak_mem_mb" in df.columns:
            agg2["peak_mem_mb_max"] = df.groupby(group_cols)["peak_mem_mb"].max().round(1)
        agg2["n_runs"] = df.groupby(group_cols).size()
        agg2 = agg2.reset_index()

        csv_path2 = os.path.join(paths.REPORTS_DIR, "table_computational_cost_by_dataset.csv")
        agg2.to_csv(csv_path2, index=False)
        print(f"  [report] Computational-cost table (por dataset × modelo × horizonte) → {csv_path2}  ({len(agg2)} rows)")
        result["csv_by_dataset"] = csv_path2

    return result


def generate_horizon_comparison_table(results_csv: str) -> dict:
    """
    NEW in versão_5 (user request — originally "implemente isso para
    horizontes de 1 e 2 anos... para que no fim possamos fazer comparações
    de desempenho", later extended to genuine forecasts for h=1..5 — no
    contemporaneous/h=0 case). Pivots mean RMSE/MAPE/R2 (aggregated across
    folds and seeds) so every horizon sits side by side per
    Specification×Model, plus the raw and relative RMSE change of each
    h≥2 relative to the smallest genuine-forecast horizon evaluated
    (normally h=1) — generalized to however many horizons
    config/features.py::FORECAST_HORIZONS actually declares, instead of a
    hardcoded h1-vs-h2 pair. A positive delta means that horizon's forecast
    is, on average, worse than the baseline one for that spec×model — the
    expected direction (harder to predict further out) but not assumed; it
    is read off the actual walk-forward results.
    """
    if not results_csv or not os.path.exists(results_csv):
        return {}
    df = _normalise(pd.read_csv(results_csv))
    if "Horizon" not in df.columns:
        return {}

    id_cols     = [c for c in ["Specification", "Model"] if c in df.columns]
    metric_cols = [c for c in ["RMSE", "MAPE", "R2"] if c in df.columns]
    if not id_cols or not metric_cols:
        return {}

    agg = df.groupby(id_cols + ["Horizon"])[metric_cols].mean().reset_index()
    piv = agg.pivot(index=id_cols, columns="Horizon", values=metric_cols)
    piv.columns = [f"{m}_h{int(h)}" for m, h in piv.columns]
    piv = piv.reset_index()

    # Baseline horizon for the delta comparison: the smallest genuine
    # forecast horizon actually evaluated (normally h=1).
    horizons_present = sorted(df["Horizon"].dropna().unique().astype(int))
    if horizons_present:
        baseline_h = horizons_present[0]
        base_col = f"RMSE_h{baseline_h}"
        if base_col in piv.columns:
            for h in horizons_present:
                if h == baseline_h:
                    continue
                col = f"RMSE_h{h}"
                if col in piv.columns:
                    piv[f"RMSE_delta_h{h}_minus_h{baseline_h}"] = piv[col] - piv[base_col]
                    piv[f"RMSE_delta_pct_h{h}"] = (
                        100 * piv[f"RMSE_delta_h{h}_minus_h{baseline_h}"] / piv[base_col].replace(0, np.nan)
                    )
    piv = piv.round(3)

    csv_path = os.path.join(paths.REPORTS_DIR, "table_horizon_comparison.csv")
    piv.to_csv(csv_path, index=False)
    print(f"  [report] Horizon-comparison table → {csv_path}  ({len(piv)} rows)")
    return {"csv": csv_path}


def generate_executive_summary(results: dict) -> str:
    now   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Pipeline Executive Summary",
        f"*Generated: {now}*\n",
        f"**Project**: {cfg.PROJECT_NAME} v{cfg.PROJECT_VERSION}\n",
        "## Method",
        "Walk-forward cross-validation (5 folds) with fold-level MICE imputation, "
        "StandardScaler, and PCA applied exclusively on training data. "
        "Optuna TPE hyperparameter search (50 trials per model). "
        "Ablation study over 5 governance specifications.\n",
        "## Key results",
    ]
    for key, val in results.items():
        fmt = f"{val:.3f}" if isinstance(val, float) else str(val)
        lines.append(f"- **{key}**: {fmt}")

    md   = "\n".join(lines)
    path = os.path.join(paths.REPORTS_DIR, "executive_summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"  [report] Executive summary → {path}")
    return path


def run_all_reports(
    results_csv:      str | None = None,
    hp_csv:           str | None = None,
    sarimax_coef_csv: str | None = None,
    ablation_csv:     str | None = None,
    ablation_dm_csv:  str | None = None,
    summary_kv:       dict | None = None,
) -> list:
    print("\n" + "=" * 60)
    print("  REPORTS: generating dissertation tables")
    print("=" * 60)

    generated = []

    if results_csv:
        r = generate_performance_table(results_csv)
        generated.extend(r.values())
        r = generate_seed_robustness_table(results_csv)
        generated.extend(r.values())
        r = generate_seed_comparison_table(results_csv)
        generated.extend(r.values())
        r = generate_cost_table(results_csv)
        generated.extend(r.values())
        r = generate_horizon_comparison_table(results_csv)   # NEW in versão_5
        generated.extend(r.values())

    if hp_csv:
        r = generate_hyperparameter_table(hp_csv)
        generated.extend(r.values())

    if sarimax_coef_csv:
        r = generate_sarimax_coef_table(sarimax_coef_csv)
        generated.extend(r.values())

    if ablation_csv:
        r = generate_ablation_table(ablation_csv, ablation_dm_csv or "")
        generated.extend(r.values())

    if summary_kv:
        p = generate_executive_summary(summary_kv)
        generated.append(p)

    print(f"\n  {len(generated)} report files generated in {paths.REPORTS_DIR}/")
    return generated
