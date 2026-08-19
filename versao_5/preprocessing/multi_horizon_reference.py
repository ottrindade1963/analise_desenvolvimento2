"""preprocessing/multi_horizon_reference.py — Wide multi-horizon REFERENCE
dataset, built at the WDI+WGI aggregation stage (before any training).

NEW in versão_5 (user request — "Antes do treinamento, no processo de
agregação de dados WDI + WGI, deve gerar um dataset com os inputs (WDI +
WGI) nos seus respetivos anos e a outputs com a variável alvo 1 (t+1),
variável alvo 2 (t+2), ..., variável alvo 5 (t+5), cujo shift vai deixar as
últimas células (últimos anos das variáveis alvo) vazias, que têm que ser
preenchidas com a mediana. Cada combinação de inputs + ano de previsão deve
gerar também um dataset independente.").

IMPORTANT — scope of this module, agreed explicitly with the user before
implementation: the median-filled values produced here are for
DOCUMENTATION / TRANSPARENCY only. They are never read by pipeline.py, by
validation/walk_forward.py, or by any other training/evaluation code path
— those continue to use validation/walk_forward.py::build_horizon_target()
+ the existing leak-guard/NaN-drop mechanism (see relatorio2), which is the
already-validated, honest way this project turns a raw panel into training
labels. Every value fabricated here is also flagged in its own boolean
column (see FLAG_SUFFIX below) so nobody reading these CSVs later mistakes
a median-filled cell for a real observation. This module writes to a
dedicated data/aggregated/referencia_multi_horizonte/ subdirectory that
pipeline.py never reads from — the separation is structural, not just a
convention that a future edit could silently break.

WHY the cells are empty in the first place: for a country's last H years of
real data, the target H years ahead does not exist yet (e.g., for h=5, a
2022 row would need the 2027 value of the target — which is not in the
panel because it has not happened / been published yet). This is the same
forecast-horizon boundary documented in relatorio1/relatorio2 — it is not a
data-quality defect.
"""
import os
import numpy as np
import pandas as pd

from preprocessing.temporal import year_exact_shift

FLAG_SUFFIX = "_preenchido_mediana"


def _per_country_median_fill(df: pd.DataFrame, col: str, id_col: str) -> tuple:
    """
    Fill NaN cells in `col` with that row's OWN country's median of `col`
    (computed only from the REAL, non-fabricated values already in that
    column for that country — never from other countries, and never
    recursively from other fabricated cells). Returns (filled_series,
    flag_series) — flag_series is True exactly where a fabricated value was
    substituted in.

    Per-country (not global) median, per explicit user confirmation: a
    single global median across all 37 countries would blend economies of
    very different scale/structure (e.g. a small economy inheriting a
    large one's typical level), which would be a substantially worse and
    more misleading fill than each country's own historical median.
    """
    is_missing = df[col].isna()
    medians = df.groupby(id_col)[col].transform("median")
    filled = df[col].where(~is_missing, medians)
    # A country with EVERY value missing for this horizon (should not occur
    # in this panel, but guarded rather than assumed) would still have a
    # NaN median — leave those as NaN rather than inventing a value with no
    # basis whatsoever, and do not mark them as "filled".
    truly_filled = is_missing & filled.notna()
    return filled, truly_filled


def build_multi_horizon_reference(
    df_raw: pd.DataFrame,
    target_col: str,
    horizons: list = (1, 2, 3, 4, 5),
    id_col: str = "country_code",
    year_col: str = "year",
) -> dict:
    """
    Build the wide multi-horizon reference dataset plus one independent
    per-horizon dataset (inputs + single shifted target).

    Returns a dict:
        {
          "wide":            DataFrame — every input column from df_raw,
                              plus one f"{target_col}_t+{h}" (and its
                              f"...{FLAG_SUFFIX}") column per horizon,
          "per_horizon": {h: DataFrame(inputs + f"{target_col}_t+{h}"
                              + flag) for h in horizons},
        }

    "Inputs" = every column of df_raw EXCEPT the target itself (WDI + WGI
    indicators, id_col, year_col) at each row's own year t — i.e. exactly
    what the row already contains; only the target columns are new/shifted.
    """
    df_wide = df_raw.copy()

    for h in horizons:
        col = f"{target_col}_t+{h}"
        # Reuses the SAME year-exact self-join mechanism already validated
        # for genuine-horizon label construction (preprocessing/temporal.py
        # — see relatorio2 §3.1/§6 for the sandbox-caught bug this exact
        # function was hardened against). offset=-h → lead (future) lookup.
        df_wide[col] = year_exact_shift(df_raw, target_col, -h,
                                        id_col=id_col, year_col=year_col)
        filled, flag = _per_country_median_fill(df_wide, col, id_col)
        df_wide[col] = filled
        df_wide[f"{col}{FLAG_SUFFIX}"] = flag

    per_horizon = {}
    input_cols = [c for c in df_raw.columns if c != target_col]
    for h in horizons:
        col = f"{target_col}_t+{h}"
        cols = input_cols + [col, f"{col}{FLAG_SUFFIX}"]
        per_horizon[h] = df_wide[cols].copy()

    return {"wide": df_wide, "per_horizon": per_horizon}


def save_multi_horizon_reference(result: dict, out_dir: str,
                                  target_col: str) -> dict:
    """Write the wide dataset and every per-horizon dataset to disk."""
    os.makedirs(out_dir, exist_ok=True)
    paths_written = {}

    wide_path = os.path.join(out_dir, "dataset_largo_multi_horizonte_REFERENCIA.csv")
    result["wide"].to_csv(wide_path, index=False)
    paths_written["wide"] = wide_path
    print(f"  [multi-horizonte] Dataset largo de referência (h=1..{max(result['per_horizon'])}) "
          f"→ {wide_path}  ({len(result['wide'])} linhas)")

    for h, df_h in result["per_horizon"].items():
        h_dir = os.path.join(out_dir, f"h{h}")
        os.makedirs(h_dir, exist_ok=True)
        p = os.path.join(h_dir, f"dataset_inputs_mais_alvo_t+{h}_REFERENCIA.csv")
        df_h.to_csv(p, index=False)
        n_filled = int(df_h[f"{target_col}_t+{h}{FLAG_SUFFIX}"].sum())
        paths_written[h] = p
        print(f"  [multi-horizonte] Dataset independente h={h} → {p}  "
              f"({len(df_h)} linhas, {n_filled} células de alvo preenchidas por mediana)")

    return paths_written
