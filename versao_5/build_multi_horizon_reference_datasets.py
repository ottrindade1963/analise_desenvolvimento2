"""build_multi_horizon_reference_datasets.py — Standalone pre-training step.

NEW in versão_5 (user request — "Antes do treinamento, no processo de
agregação de dados WDI + WGI, deve gerar um dataset com os inputs (WDI +
WGI) nos seus respetivos anos e a outputs com a variável alvo 1 (t+1)...
variável alvo 5 (t+5)... preenchidas com a mediana. Cada combinação de
inputs + ano de previsão deve gerar também um dataset independente.").

Run this AFTER the WDI+WGI aggregation step has produced
data/aggregated/agregado_inner_join.csv (the same file pipeline.py's
_load_clean_data() reads) and BEFORE running pipeline.py:

    python build_multi_horizon_reference_datasets.py

WHY THIS IS A SEPARATE SCRIPT, NOT A STEP INSIDE pipeline.py: the datasets
it produces have their trailing empty target cells filled with each
country's own historical median — a deliberate, clearly-flagged
fabrication for documentation/completeness purposes only (see
preprocessing/multi_horizon_reference.py's module docstring for the full
rationale, and relatorio2 for the disclosure). Keeping this entirely
outside pipeline.py's import graph is a structural guarantee — not just a
convention — that these median-filled values can never be accidentally
wired into model training or evaluation by some future change to
pipeline.py: there is no import path from pipeline.py to this script, and
the directory it writes to (data/aggregated/referencia_multi_horizonte/)
is never read by _load_clean_data() or by anything else pipeline.py calls.
The actual training/evaluation labels keep coming exclusively from
validation/walk_forward.py::build_horizon_target(), which drops
(never fabricates) rows whose true target is unknown.
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd

import config.paths     as paths
import config.variables as var
from preprocessing.multi_horizon_reference import (
    build_multi_horizon_reference,
    save_multi_horizon_reference,
)


def main():
    src_path = os.path.join(paths.AGGREGATED_DIR, "agregado_inner_join.csv")
    if not os.path.exists(src_path):
        raise FileNotFoundError(
            f"Execute primeiro a agregação WDI+WGI (INNER JOIN). Esperado: {src_path}"
        )
    df_raw = pd.read_csv(src_path)
    print(f"  Dados de entrada: {df_raw.shape}  "
          f"({df_raw['country_code'].nunique()} países, "
          f"{df_raw['year'].min()}–{df_raw['year'].max()})")

    result = build_multi_horizon_reference(
        df_raw,
        target_col=var.TARGET,
        horizons=[1, 2, 3, 4, 5],
        id_col="country_code",
        year_col="year",
    )

    out_dir = os.path.join(paths.AGGREGATED_DIR, "referencia_multi_horizonte")
    save_multi_horizon_reference(result, out_dir, target_col=var.TARGET)

    print("\n  NOTA: estes ficheiros são de referência/documentação. As células de alvo")
    print("  preenchidas pela mediana (por país) estão sinalizadas nas colunas")
    print("  '*_preenchido_mediana' e NÃO são usadas no treino nem na avaliação dos")
    print("  modelos — ver preprocessing/multi_horizon_reference.py e relatorio2.")


if __name__ == "__main__":
    main()
