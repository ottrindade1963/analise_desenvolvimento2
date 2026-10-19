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
