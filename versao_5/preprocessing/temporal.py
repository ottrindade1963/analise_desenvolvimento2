import pandas as pd


def year_exact_shift(df: pd.DataFrame, col: str, offset: int,
                      id_col: str = "country_code",
                      year_col: str = "year",
                      source: pd.DataFrame = None) -> pd.Series:
   
    src_df = source if source is not None else df
    src = src_df[[id_col, year_col, col]].copy()
    src[year_col] = src[year_col] + offset
    src = src.rename(columns={col: "__year_exact_shift__"})
    # keep only the first occurrence per (id_col, year_col) in case of
    # duplicate rows — defensive, should not happen given upstream
    # validate="1:1" join, but a silent duplicate would otherwise silently
    # multiply rows through this merge.
    src = src.drop_duplicates(subset=[id_col, year_col], keep="first")
    merged = df[[id_col, year_col]].merge(
        src, on=[id_col, year_col], how="left"
    )
    return merged["__year_exact_shift__"].values
