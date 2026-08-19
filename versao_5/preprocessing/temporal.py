"""preprocessing/temporal.py — Calendar-year-exact shift for panel data.

NEW in versão_5. Discovered while implementing genuine h-step-ahead
forecasting (see validation/walk_forward.py::build_horizon_target): the
World Bank WGI indicators (and therefore this project's merged panel) are
only available for 1996, 1998, 2000 and 2002 — biennial, not annual — and
only become annual from 2003 onward. features/engineer.py's existing
_add_lags()/_add_deltas() build lag/delta features with
`df.groupby("country_code")[col].shift(lag)`, a POSITIONAL shift over each
country's sorted rows. For any country-year inside the 1996-2002 stretch,
the "previous row" is 2 calendar years back, not 1 — so a column labelled
"_lag1" silently contains a 2-year-old value there, and "_delta" silently
computes a 2-year change. This is invisible from the output (no error, no
NaN) and was not caught before because no part of the original pipeline
depended on the shift being calendar-exact — it only had to be "some
consistent notion of past". Genuine h-step-ahead forecasting does depend on
being calendar-exact: mislabelling a 2-year gap as h=1 would silently
change what is actually being forecast.

year_exact_shift() replaces the positional shift with a self-join on
(id_col, year_col ± offset), so a lag/lead is only ever populated when a
row for that EXACT calendar year exists for that entity; otherwise it is
correctly NaN rather than silently wrong.
"""
import pandas as pd


def year_exact_shift(df: pd.DataFrame, col: str, offset: int,
                      id_col: str = "country_code",
                      year_col: str = "year",
                      source: pd.DataFrame = None) -> pd.Series:
    """
    Returns col's value `offset` calendar years away from each row, matched
    exactly by (id_col, year_col), NOT by row position.

    offset > 0 -> lag  (value from `offset` years in the PAST)
    offset < 0 -> lead (value from `abs(offset)` years in the FUTURE)

    Rows for which no row with the exact target year exists for that
    entity (gap years, start/end of series) get NaN — which is the
    correct, honest answer, not an approximation.

    source : DataFrame, optional
        Where (id_col, year_col, col) triples are looked up from. Defaults
        to `df` itself — fine for lag features, since the value being
        looked up is always in the PAST and therefore always already
        present in whatever fold-scoped frame `df` is. It matters for a
        LEAD/horizon lookup (offset < 0): a test row near the end of a
        walk-forward fold's own (train ∪ test) frame may need a future
        year that falls outside that fold's own slice entirely (see
        validation/walk_forward.py::build_horizon_target, which passes the
        full raw panel here for exactly this reason — the target column is
        never touched by imputation or feature engineering, so reading it
        from the full panel instead of the fold-scoped frame introduces no
        extra information the fold-scoped frame didn't already have access
        to, only a wider year range to search in).
    """
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
