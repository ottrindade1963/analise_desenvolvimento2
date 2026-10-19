import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer


class PanelMICEImputer(BaseEstimator, TransformerMixin):
    """
    Per-country MICE imputer for panel (longitudinal) data.

    Parameters
    ----------
    max_iter : int
        Maximum MICE iterations (default 20).
    random_state : int
        Reproducibility seed.
    country_col : str
        Name of the country identifier column.

    Notes
    -----
    fit() stores one IterativeImputer per country, trained on X_train.
    transform() applies each stored imputer to the corresponding country rows.
    Countries absent from training use the global imputer as fallback.
    """

    def __init__(self, max_iter: int = 20, random_state: int = 42,
                 country_col: str = "country_code",
                 exclude_cols: list = None):
        self.max_iter     = max_iter
        self.random_state = random_state
        self.country_col  = country_col
        
        self.exclude_cols = list(exclude_cols) if exclude_cols else []

    def fit(self, X: pd.DataFrame, y=None):
        self._imputers: dict = {}
        self._feature_cols: list = [
            c for c in X.select_dtypes(include=[np.number]).columns
            if c != "year" and c not in self.exclude_cols
        ]

        countries = X[self.country_col].unique() if self.country_col in X.columns else []

        for country in countries:
            mask  = X[self.country_col] == country
            X_sub = X.loc[mask, self._feature_cols].copy()
            cols_ok = [c for c in self._feature_cols if X_sub[c].notna().sum() >= 3]

            if len(cols_ok) < 2:
                continue

            try:
                imp = IterativeImputer(
                    max_iter=self.max_iter,
                    random_state=self.random_state,
                    n_nearest_features=min(5, len(cols_ok) - 1),
                    initial_strategy="median",
                    skip_complete=True,
                )
                imp.fit(X_sub[cols_ok])
                self._imputers[country] = (imp, cols_ok)
            except Exception:
                pass

        # Global fallback imputer
        self._global_imputer = IterativeImputer(
            max_iter=self.max_iter,
            random_state=self.random_state,
            initial_strategy="median",
            skip_complete=True,
        )
        self._global_imputer.fit(X[self._feature_cols].fillna(X[self._feature_cols].median()))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()

        if self.country_col not in X.columns:
            X_out[self._feature_cols] = self._global_imputer.transform(
                X_out[self._feature_cols]
            )
            return X_out

        for country, (imp, cols_ok) in self._imputers.items():
            mask = X_out[self.country_col] == country
            if mask.sum() == 0:
                continue
            try:
                X_out.loc[mask, cols_ok] = imp.transform(X_out.loc[mask, cols_ok])
            except Exception:
                X_out.loc[mask, self._feature_cols] = (
                    X_out.loc[mask, self._feature_cols]
                    .interpolate(method="linear", limit_direction="both")
                )

        # Fallback for countries not seen during fit
        still_missing = X_out[self._feature_cols].isna().any().any()
        if still_missing:
            X_out[self._feature_cols] = self._global_imputer.transform(
                X_out[self._feature_cols].fillna(X_out[self._feature_cols].median())
            )

        return X_out
