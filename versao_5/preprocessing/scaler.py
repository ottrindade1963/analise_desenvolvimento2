"""preprocessing/scaler.py — Fold-safe StandardScaler for numeric panel features."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler as _SS


class PanelScaler(BaseEstimator, TransformerMixin):
    """
    StandardScaler that operates only on numeric feature columns,
    preserving id columns (country_code, year, pais) unchanged.

    Placed inside a sklearn Pipeline it is fitted exclusively on
    training data and applied to test data, preventing scale leakage.
    """

    def __init__(self, exclude: list[str] | None = None):
        self.exclude = exclude or ["country_code", "year", "pais"]

    def fit(self, X: pd.DataFrame, y=None):
        self._num_cols = [
            c for c in X.select_dtypes(include=[np.number]).columns
            if c not in self.exclude
        ]
        self._scaler = _SS()
        self._scaler.fit(X[self._num_cols])
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        X_out[self._num_cols] = self._scaler.transform(X_out[self._num_cols])
        return X_out

    def inverse_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        X_out[self._num_cols] = self._scaler.inverse_transform(X_out[self._num_cols])
        return X_out
