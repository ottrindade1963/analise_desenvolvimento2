"""preprocessing/imputer.py — Fold-safe MICE imputer for panel data.

PanelMICEImputer is a scikit-learn compatible Transformer that applies
IterativeImputer independently per country.  Because it implements
fit() and transform() separately, it can be placed inside a Pipeline
and will never leak future information when used within walk-forward CV:
  - fit()       → learns imputation models from training data ONLY
  - transform() → applies those models to test data

This is the correct solution for Problem 1 (look-ahead bias) described
in the methodological review.
"""
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
        # CORRECTION (found while investigating suspiciously high R2 in the
        # walk-forward results): by default this transformer imputed EVERY
        # numeric column except "year" — including the target itself
        # (config.variables.TARGET is just another numeric WDI column here).
        # MICE reconstructs a missing target from the OTHER covariates in the
        # same row (e.g. manufacturing value-added, which is highly
        # correlated with industrial value-added). Those same covariates are
        # then used as model features, so any row whose true label was
        # missing got a fabricated label that the model could partially
        # "reconstruct" from its own inputs — inflating R2/deflating RMSE
        # specifically on those rows, for both training and (worse) test
        # evaluation. Callers now pass exclude_cols=[var.TARGET] so the
        # target is NEVER imputed; rows with a genuinely missing target stay
        # NaN and are dropped explicitly by the caller before fit/scoring.
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
