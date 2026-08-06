"""features/engineer.py — Fold-safe feature engineering for panel data.

All transformations (lags, rolling means, PCA) are fitted on training
data only and applied to test data via fit/transform — no look-ahead bias.

CORRECTION (versão_5, discovered while implementing genuine h-step-ahead
forecasting — see validation/walk_forward.py::build_horizon_target and
preprocessing/temporal.py): _add_lags()/_add_deltas() used to build lag/
delta features with a POSITIONAL shift within each country's sorted rows
(df.groupby("country_code")[col].shift(lag)). The WGI-merged panel is only
biennial for 1996-2002 (1997/1999/2001 do not exist for ANY country), so a
positional shift(1) at year 2000 silently pulled the 1998 value — a 2-year
gap mislabelled as "_lag1". This never surfaced before because nothing
downstream depended on the lag being calendar-exact. It matters here
because a genuine forecast horizon is defined in calendar years, so
_add_lags/_add_deltas now use preprocessing.temporal.year_exact_shift(),
which matches by (country, year) instead of by row position and returns
NaN — correctly — when no row for the exact target year exists.
_add_rolling() is unchanged: a trailing mean over "whatever observations
exist in the window" remains a defensible definition even across a gap.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import config.variables as var
import config.features  as feat
from preprocessing.temporal import year_exact_shift


class FoldFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Builds temporal features (lags, rolling means, PCA on WGI) within a fold.

    fit()      — learns PCA from the training slice only.
    transform()— applies stored PCA + creates lag/rolling features using
                 backward-only pandas shift/rolling (no leakage by construction).
    """

    def __init__(self, spec="WDI_plus_PCA1",
                 lags_wdi=None, lags_wgi=None,
                 lags_target=None, rolling_window=3):
        self.spec           = spec
        self.lags_wdi       = lags_wdi       or feat.LAGS_WDI
        self.lags_wgi       = lags_wgi       or feat.LAGS_WGI
        self.lags_target    = lags_target    or feat.LAGS_TARGET
        self.rolling_window = rolling_window or feat.ROLLING_WINDOW

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y=None):
        spec_cfg        = feat.ABLATION_SPECS.get(self.spec, {})
        self._use_pca   = spec_cfg.get("wgi_pca", False)
        self._use_raw   = spec_cfg.get("wgi_raw", False)
        self._use_inter = spec_cfg.get("interactions", False)

        self._pca        = None
        self._pca_scaler = None
        self._wgi_cols   = [c for c in var.WGI_COLS if c in X.columns]

        if self._use_pca and len(self._wgi_cols) >= 2:
            data_wgi = X[self._wgi_cols].dropna()
            if len(data_wgi) >= 10:
                self._pca_scaler = StandardScaler()
                self._pca = PCA(
                    n_components=min(feat.PCA_N_COMPONENTS, len(self._wgi_cols)),
                    random_state=42,
                )
                scaled = self._pca_scaler.fit_transform(data_wgi)
                self._pca.fit(scaled)
                var_pc1 = self._pca.explained_variance_ratio_[0] * 100
                print(f"    PCA fitted on {len(data_wgi)} obs — "
                      f"PC1 explains {var_pc1:.1f}% variance")
        return self

    # ── transform ────────────────────────────────────────────────────────────

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy().sort_values(["country_code", "year"])

        # 1. PCA governance factor
        if self._use_pca and self._pca is not None:
            cols = [c for c in self._wgi_cols if c in df.columns]
            if len(cols) >= 2:
                mask = df[cols].notna().all(axis=1)
                if mask.sum() > 0:
                    X_wgi = df.loc[mask, cols].values          # (n, n_wgi)
                    X_sc  = self._pca_scaler.transform(X_wgi)  # (n, n_wgi)
                    # Project onto PC1: (n, n_wgi) @ (n_wgi,) → (n,)
                    # FIX: was `scaled[:, 0] @ components[0]` which
                    # tried (n,) @ (n_wgi,) and raised dimension error.
                    pc1_scores = X_sc @ self._pca.components_[0]
                    df.loc[mask, "wgi_pca1"] = pc1_scores

                df["wgi_pca1"] = df.groupby("country_code")["wgi_pca1"].transform(
                    lambda x: x.interpolate(method="linear", limit_direction="both")
                )
                df["wgi_pca1"] = df["wgi_pca1"].fillna(0)

            df, _ = self._add_lags(df, ["wgi_pca1"], self.lags_wgi)
            df, _ = self._add_rolling(df, ["wgi_pca1"], self.rolling_window)
            df, _ = self._add_deltas(df, ["wgi_pca1"])
            # Remove contemporaneous PC1 (enforce antecedence)
            if "wgi_pca1" in df.columns:
                df = df.drop(columns=["wgi_pca1"])

        # 2. Raw WGI lags
        if self._use_raw:
            df, _ = self._add_lags(df, self._wgi_cols, self.lags_wgi)

        # CORRECTION (root-cause diagnostic report, Secção 2 / recomendação #1):
        # the contemporaneous raw WGI columns inherited from the input X were
        # never dropped here, regardless of self._use_pca / self._use_raw — they
        # leaked into every specification's feat_cols, including A1_WDI_only.
        # They have already been used above (to build wgi_pca1 and/or the raw
        # lag columns); now remove the contemporaneous originals unconditionally,
        # exactly as the code already did for the derived "wgi_pca1" column.
        raw_wgi_present = [c for c in self._wgi_cols if c in df.columns]
        if raw_wgi_present:
            df = df.drop(columns=raw_wgi_present)

        # 3. WDI lags and rolling means
        wdi_present = [c for c in var.WDI_COLS if c in df.columns]
        df, _ = self._add_lags(df, wdi_present, self.lags_wdi)
        df, _ = self._add_rolling(df, wdi_present[:5], self.rolling_window)

        # 4. Target autoregressive lags
        if var.TARGET in df.columns:
            df, _ = self._add_lags(df, [var.TARGET], self.lags_target)

        # 5. Interaction terms
        if self._use_inter:
            df = self._add_interactions(df)

        # 6. Drop rows without sufficient lag history
        max_lag = max(self.lags_wgi + self.lags_wdi + self.lags_target)
        yr_min  = int(df["year"].min())
        df      = df[df["year"] >= yr_min + max_lag].copy()

        # 7. Fill residual NaNs in feature columns
        feat_cols = self._feature_columns(df)
        df[feat_cols] = df[feat_cols].fillna(0)

        return df

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _add_lags(df, cols, lags):
        new = []
        for col in cols:
            if col not in df.columns:
                continue
            for lag in lags:
                name = f"{col}_lag{lag}"
                # CORRECTION (versão_5): year-exact, not positional — see
                # module docstring and preprocessing/temporal.py.
                df[name] = year_exact_shift(df, col, lag)
                new.append(name)
        return df, new

    @staticmethod
    def _add_rolling(df, cols, window):
        new = []
        for col in cols:
            if col not in df.columns:
                continue
            name = f"{col}_ma{window}"
            df[name] = df.groupby("country_code")[col].transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
            new.append(name)
        return df, new

    @staticmethod
    def _add_deltas(df, cols):
        new = []
        for col in cols:
            if col not in df.columns:
                continue
            name = f"{col}_delta"
            # CORRECTION (versão_5): year-exact 1-year delta, not positional
            # .diff(1) — see module docstring and preprocessing/temporal.py.
            df[name] = df[col].values - year_exact_shift(df, col, 1)
            new.append(name)
        return df, new

    def _add_interactions(self, df):
        # CORRECTION (root-cause diagnostic report, Secção 3 / recomendação #2):
        # this method used to hardcode "wgi_pca1_lag1" as the only possible
        # governance term. That column only exists when self._use_pca is True
        # (specs A2/A4). For A5_WDI_6WGI_inter (wgi_pca=False, wgi_raw=True,
        # interactions=True) pca_lag was always None, so this function returned
        # df unchanged — A5 silently got zero interaction terms and became
        # numerically identical to A3_WDI_6WGI.
        #
        # Fix: fall back to a composite governance signal built from the mean
        # of the raw WGI lag-1 columns when the PCA lag is not available, so
        # every spec with interactions=True actually gets non-trivial terms.
        gov_lag, gov_label = None, None
        if "wgi_pca1_lag1" in df.columns:
            gov_lag   = df["wgi_pca1_lag1"]
            gov_label = "pca1"
        else:
            raw_lag_cols = [f"{c}_lag1" for c in self._wgi_cols if f"{c}_lag1" in df.columns]
            if raw_lag_cols:
                gov_lag   = df[raw_lag_cols].mean(axis=1)
                gov_label = "wgicomp"

        if gov_lag is None:
            return df

        eco_vars = ["ied_percent_pib",
                    "formacao_bruta_capital_fixo_percent_pib",
                    "comercio_percent_pib",
                    "pib_per_capita_ppc"]
        for eco in eco_vars:
            eco_col = f"{eco}_lag1" if f"{eco}_lag1" in df.columns else eco
            if eco_col not in df.columns:
                continue
            name   = f"inter_{gov_label}_{eco.split('_')[0]}"
            s_gov  = gov_lag.std()        or 1.0
            s_eco  = df[eco_col].std()    or 1.0
            df[name] = (gov_lag / s_gov) * (df[eco_col] / s_eco)
        return df

    @staticmethod
    def _feature_columns(df) -> list:
        excl = {"country_code", "year", "pais", var.TARGET}
        return [c for c in df.select_dtypes(include=[np.number]).columns
                if c not in excl]

    def save(self, path: str) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "FoldFeatureEngineer":
        return joblib.load(path)
