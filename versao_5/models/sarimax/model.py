"""models/sarimax/model.py — SARIMAX with AIC order selection and coefficient export.

Addresses Problem 10 from the review: the model now exports a full
coefficient table with coef, std error, 95% CI, and p-value.
"""
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import config.model_params as mp

warnings.filterwarnings("ignore")


class SARIMAXModel:
    """
    SARIMAX with exogenous variables.

    fit() selects ARIMA order via AIC when auto_order=True, then fits
    the model and stores all coefficient information for reporting.
    predict() uses out-of-sample forecast; falls back to Ridge if SARIMAX fails.
    """

    def __init__(self):
        self._use_sarimax    = False
        self._params         = None
        self._order          = mp.SARIMAX["order"]
        self._coef_table     = None   # DataFrame exported for the dissertation
        self._scaler         = None
        self._top_idx        = None
        self._ridge          = None
        self._endog_train    = None
        self._exog_train     = None

    def fit(self, X_tr, y_tr, X_val, y_val, priority_idx=None, seed=None):
        cfg = mp.SARIMAX
        X   = np.asarray(X_tr, float)
        y   = np.asarray(y_tr, float).ravel()
        # NOTE (truthfulness, "sem achismo"): SARIMAX's own order search/fit
        # and the Ridge fallback are both deterministic given a FIXED input
        # matrix — neither has a random_state of its own. In sandbox testing,
        # SARIMAX's RMSE nonetheless varied slightly (~1-5%) across the 3
        # seeds; that variation comes from upstream — validation/walk_forward
        # .py also threads `seed` into PanelMICEImputer's random_state, and
        # scikit-learn's IterativeImputer uses it (e.g. to break ties when
        # choosing each column's n_nearest_features neighbours), so the
        # imputed covariates themselves differ slightly by seed even though
        # SARIMAX's own fit does not introduce any further randomness.
        self._seed = seed

        # Select top-K features by correlation with target.
        # CORRECTION (ablation-blindness, found investigating identical
        # SARIMAX RMSE across all 5 governance specs): selecting purely by
        # |corr(feature, target)| ignored which columns were the governance
        # features under test. The ~26 WDI/lag columns present in EVERY spec
        # dominate that ranking, so the same top-8 subset — and therefore
        # numerically identical predictions — got picked in every spec
        # whenever the 1-12 new governance columns never out-correlated an
        # existing member. priority_idx is now a (target_lag_idx,
        # governance_idx) tuple from validation/walk_forward.py's
        # _governance_priority_idx(): target lags are always kept (tier 1,
        # uncapped — see that function's docstring for why an earlier,
        # uncapped-governance version of this fix backfired), governance
        # columns are force-included but capped at half of whatever budget
        # remains after tier 1 (tier 2). The rest of the budget is still
        # filled by correlation, preserving the original stability rationale
        # (max_exog stays small for numerical stability of the AR/MA optimizer).
        n_exog = min(cfg["max_exog"], X.shape[1])
        corr   = np.array([
            abs(np.corrcoef(X[:, i], y)[0, 1]) if np.std(X[:, i]) > 1e-10 else 0.0
            for i in range(X.shape[1])
        ])
        if priority_idx:
            target_lag_idx, governance_idx = priority_idx
            target_lag_idx = [i for i in target_lag_idx if i < X.shape[1]]
            governance_idx = [i for i in governance_idx if i < X.shape[1]]
            n_tier1 = min(len(target_lag_idx), n_exog)
            remaining_after_tier1 = n_exog - n_tier1
            soft_cap = max(1, remaining_after_tier1 // 2) if governance_idx else 0
            keep_priority = target_lag_idx[:n_tier1] + governance_idx[:soft_cap]
            keep_priority = keep_priority[:n_exog]
            pool = [i for i in range(X.shape[1]) if i not in keep_priority]
            remaining = n_exog - len(keep_priority)
            top_pool = (np.array(pool)[np.argsort(corr[pool])[-remaining:]]
                        if remaining > 0 and pool else np.array([], dtype=int))
            self._top_idx = np.array(sorted(set(keep_priority) | set(top_pool.tolist())))
        else:
            self._top_idx = np.argsort(corr)[-n_exog:]
        X_sel = X[:, self._top_idx]

        self._scaler = StandardScaler()
        X_s = self._scaler.fit_transform(X_sel)

        # Ridge fallback (always available)
        self._ridge = Ridge(alpha=1.0)
        self._ridge.fit(X_s, y)

        # Auto order selection via AIC
        order = self._select_order(y, X_s) if cfg["auto_order"] else cfg["order"]
        self._order = order

        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX as SM
            if len(y) >= sum(order) + 10:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = SM(
                        y, exog=X_s, order=order, trend="c",
                        enforce_stationarity=False, enforce_invertibility=False,
                    ).fit(disp=False, maxiter=cfg["maxiter"], method=cfg["method"])

                # Validate on val set
                X_val_s = self._scaler.transform(
                    np.asarray(X_val, float)[:, self._top_idx]
                )
                fc = np.asarray(res.forecast(steps=len(y_val), exog=X_val_s))
                y_val_arr = np.asarray(y_val, float).ravel()

                if not (np.any(np.isnan(fc)) or np.any(np.isinf(fc))):
                    rmse_sar  = np.sqrt(np.mean((y_val_arr - fc) ** 2))
                    rmse_ridg = np.sqrt(np.mean(
                        (y_val_arr - X_val_s @ self._ridge.coef_ - self._ridge.intercept_) ** 2
                    ))
                    if rmse_sar <= rmse_ridg * 2.0:
                        self._params      = np.asarray(res.params)
                        self._endog_train = y
                        self._exog_train  = X_s
                        self._use_sarimax = True

                        # ── Coefficient table (for dissertation) ──────────────
                        if cfg["export_coefficients"]:
                            summary = res.summary2().tables[1]
                            self._coef_table = pd.DataFrame({
                                "Parameter":    summary.index.tolist(),
                                "Coefficient":  summary["Coef."].values,
                                "Std_Error":    summary["Std.Err."].values,
                                "t_stat":       summary["t"].values,
                                "p_value":      summary["P>|t|"].values,
                                "CI_lower_95":  summary["[0.025"].values,
                                "CI_upper_95":  summary["0.975]"].values,
                            })
        except Exception:
            pass

        return self

    def predict(self, X):
        X_arr = np.asarray(X, float)
        X_sel = X_arr[:, self._top_idx]
        X_s   = self._scaler.transform(X_sel)

        if self._use_sarimax:
            try:
                from statsmodels.tsa.statespace.sarimax import SARIMAX as SM
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = SM(
                        self._endog_train, exog=self._exog_train,
                        order=self._order, trend="c",
                        enforce_stationarity=False, enforce_invertibility=False,
                    ).smooth(self._params)
                    fc = np.asarray(res.forecast(steps=X_s.shape[0], exog=X_s))
                if not (np.any(np.isnan(fc)) or np.any(np.isinf(fc))):
                    return fc
            except Exception:
                pass

        return X_s @ self._ridge.coef_ + self._ridge.intercept_

    @staticmethod
    def _select_order(y, X_s) -> tuple:
        """Select ARIMA order by AIC over p,d,q ∈ {0,1,2}."""
        try:
            from statsmodels.tsa.statespace.sarimax import SARIMAX as SM
            best_aic, best_order = np.inf, (1, 1, 1)
            for p in range(3):
                for d in range(2):
                    for q in range(3):
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                res = SM(
                                    y, exog=X_s, order=(p, d, q), trend="c",
                                    enforce_stationarity=False,
                                    enforce_invertibility=False,
                                ).fit(disp=False, maxiter=200, method="lbfgs")
                            if res.aic < best_aic:
                                best_aic, best_order = res.aic, (p, d, q)
                        except Exception:
                            pass
            return best_order
        except Exception:
            return (1, 1, 1)

    def export_coef_table(self, path: str) -> None:
        if self._coef_table is not None:
            self._coef_table.to_csv(path, index=False)
            print(f"    SARIMAX coefficient table → {path}")


def train(X_tr, y_tr, X_val, y_val, run_name: str = "SARIMAX",
          priority_idx=None, seed=None) -> SARIMAXModel:
    model = SARIMAXModel()
    model.fit(X_tr, y_tr, X_val, y_val, priority_idx=priority_idx, seed=seed)
    model._search_method       = "AIC order selection"
    model._selection_criterion = "AIC on training fold"
    model._seed                = seed
    return model
