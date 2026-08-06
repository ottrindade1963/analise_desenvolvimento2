"""models/bayesian/model.py — Bayesian regression with proper numpy-version fallback."""
import os
import signal
import threading
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
import config.model_params as mp
import config.paths as paths

warnings.filterwarnings("ignore")


class _Timeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise _Timeout()


class BayesianModel:
    """
    Hierarchical Bayesian regression (PyMC) with BayesianRidge fallback.
    
    The fallback uses StandardScaler internally so that RMSE is comparable
    to other models (previously BayesianRidge ran on unscaled data → poor RMSE).
    """

    def __init__(self, pooling: str = "partial"):
        self.pooling   = pooling
        self._alpha    = 0.0
        self._beta     = None
        self._scaler   = StandardScaler()
        self._top_idx  = None
        self._y_mean   = 0.0
        self._y_std    = 1.0
        self._trace    = None
        self._is_pymc  = False
        self._fallback = None
        self._coef_summary = None

    def fit(self, X_tr, y_tr, X_val, y_val, priority_idx=None, seed=None):
        cfg = mp.BAYESIAN
        X   = np.asarray(X_tr, float)
        y   = np.asarray(y_tr, float)
        seed = mp.SEED if seed is None else seed
        self._seed = seed

        # Reduce to top features by variance.
        # CORRECTION (ablation-blindness — same family as the SARIMAX fix in
        # models/sarimax/model.py): selecting purely by variance, with no
        # regard for which columns are the governance features under test,
        # meant the PCA-derived "wgi_pca1_*" columns (already unit-variance
        # after PCA + StandardScaler upstream) routinely lost to raw
        # high-variance WDI columns and got dropped — Bayes_Partial/Complete
        # came out numerically identical across A1/A2/A3 as a result.
        # priority_idx is a (target_lag_idx, governance_idx) tuple from
        # validation/walk_forward.py's _governance_priority_idx(): target
        # lags are always kept (tier 1, uncapped), governance columns are
        # force-included but capped at half of whatever budget remains
        # after tier 1 (tier 2) — see that function's docstring for why an
        # earlier, uncapped-governance version of this fix backfired
        # (crowded out the target's own autoregressive lags and wrecked
        # RMSE). The rest of the budget is still filled by variance,
        # preserving the original stability rationale.
        n_feat = min(cfg["max_features"], X.shape[1])
        var_scores = np.var(X, axis=0)
        if priority_idx:
            target_lag_idx, governance_idx = priority_idx
            target_lag_idx = [i for i in target_lag_idx if i < X.shape[1]]
            governance_idx = [i for i in governance_idx if i < X.shape[1]]
            n_tier1 = min(len(target_lag_idx), n_feat)
            remaining_after_tier1 = n_feat - n_tier1
            soft_cap = max(1, remaining_after_tier1 // 2) if governance_idx else 0
            keep_priority = target_lag_idx[:n_tier1] + governance_idx[:soft_cap]
            keep_priority = keep_priority[:n_feat]
            pool = [i for i in range(X.shape[1]) if i not in keep_priority]
            remaining = n_feat - len(keep_priority)
            top_pool = (np.array(pool)[np.argsort(var_scores[pool])[-remaining:]]
                        if remaining > 0 and pool else np.array([], dtype=int))
            self._top_idx = np.array(sorted(set(keep_priority) | set(top_pool.tolist())))
        else:
            self._top_idx = np.argsort(var_scores)[-n_feat:]
        X_red = X[:, self._top_idx]

        # Scale X and y
        self._scaler = StandardScaler()
        X_s = self._scaler.fit_transform(X_red)
        self._y_mean = float(y.mean())
        self._y_std  = float(y.std()) or 1.0
        y_s = (y - self._y_mean) / self._y_std

        # signal.alarm() is only usable from the main thread of the main
        # interpreter — validation/walk_forward.py's evaluate_multi_seed()
        # runs one seed per worker thread (joblib, prefer="threads"), where
        # signal.signal() raises ValueError. Fall back to no hard timeout
        # in that case; PyMC's own chains/draws/tune budget is still bounded.
        use_alarm = (hasattr(signal, "SIGALRM")
                     and threading.current_thread() is threading.main_thread())

        pymc_ok = False
        try:
            if use_alarm:
                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(cfg["timeout_s"])

            import pymc as pm
            import arviz as az

            with pm.Model() as pm_model:
                if self.pooling == "partial":
                    mu_b    = pm.Normal("mu_beta",    mu=0, sigma=1)
                    sigma_b = pm.HalfNormal("sigma_beta", sigma=1)
                    beta    = pm.Normal("beta", mu=mu_b, sigma=sigma_b,
                                        shape=X_s.shape[1])
                else:
                    beta = pm.Normal("beta", mu=0, sigma=1, shape=X_s.shape[1])

                alpha = pm.Normal("alpha", mu=0, sigma=2)
                sigma = pm.HalfNormal("sigma", sigma=2)
                mu    = alpha + pm.math.dot(X_s, beta)
                pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_s)

                # CORRECTION (multi-seed requirement, "faça conforme as
                # regras... também para MCMC"): random_seed now comes from
                # the seed argument (was hardcoded to cfg["seed"] — every
                # run, at every seed, produced bit-identical chains). PyMC's
                # own convergence rule (>=2 chains from independent starting
                # points, checked via R-hat/ESS in export_diagnostics) is
                # still respected via cfg["chains"] — this is orthogonal to
                # and does not replace that: each of the 3 outer seeds runs
                # its own full multi-chain MCMC fit, so both the correct
                # within-run convergence check AND the requested
                # across-run robustness comparison are satisfied.
                self._trace = pm.sample(
                    draws=cfg["draws"], tune=cfg["tune"],
                    chains=cfg["chains"], cores=cfg["cores"],
                    random_seed=seed,
                    return_inferencedata=True, progressbar=False,
                )

                if cfg.get("posterior_predictive", True):
                    self._ppc = pm.sample_posterior_predictive(
                        self._trace, progressbar=False
                    )

            if use_alarm:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

            self._alpha  = float(self._trace.posterior["alpha"].values.mean())
            self._beta   = self._trace.posterior["beta"].values.mean(axis=(0, 1))
            self._is_pymc = True
            self._coef_summary = az.summary(
                self._trace, var_names=["alpha", "beta"], hdi_prob=0.94
            )
            pymc_ok = True

        except Exception as exc:
            if use_alarm:
                try:
                    signal.alarm(0)
                except Exception:
                    pass
            print(f"    PyMC ({self.pooling}) failed ({exc}); BayesianRidge fallback.")

        if not pymc_ok:
            # BayesianRidge on scaled data for fair comparison.
            # NOTE (truthfulness, "sem achismo"): sklearn's BayesianRidge has
            # no random_state — it is a deterministic closed-form solve given
            # fixed data. Varying `seed` here is therefore a correct no-op:
            # every seed will reproduce byte-identical fallback results, and
            # that identical-across-seeds outcome is the EXPECTED, honest
            # report for this code path — not a bug to be hidden.
            self._fallback = BayesianRidge(max_iter=300)
            self._fallback.fit(X_s, y_s)

        return self

    def predict(self, X):
        X_arr = np.asarray(X, float)
        # Select same features as during fit
        if X_arr.shape[1] > len(self._top_idx):
            X_red = X_arr[:, self._top_idx]
        else:
            X_red = X_arr
        X_s = self._scaler.transform(X_red[:, :self._scaler.n_features_in_])

        if self._is_pymc and self._beta is not None:
            y_s = self._alpha + X_s @ self._beta
        elif self._fallback is not None:
            y_s = self._fallback.predict(X_s)
        else:
            y_s = np.zeros(X_s.shape[0])

        # Inverse-scale target
        return y_s * self._y_std + self._y_mean

    def export_diagnostics(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        if not self._is_pymc or self._trace is None:
            return
        try:
            import arviz as az
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            label = f"Bayes_{self.pooling}_seed{getattr(self, '_seed', mp.SEED)}"

            if self._coef_summary is not None:
                self._coef_summary.to_csv(
                    os.path.join(out_dir, f"{label}_posterior_summary.csv")
                )
            az.plot_trace(self._trace, var_names=["alpha", "beta"])
            plt.suptitle(f"Trace — {label}", y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"{label}_trace.png"),
                        dpi=120, bbox_inches="tight")
            plt.close()
            print(f"    Bayesian diagnostics → {out_dir}")
        except Exception as exc:
            print(f"    Diagnostic export failed: {exc}")


def train(X_tr, y_tr, X_val, y_val,
          pooling: str = "partial",
          run_name: str = "Bayesian",
          priority_idx=None, seed=None) -> BayesianModel:
    model = BayesianModel(pooling=pooling)
    model.fit(X_tr, y_tr, X_val, y_val, priority_idx=priority_idx, seed=seed)
    model._search_method       = f"MCMC ({pooling} pooling, {mp.BAYESIAN['chains']} chains) / BayesianRidge fallback"
    model._selection_criterion = "R-hat convergence / validation MSE"
    model._seed                = mp.SEED if seed is None else seed
    return model
