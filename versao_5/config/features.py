"""config/features.py — Feature engineering and ablation settings."""

# ── Temporal features ─────────────────────────────────────────────────────────
# All computed INSIDE each fold — never on the full panel (no look-ahead bias).
LAGS_WGI        = [1, 2]   # WGI/PCA lags
LAGS_WDI        = [1]      # Economic indicator lags
LAGS_TARGET     = [1, 2]   # Autoregressive lags of the target
ROLLING_WINDOW  = 3        # Rolling-mean window

# ── PCA ───────────────────────────────────────────────────────────────────────
PCA_TRAIN_FRAC   = 0.80    # Fraction of each country's series used to FIT PCA
PCA_N_COMPONENTS = 3       # Components extracted (only PC1 used as feature)

# ── Ablation specifications ───────────────────────────────────────────────────
# DATA: all specs use the same INNER JOIN dataset (WDI ∩ WGI by country+year).
#       "INNER" refers to the data merge strategy, already applied in df_raw.
#       "inter" below means *interaction terms* (WGI × economic variables).
#
# Each spec controls which governance channel enters the feature set:
ABLATION_SPECS = {
    # A1: Baseline — WDI only, governance variables excluded from features.
    #     Uses the INNER JOIN dataset but ignores WGI columns.
    #     Answers: what is the no-governance benchmark?
    "A1_WDI_only":       {"wgi_pca": False, "wgi_raw": False, "interactions": False},

    # A2: WDI + single latent governance factor (PC1 from 6 WGI via PCA).
    #     Answers: does a compressed governance index help?
    "A2_WDI_PCA1":       {"wgi_pca": True,  "wgi_raw": False, "interactions": False},

    # A3: WDI + all 6 raw WGI indicators (no compression).
    #     Answers: are individual governance dimensions informative?
    "A3_WDI_6WGI":       {"wgi_pca": False, "wgi_raw": True,  "interactions": False},

    # A4: WDI + PCA governance factor + interaction terms (PCA × economic vars).
    #     Answers: does governance moderate economic channels?
    #     NOTE: previous WDI_plus_inter and WDI_PCA_inter were identical — merged here.
    "A4_WDI_PCA_inter":  {"wgi_pca": True,  "wgi_raw": False, "interactions": True},

    # A5: WDI + 6 raw WGI + interaction terms — most complete governance specification.
    #     Answers: full governance specification with moderation effects.
    "A5_WDI_6WGI_inter": {"wgi_pca": False, "wgi_raw": True,  "interactions": True},
}

# ── Forecast horizons ─────────────────────────────────────────────────────────
# NEW in versão_5: this constant existed in the original project too, but
# was never referenced anywhere — every prediction there was contemporaneous
# (same-year covariates and label), not a genuine forward forecast (see
# relatorio1's horizon section). Here it is actually wired in:
# validation/walk_forward.py::build_horizon_target() shifts the label
# `horizon` calendar years into the future (year-exact, not positional —
# see preprocessing/temporal.py), and pipeline.py::run_pipeline() loops
# over every horizon below, evaluating every spec×model combination once
# per horizon.
#
# CORRECTION (user request — "deve fazer a previsão para ano t, t+1, t+2,
# t+3, t+4 e t+5", clarified afterward — "eu quero previsão genuína. Ele
# tem que prever até 5 anos à frente"): extended from the original {1, 2}
# to {1, 2, 3, 4, 5} — every horizon here is a GENUINE forward forecast
# (target `horizon` calendar years after each row's own year). h=0
# (contemporaneous/nowcasting — predicting the target from covariates of
# its own year) is deliberately NOT included: that is what the original
# project (versão_4) already does, and is explicitly not the point of
# versão_5. h=1 remains the "primary" horizon that explainability/ablation
# default to (see pipeline.py::PRIMARY_HORIZON).
FORECAST_HORIZONS = [1, 2, 3, 4, 5]  # t+1, t+2, t+3, t+4, t+5 — genuine forecasts only
