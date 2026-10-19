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
