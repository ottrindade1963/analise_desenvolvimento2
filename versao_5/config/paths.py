"""config/paths.py — Filesystem paths for the pipeline."""
import os
import sys

_IN_COLAB = "google.colab" in sys.modules

if _IN_COLAB:
    # Hardcode: the notebook clones this project's own GitHub repository —
    # "analise_desenvolvimento2" (versão_5), distinct from the original
    # project's repository ("analise_desenvolvimento", which holds
    # versão_4). `git clone` names the local directory after the repo by
    # default, so Colab will place it at /content/analise_desenvolvimento2.
    # If the repo is ever cloned under a different local name, update this
    # constant to match — the _candidates[0] auto-detection approach was
    # tried before and rejected as fragile (picks the wrong directory if
    # /content has any other entry sorting before it alphabetically).
    ROOT = "/content/analise_desenvolvimento2"
else:
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DIR            = os.path.join(ROOT, "data", "raw")
CLEAN_DIR          = os.path.join(ROOT, "data", "clean")
AGGREGATED_DIR     = os.path.join(ROOT, "data", "aggregated")
SYNTHETIC_DIR      = os.path.join(ROOT, "data", "synthetic")
FEATURES_DIR       = os.path.join(ROOT, "data", "features")
MODELS_DIR         = os.path.join(ROOT, "models", "artefacts")
TUNING_DIR         = os.path.join(ROOT, "tuning", "results")
EXPLAINABILITY_DIR = os.path.join(ROOT, "explainability", "results")
REPORTS_DIR        = os.path.join(ROOT, "reports")
FIGURES_DIR        = os.path.join(ROOT, "figures")
METADATA_DIR       = os.path.join(ROOT, "utils", "metadata")

# Drive path — consistent with what the notebook cells use
DRIVE_DIR = "/content/drive/MyDrive/africa_mo_pipeline/" if _IN_COLAB else None

for _d in [
    RAW_DIR, CLEAN_DIR, AGGREGATED_DIR, SYNTHETIC_DIR, FEATURES_DIR,
    MODELS_DIR, TUNING_DIR, EXPLAINABILITY_DIR,
    REPORTS_DIR, FIGURES_DIR, METADATA_DIR,
]:
    os.makedirs(_d, exist_ok=True)

if DRIVE_DIR:
    os.makedirs(DRIVE_DIR, exist_ok=True)


# CORRECTION (user request — "em cada ano de previsão tem que gerar os
# artefactos... de forma independente e salvar tudo"): every genuine
# forecast horizon (t+1, ..., t+5 — no contemporaneous/t case) now gets its
# own subdirectory under each artifact root
# (models/artefacts/h{H}/, tuning/results/h{H}/, reports/h{H}/,
# figures/h{H}/, explainability/results/h{H}/), so a model, hyperparameter
# table, results table or figure never collides across horizons and each
# horizon's full output set can be inspected/shipped on its own.
def horizon_dir(base_dir: str, horizon: int) -> str:
    """Return base_dir/h{horizon}/, creating it if needed."""
    d = os.path.join(base_dir, f"h{horizon}")
    os.makedirs(d, exist_ok=True)
    return d
