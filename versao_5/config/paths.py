"""config/paths.py — Filesystem paths for the pipeline."""
import os
import sys

_IN_COLAB = "google.colab" in sys.modules

if _IN_COLAB:

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

def horizon_dir(base_dir: str, horizon: int) -> str:
    """Return base_dir/h{horizon}/, creating it if needed."""
    d = os.path.join(base_dir, f"h{horizon}")
    os.makedirs(d, exist_ok=True)
    return d
