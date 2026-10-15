"""Project-wide settings: paths, random seed and evaluation settings.

Every other module imports from here, so a setting is changed in exactly one place.
"""

from pathlib import Path

# ---- Paths (all relative to the repository root) -------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "2017-2018"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"

MODEL_PATH = MODELS_DIR / "somnoscreen.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"
SHAP_BACKGROUND_PATH = MODELS_DIR / "shap_background.csv"

# ---- Reproducibility ------------------------------------------------------------------------
RANDOM_STATE = 42

# ---- Train/test split (same as the thesis) ---------------------------------------------------
TEST_SIZE = 0.20

# ---- Cross-validation -----------------------------------------------------------------------
CV_FOLDS = 5  # folds used by GridSearchCV (as in the thesis)
CV_REPEATS = 5  # Phase 2: 5 folds x 5 repeats of stratified CV
N_JOBS = -1  # use all CPU cores

# ---- Clustering (thesis) --------------------------------------------------------------------
N_CLUSTERS = 4

# ---- Statistics -----------------------------------------------------------------------------
N_BOOTSTRAP = 1000

# ---- Screening ------------------------------------------------------------------------------
# The screening threshold is the highest probability cut-off that still catches at least this
# share of true "Yes" cases in out-of-fold TRAINING predictions. It is never tuned on the test set.
TARGET_RECALL = 0.70

# Tuning / selection metric for Phase 2 (PR-AUC = average precision).
PRIMARY_SCORING = "average_precision"

# Calibration is only used if it lowers the mean CV Brier score by at least this much.
MIN_BRIER_IMPROVEMENT = 0.005

# Final model rule: prefer Logistic Regression (interpretable) unless another model beats its mean
# CV PR-AUC by more than one standard deviation of Logistic Regression's CV PR-AUC.
PREFERRED_MODEL = "Logistic Regression"

# Age split used in the fairness check.
FAIRNESS_AGE_CUTOFF = 40
