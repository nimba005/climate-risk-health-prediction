"""Project-wide paths and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
SYNTHETIC_DIR = ROOT / "data" / "synthetic"
MODELS_DIR = ROOT / "models"
SUBMISSIONS_DIR = ROOT / "submissions"
REPORTS_DIR = ROOT / "reports"

SEED = 42
N_FOLDS = 5

# Competition rules: binary predictions must use the default 0.5 threshold.
THRESHOLD = 0.5

# Leaderboard score = 60% F1 (on TargetF1) + 40% ROC-AUC (on TargetRAUC).
F1_WEIGHT = 0.6
AUC_WEIGHT = 0.4

# Categories seen fewer times than this (train + test combined) are merged into one bucket.
RARE_CATEGORY_MIN_COUNT = 5

# A single feature with CV AUC above this on its own is flagged as possible target leakage.
LEAKAGE_AUC_WARNING = 0.98

ALL_MODELS = ("lgbm", "catboost", "xgb", "logreg")
