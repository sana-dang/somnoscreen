"""Fit the final SomnoScreen model on ALL 2017-18 rows and save the artifacts the app uses.

The model choice, imbalance strategy, hyper-parameters and calibration come from the Phase 2
evaluation (results/final_model_evaluation.json); nothing is re-decided here. The held-out test
metrics in metadata.json are the honest estimate of performance: the final model itself has seen
every row, so it cannot be tested again.

Run: python -m somnoscreen.train
"""

from __future__ import annotations

import json
import platform
from datetime import date

import imblearn
import joblib
import numpy as np
import pandas as pd
import shap
import sklearn
from sklearn.base import clone

from somnoscreen import config
from somnoscreen.data import load_dataset
from somnoscreen.evaluate import (
    calibrated,
    final_base_estimator,
    oof_probabilities,
    save_json,
    screening_metrics,
    threshold_for_recall,
)
from somnoscreen.features import FEATURE_NAMES, FEATURES, TARGET

HIGH_RISK_CUTOFF = 0.50  # "High" = the model estimates a better-than-even chance


def risk_bands(threshold: float) -> dict[str, list[float]]:
    """Low: below the screening threshold. Moderate: flagged, but under 50%. High: 50% or more."""
    if not 0 < threshold < HIGH_RISK_CUTOFF:
        raise ValueError(f"Screening threshold {threshold:.3f} must lie between 0 and {HIGH_RISK_CUTOFF}")
    return {"Low": [0.0, threshold], "Moderate": [threshold, HIGH_RISK_CUTOFF],
            "High": [HIGH_RISK_CUTOFF, 1.0]}


def library_versions() -> dict[str, str]:
    """Versions needed to load the saved model safely."""
    import lightgbm
    import xgboost

    return {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
            "imbalanced-learn": imblearn.__version__, "numpy": np.__version__, "pandas": pd.__version__,
            "xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__, "shap": shap.__version__,
            "joblib": joblib.__version__}


def main() -> None:
    """Train on all rows, pick the deployed threshold from out-of-fold predictions, save artifacts."""
    evaluation = json.loads((config.RESULTS_DIR / "final_model_evaluation.json").read_text())
    selection = evaluation["selection"]

    df = load_dataset()
    X, y = df[FEATURE_NAMES], df[TARGET]
    final = calibrated(final_base_estimator(selection), selection["calibration"])

    # Deployed threshold: same rule as in evaluation (target recall on out-of-fold predictions),
    # now using all rows because the deployed model is trained on all rows.
    oof = oof_probabilities(final, X, y)
    threshold = threshold_for_recall(y, oof)
    oof_metrics = screening_metrics(y, oof, threshold)

    model = clone(final).fit(X, y)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)

    # Background sample for SHAP ("the average person" the app compares against).
    X.sample(100, random_state=config.RANDOM_STATE).to_csv(config.SHAP_BACKGROUND_PATH, index=False)

    metadata = {
        "name": "SomnoScreen",
        "created": date.today().isoformat(),
        "target": "Self-reported: ever told a doctor you had trouble sleeping (NHANES SLQ050). "
                  "Not a clinical diagnosis.",
        "data": "NHANES 2017-2018, adults who answered SLQ050 Yes/No",
        "n_rows": len(df),
        "base_rate": float(y.mean()),
        "model": selection["model"],
        "imbalance_strategy": selection["imbalance_strategy"],
        "hyperparameters": selection["params"],
        "calibration": selection["calibration"],
        "feature_order": FEATURE_NAMES,
        "feature_labels": {f.name: f.label for f in FEATURES},
        "target_recall": config.TARGET_RECALL,
        "screening_threshold": threshold,
        "risk_bands": risk_bands(threshold),
        "metrics": {
            "held_out_test": evaluation["test_at_threshold"],
            "held_out_test_threshold": evaluation["screening_threshold"],
            "held_out_test_95ci": evaluation["test_bootstrap_95ci"],
            "cv_5x5_training": evaluation["cv_5x5_final_model"],
            "all_data_out_of_fold_at_deployed_threshold": oof_metrics,
        },
        "library_versions": library_versions(),
    }
    save_json(metadata, config.METADATA_PATH)
    print(f"Saved {config.MODEL_PATH.name} and {config.METADATA_PATH.name}")
    print(f"Deployed screening threshold {threshold:.4f}; out-of-fold recall {oof_metrics['Recall']:.3f}, "
          f"precision {oof_metrics['Precision']:.3f}")


if __name__ == "__main__":
    main()
