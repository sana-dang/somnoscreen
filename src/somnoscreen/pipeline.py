"""Model definitions and the leak-free training pipeline.

`build_pipeline` returns ONE object: imputer -> scaler -> (optional oversampler) -> model.
Because every preprocessing step lives inside the pipeline, cross-validation fits them on the
training folds only, and the saved model fills blank app answers exactly as training did.
"""

from __future__ import annotations

from typing import Any

from imblearn.over_sampling import SMOTE, SMOTENC
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from somnoscreen.config import RANDOM_STATE

IMBALANCE_STRATEGIES = ("smote", "smotenc", "class_weight")

MODEL_NAMES = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost",
    "SVM",
    "LightGBM",
    "Gradient Boosting",
    "KNN",
]

# Hyper-parameter grids, identical to the thesis. The three boosting models share one grid.
_BOOST_GRID = {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]}
PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "Logistic Regression": {"C": [0.01, 0.1, 1, 10, 100]},
    "Random Forest": {"n_estimators": [100, 200], "max_depth": [5, 10, None], "min_samples_leaf": [1, 5]},
    "XGBoost": _BOOST_GRID,
    "SVM": {"C": [0.1, 1, 10], "kernel": ["rbf"]},
    "LightGBM": _BOOST_GRID,
    "Gradient Boosting": _BOOST_GRID,
    "KNN": {"n_neighbors": [5, 9, 15, 21], "weights": ["uniform", "distance"]},
}


class BalancedGradientBoostingClassifier(GradientBoostingClassifier):
    """Gradient boosting that gives both classes equal total weight.

    scikit-learn's GradientBoostingClassifier has no `class_weight` option, so we pass
    'balanced' sample weights to `fit` instead (the same weights class_weight='balanced' gives).
    """

    def fit(self, X, y, sample_weight=None):  # noqa: D102
        return super().fit(X, y, sample_weight=compute_sample_weight("balanced", y))


class BalancedXGBClassifier(XGBClassifier):
    """XGBoost with 'balanced' sample weights (same idea as BalancedGradientBoostingClassifier)."""

    def fit(self, X, y, **kwargs):  # noqa: D102
        kwargs["sample_weight"] = compute_sample_weight("balanced", y)
        return super().fit(X, y, **kwargs)


def supports_class_weight(name: str) -> bool:
    """KNN has no notion of sample weights, so 'class_weight' does not apply to it."""
    return name != "KNN"


def make_model(name: str, imbalance_strategy: str = "smote") -> BaseEstimator:
    """Create one of the 7 classifiers with the thesis settings.

    With imbalance_strategy='class_weight' the model itself weights the rare class up
    (instead of creating synthetic rows with SMOTE).
    """
    balanced = imbalance_strategy == "class_weight"
    if balanced and not supports_class_weight(name):
        raise ValueError(f"{name} does not support class weights")
    weight = "balanced" if balanced else None

    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight=weight)
    if name == "Random Forest":
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1, class_weight=weight)
    if name == "XGBoost":
        xgb = BalancedXGBClassifier if balanced else XGBClassifier
        return xgb(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1)
    if name == "SVM":
        return SVC(probability=True, random_state=RANDOM_STATE, class_weight=weight)
    if name == "LightGBM":
        return LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, n_jobs=1, class_weight=weight)
    if name == "Gradient Boosting":
        gb = BalancedGradientBoostingClassifier if balanced else GradientBoostingClassifier
        return gb(random_state=RANDOM_STATE)
    if name == "KNN":
        return KNeighborsClassifier()
    raise ValueError(f"Unknown model: {name}")


def build_pipeline(
    model: BaseEstimator,
    imbalance_strategy: str = "smote",
    categorical_columns: list[int] | None = None,
    first_step: tuple[str, BaseEstimator] | None = None,
) -> Pipeline:
    """imputer -> scaler -> oversampler (if any) -> model, as one imblearn Pipeline.

    - "smote": SMOTE creates synthetic "Yes" rows (thesis setting).
    - "smotenc": like SMOTE, but `categorical_columns` (column positions) get a real category
      value instead of an in-between number.
    - "class_weight": no synthetic rows; the model must already be built with class weights.

    The oversampler only runs during `fit`, never during `predict`. `first_step` lets Approach B
    put the cluster-label step in front of the imputer.
    """
    steps: list[tuple[str, Any]] = []
    if first_step is not None:
        steps.append(first_step)
    steps += [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]

    if imbalance_strategy == "smote":
        steps.append(("sampler", SMOTE(random_state=RANDOM_STATE)))
    elif imbalance_strategy == "smotenc":
        if not categorical_columns:
            raise ValueError("SMOTENC needs the positions of the categorical columns")
        sampler = SMOTENC(categorical_features=categorical_columns, random_state=RANDOM_STATE)
        steps.append(("sampler", sampler))
    elif imbalance_strategy != "class_weight":
        raise ValueError(f"Unknown imbalance strategy: {imbalance_strategy}")

    steps.append(("model", clone(model)))
    return Pipeline(steps)


def grid_for_pipeline(name: str) -> dict[str, list[Any]]:
    """Prefix the model's grid with 'model__' so GridSearchCV can tune it inside the pipeline."""
    return {f"model__{key}": values for key, values in PARAM_GRIDS[name].items()}
