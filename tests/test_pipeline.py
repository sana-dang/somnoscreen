"""Tests for the pipeline, clustering and evaluation helpers (synthetic data, no NHANES needed)."""

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score

from somnoscreen.clustering import PerClusterClassifier, WardClusterer, cluster_columns
from somnoscreen.evaluate import corrected_paired_ttest, threshold_for_recall
from somnoscreen.features import CATEGORICAL_FEATURES, FEATURE_NAMES, FEATURES_BY_NAME
from somnoscreen.pipeline import build_pipeline, make_model


def synthetic(n: int = 300, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """Random rows that look like the decoded data (categories take only valid values)."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({name: rng.normal(50, 10, n) for name in FEATURE_NAMES})
    for name in CATEGORICAL_FEATURES:
        X[name] = rng.choice(list(FEATURES_BY_NAME[name].value_map.values()), n).astype(float)
    X.loc[::7, "waist"] = np.nan  # some missing values
    y = pd.Series((X["phq_score"] + rng.normal(0, 10, n) > 55).astype(int))
    return X, y


def test_pipeline_handles_missing_values_and_returns_probabilities():
    X, y = synthetic()
    model = build_pipeline(make_model("Logistic Regression"), "smote").fit(X, y)
    p = model.predict_proba(X)[:, 1]
    assert ((p >= 0) & (p <= 1)).all()


def test_smotenc_keeps_categories_valid():
    X, y = synthetic()
    positions = [FEATURE_NAMES.index(c) for c in CATEGORICAL_FEATURES]
    pipe = build_pipeline(make_model("Logistic Regression"), "smotenc", positions)
    Xt = pipe[:-2].fit_transform(X, y)  # imputer + scaler
    X_res, _ = pipe.named_steps["sampler"].fit_resample(Xt, y)
    for pos in positions:
        assert set(np.round(X_res[:, pos], 8)) <= set(np.round(Xt[:, pos], 8))


def test_class_weight_models_fit():
    X, y = synthetic()
    for name in ["Logistic Regression", "XGBoost", "Gradient Boosting"]:
        build_pipeline(make_model(name, "class_weight"), "class_weight").fit(X, y)


def test_ward_clusterer_one_hot():
    X, _ = synthetic()
    out = WardClusterer(4).fit_transform(X)
    assert list(out.columns[-4:]) == cluster_columns(4)
    assert (out[cluster_columns(4)].sum(axis=1) == 1).all()


def test_per_cluster_classifier_predicts_every_row():
    X, y = synthetic(400)
    model = PerClusterClassifier(build_pipeline(make_model("Logistic Regression"), "smote"), 2).fit(X, y)
    p = model.predict_proba(X)
    assert p.shape == (400, 2)
    assert np.allclose(p.sum(axis=1), 1)


def test_threshold_reaches_target_recall():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    probs = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
    t = threshold_for_recall(y, probs, 0.7)
    assert recall_score(y, probs >= t) >= 0.7
    assert recall_score(y, probs >= t + 1e-6) < 0.7 or t == probs.max()


def test_corrected_ttest_is_more_cautious_than_plain():
    diffs = np.array([0.01, 0.02, 0.015, 0.012, 0.018] * 5)
    _, p_corrected = corrected_paired_ttest(diffs, 3727, 932)
    from scipy import stats

    assert p_corrected > stats.ttest_1samp(diffs, 0).pvalue
