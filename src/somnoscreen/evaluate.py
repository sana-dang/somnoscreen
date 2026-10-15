"""Evaluation: metrics, repeated CV, screening threshold, calibration, bootstrap, fairness.

`python -m somnoscreen.evaluate` runs the Phase 2 evaluation of the chosen final model:
model choice -> calibration check -> screening threshold -> one final check on the test set ->
fairness -> figures (ROC, PR, calibration, fairness, SHAP).
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict

from somnoscreen import config

warnings.filterwarnings("ignore")

# =============================================================================================
# Metrics
# =============================================================================================


def classification_metrics(y_true, preds, probs) -> dict[str, float]:
    """Threshold metrics (from `preds`) and ranking / probability metrics (from `probs`)."""
    return {
        "Accuracy": accuracy_score(y_true, preds),
        "Balanced_Acc": balanced_accuracy_score(y_true, preds),
        "Precision": precision_score(y_true, preds, zero_division=0),
        "Recall": recall_score(y_true, preds, zero_division=0),
        "F1": f1_score(y_true, preds, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, probs),
        "PR_AUC": average_precision_score(y_true, probs),
        "Brier": brier_score_loss(y_true, probs),
    }


def threshold_for_recall(y_true, probs, target_recall: float = config.TARGET_RECALL) -> float:
    """Highest cut-off whose recall is still >= target_recall.

    A higher cut-off flags fewer people (better precision), so we take the highest one that still
    catches the target share of true "Yes" cases.
    """
    _, recall, thresholds = precision_recall_curve(y_true, probs)
    # recall[i] is the recall when predicting "Yes" for probs >= thresholds[i]
    ok = recall[:-1] >= target_recall
    return float(thresholds[ok].max())


def screening_metrics(y_true, probs, threshold: float) -> dict[str, float]:
    """All metrics when people with probability >= threshold are flagged."""
    preds = (np.asarray(probs) >= threshold).astype(int)
    return {"Threshold": threshold, **classification_metrics(y_true, preds, probs)}


def fold_metrics(y_true, probs, target_recall: float = config.TARGET_RECALL) -> dict[str, float]:
    """Metrics for one CV fold.

    Threshold metrics use the default 0.5 cut-off (labelled '@0.5'), because a fold has no
    separate data to tune a cut-off on. 'Precision@Recall' is the precision at the cut-off that
    reaches the target recall in that fold (a point on the fold's precision-recall curve).
    """
    m = classification_metrics(y_true, (np.asarray(probs) >= 0.5).astype(int), probs)
    out = {k: m[k] for k in ["ROC_AUC", "PR_AUC", "Brier"]}
    out.update({f"{k}@0.5": m[k] for k in ["Accuracy", "Balanced_Acc", "Precision", "Recall", "F1"]})
    t = threshold_for_recall(y_true, probs, target_recall)
    out["Precision@Recall"] = precision_score(y_true, (np.asarray(probs) >= t).astype(int))
    return out


# =============================================================================================
# Repeated cross-validation
# =============================================================================================


def repeated_cv_splits(y: pd.Series) -> list[tuple[np.ndarray, np.ndarray]]:
    """5 folds x 5 repeats of stratified CV. Same seed -> identical folds for every model/approach."""
    rskf = RepeatedStratifiedKFold(
        n_splits=config.CV_FOLDS, n_repeats=config.CV_REPEATS, random_state=config.RANDOM_STATE
    )
    return list(rskf.split(np.zeros(len(y)), y))


def _fit_predict(estimator: BaseEstimator, X: pd.DataFrame, y: pd.Series, train, val) -> np.ndarray:
    model = clone(estimator).fit(X.iloc[train], y.iloc[train])
    return model.predict_proba(X.iloc[val])[:, 1]


def repeated_cv(estimator: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, np.ndarray]:
    """Evaluate `estimator` with 5x5 repeated stratified CV (folds run in parallel).

    Returns (one row of metrics per fold, out-of-fold probabilities with shape (repeats, rows)).
    Every preprocessing step is inside `estimator`, so it is re-fitted on each training fold.
    """
    splits = repeated_cv_splits(y)
    probs = Parallel(n_jobs=config.N_JOBS)(
        delayed(_fit_predict)(estimator, X, y, train, val) for train, val in splits
    )
    oof = np.full((config.CV_REPEATS, len(y)), np.nan)
    rows = []
    for i, ((_, val), p) in enumerate(zip(splits, probs, strict=True)):
        repeat, fold = divmod(i, config.CV_FOLDS)
        oof[repeat, val] = p
        rows.append({"Repeat": repeat, "Fold": fold, **fold_metrics(y.iloc[val], p)})
    return pd.DataFrame(rows), oof


def summarize_folds(folds: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Mean and standard deviation of every metric over the CV folds, per group of `keys`."""
    metrics = [c for c in folds.columns if c not in [*keys, "Repeat", "Fold"]]
    summary = folds.groupby(keys, sort=False)[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def corrected_paired_ttest(diffs: np.ndarray, n_train: int, n_test: int) -> tuple[float, float]:
    """Nadeau & Bengio corrected resampled t-test for repeated-CV differences.

    CV folds share training rows, so the plain t-test is over-confident; the correction inflates
    the variance by (1/k + n_test/n_train). Returns (t statistic, two-sided p-value).
    """
    k = len(diffs)
    var = np.var(diffs, ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = np.mean(diffs) / np.sqrt((1 / k + n_test / n_train) * var)
    return float(t), float(2 * stats.t.sf(abs(t), df=k - 1))


# =============================================================================================
# Bootstrap and fairness
# =============================================================================================


def bootstrap_indices(n_rows: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n_rows) matrix of row positions drawn with replacement (the thesis scheme)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_rows, size=(n_boot, n_rows))


def bootstrap_metric(y_true: np.ndarray, values: np.ndarray, boot_idx: np.ndarray, metric) -> np.ndarray:
    """Metric on every bootstrap resample. `values` are predictions or probabilities."""
    return np.array([metric(y_true[idx], values[idx]) for idx in boot_idx])


def paired_difference(diffs: np.ndarray, point: float) -> dict[str, float | str]:
    """95% percentile interval of a paired bootstrap difference and a plain-English verdict."""
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    verdict = "better" if lo > 0 else ("worse" if hi < 0 else "no clear difference")
    return {"Diff": point, "CI_low": lo, "CI_high": hi, "Verdict": verdict}


def group_metrics(y_true, preds, groups: pd.Series, group_type: str) -> list[dict]:
    """Accuracy / precision / recall / F1 for every value of `groups` (fairness check)."""
    y_true, preds = np.asarray(y_true), np.asarray(preds)
    rows = []
    for group in groups.unique():
        mask = (groups == group).to_numpy()
        rows.append({
            "Group_Type": group_type,
            "Group": group,
            "N": int(mask.sum()),
            "Positive_Rate": y_true[mask].mean(),
            "Flagged_Rate": preds[mask].mean(),
            "Accuracy": accuracy_score(y_true[mask], preds[mask]),
            "Precision": precision_score(y_true[mask], preds[mask], zero_division=0),
            "Recall": recall_score(y_true[mask], preds[mask], zero_division=0),
            "F1": f1_score(y_true[mask], preds[mask], zero_division=0),
        })
    return rows


def fairness_groups(df: pd.DataFrame) -> dict[str, pd.Series]:
    """Gender and age group (<40 / 40+) labels for the fairness check."""
    cut = config.FAIRNESS_AGE_CUTOFF
    return {
        "Gender": df["gender"].map({0: "Male", 1: "Female"}),
        "Age": pd.Series(np.where(df["age"] < cut, f"Under {cut}", f"{cut} and above"), index=df.index),
    }


# =============================================================================================
# Calibration
# =============================================================================================


def calibrated(estimator: BaseEstimator, method: str) -> BaseEstimator:
    """Wrap a pipeline so its probabilities are re-mapped to match observed rates.

    ensemble=False: one model is fitted on all training rows; the mapping (sigmoid or isotonic)
    is learned from 5-fold out-of-fold predictions, so it never sees its own training predictions.
    The oversampler inside the pipeline only touches the model's training rows, never the rows
    used to learn the mapping.
    """
    if method == "none":
        return clone(estimator)
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    return CalibratedClassifierCV(clone(estimator), method=method, cv=cv, ensemble=False)


def oof_probabilities(estimator: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """Out-of-fold probabilities from one 5-fold CV (used to choose the screening threshold)."""
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    probs = cross_val_predict(clone(estimator), X, y, cv=cv, method="predict_proba", n_jobs=config.N_JOBS)
    return probs[:, 1]


def save_json(obj: dict, path) -> None:
    """Write a dict as pretty JSON (numpy numbers converted to plain Python)."""
    def convert(o):
        if isinstance(o, np.generic):
            return o.item()
        raise TypeError(type(o))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=convert))
