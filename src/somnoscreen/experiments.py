"""Thesis research question: does clustering before classification improve prediction?

    A: direct classification on the 19 base features
    B: base features + one-hot Ward cluster label (k=4)
    C: a separate model trained inside each cluster

Two protocols live here:

- `reproduce`: the ORIGINAL thesis protocol, ported unchanged (accuracy as tuning metric, a
  decision threshold tuned for accuracy, single 80/20 split, paired bootstrap on the test set).
  Its job is to prove the port is faithful (results/reproduction_check.csv).
- `compare` (Phase 2): the same question under the stronger evaluation (PR-AUC tuning,
  repeated CV, imbalance strategies, screening threshold). See `run_phase2`.

Run:  python -m somnoscreen.experiments reproduce
      python -m somnoscreen.experiments compare
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

from somnoscreen import config
from somnoscreen.clustering import PerClusterClassifier, WardClusterer, cluster_columns
from somnoscreen.data import load_dataset, split_train_test
from somnoscreen.evaluate import (
    bootstrap_indices,
    bootstrap_metric,
    classification_metrics,
    corrected_paired_ttest,
    group_metrics,
    oof_probabilities,
    paired_difference,
    repeated_cv,
    save_json,
    screening_metrics,
    summarize_folds,
    threshold_for_recall,
)
from somnoscreen.features import CATEGORICAL_FEATURES, FEATURE_NAMES, TARGET
from somnoscreen.pipeline import (
    IMBALANCE_STRATEGIES,
    MODEL_NAMES,
    build_pipeline,
    grid_for_pipeline,
    make_model,
    supports_class_weight,
)

warnings.filterwarnings("ignore")

APPROACH_LABELS = {
    "A": "A: Direct Classification",
    "B": "B: Cluster-then-Classify",
    "C": "C: Per-Cluster Models",
}

# =============================================================================================
# Thesis protocol (Phase 1 reproduction): behaviour identical to the original notebook
# =============================================================================================
THESIS_SCORING = "accuracy"  # the thesis tuned and selected models on accuracy
THESIS_THRESHOLDS = np.round(np.arange(0.20, 0.851, 0.01), 2)  # cut-offs searched for accuracy
THESIS_METRICS = ["Accuracy", "Balanced_Acc", "Precision", "Recall", "F1", "ROC_AUC"]


def thesis_cv() -> StratifiedKFold:
    """5-fold stratified CV used for tuning, exactly as in the thesis."""
    return StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)


def thesis_tune(name: str, X_tr: pd.DataFrame, y_tr: pd.Series) -> GridSearchCV:
    """GridSearchCV over the thesis grid; imputer + scaler + SMOTE inside every fold."""
    grid = GridSearchCV(
        build_pipeline(make_model(name), "smote"),
        grid_for_pipeline(name),
        cv=thesis_cv(),
        scoring=THESIS_SCORING,
        n_jobs=config.N_JOBS,
    )
    return grid.fit(X_tr, y_tr)


def thesis_threshold(estimator, X_tr: pd.DataFrame, y_tr: pd.Series) -> tuple[float, float]:
    """Cut-off with the best accuracy on out-of-fold TRAINING predictions (never the test set)."""
    oof = cross_val_predict(
        clone(estimator), X_tr, y_tr, cv=thesis_cv(), method="predict_proba", n_jobs=config.N_JOBS
    )[:, 1]
    scores = [accuracy_score(y_tr, (oof >= t).astype(int)) for t in THESIS_THRESHOLDS]
    best = int(np.argmax(scores))
    return float(THESIS_THRESHOLDS[best]), scores[best]


def _thesis_metrics(y_true, preds, probs) -> dict[str, float]:
    m = classification_metrics(y_true, preds, probs)
    return {k: m[k] for k in THESIS_METRICS}


def thesis_run_global(X: pd.DataFrame, y: pd.Series, idx_train, idx_test, approach: str):
    """Approaches A and B: one model per algorithm on all training rows."""
    X_tr, X_te, y_tr, y_te = X.loc[idx_train], X.loc[idx_test], y.loc[idx_train], y.loc[idx_test]
    rows, fitted = [], {}
    for name in MODEL_NAMES:
        grid = thesis_tune(name, X_tr, y_tr)
        threshold, cv_score = thesis_threshold(grid.best_estimator_, X_tr, y_tr)
        probs = grid.best_estimator_.predict_proba(X_te)[:, 1]
        preds = (probs >= threshold).astype(int)
        rows.append({
            "Approach": APPROACH_LABELS[approach], "Model": name, "Threshold": threshold,
            "CV_Score": cv_score, **_thesis_metrics(y_te, preds, probs),
            "Confusion_Matrix": confusion_matrix(y_te, preds).tolist(),
        })
        fitted[name] = {"preds": preds, "probs": probs}
        print(f"  {approach} | {name} done", flush=True)
    return pd.DataFrame(rows), fitted


def thesis_run_per_cluster(X: pd.DataFrame, y: pd.Series, clusters: pd.Series, idx_train, idx_test):
    """Approach C: every algorithm tuned and fitted separately inside each cluster."""
    y_te = y.loc[idx_test]
    train_clusters = clusters.loc[idx_train].to_numpy()
    test_clusters = clusters.loc[idx_test].to_numpy()
    rows, fitted = [], {}
    for name in MODEL_NAMES:
        probs = pd.Series(np.nan, index=idx_test)
        preds = pd.Series(np.nan, index=idx_test)
        thresholds, cv_sum, n_sum = {}, 0.0, 0
        for c in sorted(clusters.unique()):
            tr_idx, te_idx = idx_train[train_clusters == c], idx_test[test_clusters == c]
            grid = thesis_tune(name, X.loc[tr_idx], y.loc[tr_idx])
            threshold, cv_score = thesis_threshold(grid.best_estimator_, X.loc[tr_idx], y.loc[tr_idx])
            if len(te_idx):
                probs.loc[te_idx] = grid.best_estimator_.predict_proba(X.loc[te_idx])[:, 1]
                preds.loc[te_idx] = (probs.loc[te_idx] >= threshold).astype(int)
            thresholds[int(c)] = threshold
            cv_sum += cv_score * len(tr_idx)  # train-size-weighted mean of per-cluster CV scores
            n_sum += len(tr_idx)
        p, pr = preds.to_numpy().astype(int), probs.to_numpy()
        rows.append({
            "Approach": APPROACH_LABELS["C"], "Model": name, "Threshold": str(thresholds),
            "CV_Score": cv_sum / n_sum, **_thesis_metrics(y_te, p, pr),
            "Confusion_Matrix": confusion_matrix(y_te, p).tolist(),
        })
        fitted[name] = {"preds": p, "probs": pr}
        print(f"  C | {name} done", flush=True)
    return pd.DataFrame(rows), fitted


def thesis_bootstrap(y_test: np.ndarray, fitted_all: dict, results: pd.DataFrame) -> pd.DataFrame:
    """Paired bootstrap (1,000 resamples of the test set, same resamples for every approach)."""
    boot_idx = bootstrap_indices(len(y_test), config.N_BOOTSTRAP, config.RANDOM_STATE)
    metric_fns = {"Accuracy": accuracy_score, "F1": f1_score, "ROC_AUC": roc_auc_score}
    point = results.set_index(["Approach", "Model"])
    boot = {}
    for approach, models in fitted_all.items():
        for name, out in models.items():
            for metric, fn in metric_fns.items():
                values = out["probs"] if metric == "ROC_AUC" else out["preds"]
                boot[(approach, name, metric)] = bootstrap_metric(y_test, values, boot_idx, fn)
    rows = []
    for name in MODEL_NAMES:
        for comp in ["B", "C"]:
            for metric in metric_fns:
                diffs = boot[(comp, name, metric)] - boot[("A", name, metric)]
                diff = (point.loc[(APPROACH_LABELS[comp], name), metric]
                        - point.loc[(APPROACH_LABELS["A"], name), metric])
                rows.append({"Model": name, "Comparison": f"{comp} vs A", "Metric": metric,
                             **paired_difference(diffs, diff)})
    return pd.DataFrame(rows)


def reproduce() -> None:
    """Run the original thesis comparison and write results/reproduction_check.csv."""
    start = time.time()
    out_dir = config.RESULTS_DIR / "thesis_reproduced"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    X, y = df[FEATURE_NAMES], df[TARGET]
    idx_train, idx_test = split_train_test(df)

    # Cluster labels fitted on TRAINING rows only; test rows go to the nearest training centre.
    clusterer = WardClusterer(config.N_CLUSTERS).fit(X.loc[idx_train])
    clusters = pd.Series(clusterer.predict(X), index=X.index)
    clusters.loc[idx_train] = clusterer.labels_
    X_b = pd.concat([X, pd.get_dummies(clusters).astype(int).set_axis(cluster_columns(), axis=1)], axis=1)

    print("Approach A")
    res_a, fit_a = thesis_run_global(X, y, idx_train, idx_test, "A")
    print("Approach B")
    res_b, fit_b = thesis_run_global(X_b, y, idx_train, idx_test, "B")
    print("Approach C")
    res_c, fit_c = thesis_run_per_cluster(X, y, clusters, idx_train, idx_test)
    results = pd.concat([res_a, res_b, res_c], ignore_index=True)

    # Reference: always predict the majority class ("No").
    dummy = DummyClassifier(strategy="most_frequent").fit(X.loc[idx_train], y.loc[idx_train])
    baseline = {"Approach": "Baseline: always predict majority class", "Model": "Dummy",
                **_thesis_metrics(y.loc[idx_test], dummy.predict(X.loc[idx_test]),
                                  dummy.predict_proba(X.loc[idx_test])[:, 1])}
    results_with_baseline = pd.concat([results, pd.DataFrame([baseline])], ignore_index=True)
    results_with_baseline.to_csv(out_dir / "comparison.csv", index=False)

    cluster_info = pd.DataFrame([
        {"Cluster": c, "Train_N": int((clusters.loc[idx_train] == c).sum()),
         "Test_N": int((clusters.loc[idx_test] == c).sum()),
         "Train_Sleep_Trouble_Rate": round(y.loc[idx_train][clusters.loc[idx_train] == c].mean(), 3)}
        for c in sorted(clusters.unique())
    ])
    cluster_info.to_csv(out_dir / "cluster_info.csv", index=False)

    boot = thesis_bootstrap(y.loc[idx_test].to_numpy(), {"A": fit_a, "B": fit_b, "C": fit_c}, results)
    boot.to_csv(out_dir / "bootstrap.csv", index=False)

    # Fairness of the best CV-scored global model (A or B), as in the thesis.
    global_rows = results[results["Approach"].str[0].isin(["A", "B"])]
    best = global_rows.loc[global_rows["CV_Score"].idxmax()]
    preds = {"A": fit_a, "B": fit_b}[best["Approach"][0]][best["Model"]]["preds"]
    test = df.loc[idx_test]
    gender = test["gender"].map({0: "Male", 1: "Female"})
    age_group = pd.Series(np.where(test["age"] < 40, "Under 40", "40 and above"), index=test.index)
    fairness = pd.DataFrame(group_metrics(test[TARGET], preds, gender, "Gender")
                            + group_metrics(test[TARGET], preds, age_group, "Age"))
    fairness.to_csv(out_dir / "fairness.csv", index=False)

    check = build_reproduction_check(results_with_baseline, boot, cluster_info, fairness)
    check.to_csv(config.RESULTS_DIR / "reproduction_check.csv", index=False)
    print(f"\nLargest absolute difference vs original: {check['Abs_Diff'].max():.4f}")
    print(f"Significance verdicts that differ: {int((~check['Match']).sum())}")
    print(f"Done in {(time.time() - start) / 60:.1f} min")


def build_reproduction_check(results: pd.DataFrame, boot: pd.DataFrame, cluster_info: pd.DataFrame,
                             fairness: pd.DataFrame) -> pd.DataFrame:
    """Side-by-side table: original thesis number vs reproduced number."""
    orig_dir = config.RESULTS_DIR / "thesis_original"
    rows = []

    def add(section: str, key: str, metric: str, original: float, reproduced: float) -> None:
        diff = abs(float(original) - float(reproduced))
        rows.append({"Section": section, "Item": key, "Metric": metric, "Original": float(original),
                     "Reproduced": round(float(reproduced), 4), "Abs_Diff": round(diff, 4),
                     "Match": bool(diff <= 0.00005 + 1e-9)})

    orig = pd.read_csv(orig_dir / "comparison.csv").set_index(["Approach", "Model"])
    repro = results.set_index(["Approach", "Model"])
    for key in orig.index:
        for metric in ["CV_Score", *THESIS_METRICS]:
            if pd.notna(orig.loc[key, metric]):
                add("Test-set comparison", " | ".join(key), metric,
                    orig.loc[key, metric], repro.loc[key, metric])
        if key[0] in (APPROACH_LABELS["A"], APPROACH_LABELS["B"]):  # C has one threshold per cluster
            add("Test-set comparison", " | ".join(key), "Threshold", orig.loc[key, "Threshold"],
                repro.loc[key, "Threshold"])

    orig_ci = pd.read_csv(orig_dir / "cluster_info.csv").set_index("Cluster")
    for c, r in cluster_info.set_index("Cluster").iterrows():
        add("Cluster sizes", f"Cluster {c}", "Train_N", orig_ci.loc[c, "Train_N"], r["Train_N"])

    orig_boot = pd.read_csv(orig_dir / "bootstrap.csv").set_index(["Model", "Comparison", "Metric"])
    for _, r in boot.iterrows():
        key = (r["Model"], r["Comparison"], r["Metric"])
        o = orig_boot.loc[key]
        add("Paired bootstrap", " | ".join(key), "Diff", o["Diff"], r["Diff"])
        same_verdict = bool(o["Significant"]) == (r["Verdict"] != "no clear difference")
        rows.append({"Section": "Paired bootstrap", "Item": " | ".join(key), "Metric": "Significant",
                     "Original": float(o["Significant"]),
                     "Reproduced": float(r["Verdict"] != "no clear difference"),
                     "Abs_Diff": 0.0 if same_verdict else 1.0, "Match": same_verdict})

    orig_fair = pd.read_csv(orig_dir / "fairness.csv").set_index("Group")
    for _, r in fairness.iterrows():
        for metric in ["N", "Accuracy", "Precision", "Recall", "F1"]:
            add("Fairness", r["Group"], metric, orig_fair.loc[r["Group"], metric], r[metric])
    return pd.DataFrame(rows)




# =============================================================================================
# Phase 2: the same research question under the stronger evaluation
# =============================================================================================
PHASE2_DIR = config.RESULTS_DIR / "phase2"
RANKING_METRICS = {"ROC_AUC": roc_auc_score, "PR_AUC": average_precision_score}


def categorical_positions(n_cluster_columns: int = 0) -> list[int]:
    """Column positions SMOTENC must treat as categories (plus appended one-hot cluster columns)."""
    positions = [FEATURE_NAMES.index(c) for c in CATEGORICAL_FEATURES]
    first_extra = len(FEATURE_NAMES)
    return positions + list(range(first_extra, first_extra + n_cluster_columns))


def strategy_for(name: str, strategy: str) -> str:
    """KNN cannot use class weights; it falls back to SMOTE (the thesis setting) in that case."""
    return "smote" if strategy == "class_weight" and not supports_class_weight(name) else strategy


def approach_pipeline(name: str, approach: str, strategy: str):
    """Untuned pipeline for Approach A or B (B puts the Ward cluster step first).

    The cluster step is inside the pipeline, so in every CV fold the clusters are fitted on that
    fold's training rows only.
    """
    model = make_model(name, strategy)
    if approach == "B":
        return build_pipeline(model, strategy, categorical_positions(config.N_CLUSTERS),
                              first_step=("clusters", WardClusterer(config.N_CLUSTERS)))
    return build_pipeline(model, strategy, categorical_positions())


def pr_auc_search(estimator, name: str, n_jobs: int = config.N_JOBS) -> GridSearchCV:
    """GridSearchCV over the thesis grid, now scored on PR-AUC (average precision)."""
    return GridSearchCV(estimator, grid_for_pipeline(name), cv=thesis_cv(),
                        scoring=config.PRIMARY_SCORING, n_jobs=n_jobs)


def choose_imbalance_strategy(summary: pd.DataFrame) -> tuple[str, pd.Series]:
    """Strategy with the highest mean CV PR-AUC over the models that support all three strategies."""
    supported = [n for n in MODEL_NAMES if supports_class_weight(n)]
    rows = summary[summary["Model"].isin(supported)]
    means = rows.groupby("Imbalance")["PR_AUC_mean"].mean().sort_values(ascending=False)
    return str(means.index[0]), means


def paired_cv_tests(folds: pd.DataFrame, n_train_rows: int) -> pd.DataFrame:
    """B vs A and C vs A on the SAME 25 CV folds: mean difference and corrected t-test."""
    n_test = n_train_rows / config.CV_FOLDS
    n_train = n_train_rows - n_test
    rows = []
    for name in MODEL_NAMES:
        for comp in ["B", "C"]:
            for metric in RANKING_METRICS:
                a = folds[(folds.Approach == "A") & (folds.Model == name)].sort_values(["Repeat", "Fold"])
                b = folds[(folds.Approach == comp) & (folds.Model == name)].sort_values(["Repeat", "Fold"])
                diffs = b[metric].to_numpy() - a[metric].to_numpy()
                t, p = corrected_paired_ttest(diffs, int(n_train), int(n_test))
                verdict = "no clear difference" if p >= 0.05 else ("better" if diffs.mean() > 0 else "worse")
                rows.append({"Model": name, "Comparison": f"{comp} vs A", "Metric": metric,
                             "Mean_Diff": diffs.mean(), "SD_Diff": diffs.std(ddof=1),
                             "Folds_Worse": int((diffs < 0).sum()), "Folds": len(diffs),
                             "t": t, "p_value": p, "Verdict": verdict})
    return pd.DataFrame(rows)


def bootstrap_ranking(y_test: np.ndarray, probs: dict[tuple[str, str], np.ndarray]) -> pd.DataFrame:
    """Paired bootstrap of ROC-AUC and PR-AUC differences (B vs A, C vs A) on the test set."""
    boot_idx = bootstrap_indices(len(y_test), config.N_BOOTSTRAP, config.RANDOM_STATE)
    rows = []
    for name in MODEL_NAMES:
        for metric, fn in RANKING_METRICS.items():
            base = bootstrap_metric(y_test, probs[("A", name)], boot_idx, fn)
            for comp in ["B", "C"]:
                other = bootstrap_metric(y_test, probs[(comp, name)], boot_idx, fn)
                point = fn(y_test, probs[(comp, name)]) - fn(y_test, probs[("A", name)])
                rows.append({"Model": name, "Comparison": f"{comp} vs A", "Metric": metric,
                             **paired_difference(other - base, point)})
    return pd.DataFrame(rows)


def verdict_counts(table: pd.DataFrame) -> dict[str, dict[str, int]]:
    """How many of the 7 models are significantly better / worse, per comparison and metric."""
    out = {}
    for (comp, metric), group in table.groupby(["Comparison", "Metric"]):
        out[f"{comp} | {metric}"] = {
            "better": int((group.Verdict == "better").sum()),
            "worse": int((group.Verdict == "worse").sum()),
            "no clear difference": int((group.Verdict == "no clear difference").sum()),
        }
    return out


def run_phase2() -> None:
    """Imbalance comparison + A/B/C comparison with repeated CV, a screening threshold and bootstrap.

    Steps (all tuning and threshold choices use TRAINING rows only):
      1. Approach A, 7 models x 3 imbalance strategies: GridSearchCV on PR-AUC, then 5x5 CV.
      2. Pick the imbalance strategy by mean CV PR-AUC.
      3. Approaches B and C with that strategy, evaluated on the same 25 CV folds.
         C in CV uses Approach A's tuned settings inside every cluster (cluster ids change from fold
         to fold, so per-cluster settings cannot be carried over); the C model checked on the test
         set tunes each cluster separately, as in the thesis.
      4. Screening threshold per model from out-of-fold training predictions (target recall).
      5. One final check on the 20% test set + paired bootstrap; corrected t-test on the CV folds.
    """
    start = time.time()
    PHASE2_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    X, y = df[FEATURE_NAMES], df[TARGET]
    idx_train, idx_test = split_train_test(df)
    X_tr, y_tr, X_te, y_te = X.loc[idx_train], y.loc[idx_train], X.loc[idx_test], y.loc[idx_test]

    fold_tables, best_params, estimators, oof_first_repeat = [], {}, {}, {}

    def record(approach: str, strategy: str, name: str, estimator, params) -> None:
        folds, oof = repeated_cv(estimator, X_tr, y_tr)
        fold_tables.append(folds.assign(Approach=approach, Imbalance=strategy, Model=name))
        best_params[f"{approach} | {strategy} | {name}"] = params
        estimators[(approach, strategy, name)] = estimator
        oof_first_repeat[(approach, strategy, name)] = oof[0]
        pr_auc = folds["PR_AUC"].mean()
        print(f"  {approach} | {strategy:12s} | {name:20s} CV PR-AUC {pr_auc:.4f}", flush=True)

    # ---- 1. Imbalance strategies (Approach A) --------------------------------------------------
    print("Step 1: imbalance strategies (Approach A)")
    for strategy in IMBALANCE_STRATEGIES:
        for name in MODEL_NAMES:
            if strategy == "class_weight" and not supports_class_weight(name):
                continue
            search = pr_auc_search(approach_pipeline(name, "A", strategy), name).fit(X_tr, y_tr)
            record("A", strategy, name, search.best_estimator_, search.best_params_)

    imbalance_folds = pd.concat(fold_tables, ignore_index=True)
    imbalance_summary = summarize_folds(imbalance_folds, ["Imbalance", "Model"])
    imbalance_folds.to_csv(PHASE2_DIR / "imbalance_cv_folds.csv", index=False)
    imbalance_summary.to_csv(PHASE2_DIR / "imbalance_cv_summary.csv", index=False)
    chosen, strategy_means = choose_imbalance_strategy(imbalance_summary)
    print(f"Chosen imbalance strategy: {chosen}\n{strategy_means.round(4).to_string()}")

    # ---- 2. Approaches B and C with the chosen strategy ----------------------------------------
    print("Step 2: approaches B and C")
    test_models = {}
    for name in MODEL_NAMES:
        strategy = strategy_for(name, chosen)
        a_estimator = estimators[("A", strategy, name)]
        test_models[("A", name)] = a_estimator

        search = pr_auc_search(approach_pipeline(name, "B", strategy), name).fit(X_tr, y_tr)
        record("B", strategy, name, search.best_estimator_, search.best_params_)
        test_models[("B", name)] = search.best_estimator_

        # C in CV: A's tuned settings inside each cluster
        record("C", strategy, name, PerClusterClassifier(clone(a_estimator)), "A settings in every cluster")
        # C on the test set: each cluster tunes its own settings (thesis Approach C)
        per_cluster_search = PerClusterClassifier(
            pr_auc_search(approach_pipeline(name, "A", strategy), name)
        )
        test_models[("C", name)] = per_cluster_search.fit(X_tr, y_tr)
        best_params[f"C | {strategy} | {name} | test model"] = {
            f"cluster {c}": m.best_params_ for c, m in per_cluster_search.models_.items()
        }

    all_folds = pd.concat(fold_tables, ignore_index=True)
    approach_folds = pd.concat([
        all_folds[(all_folds.Approach == a) & (all_folds.Model == n)
                  & (all_folds.Imbalance == strategy_for(n, chosen))]
        for a in "ABC" for n in MODEL_NAMES
    ], ignore_index=True)
    approach_folds.to_csv(PHASE2_DIR / "approach_cv_folds.csv", index=False)
    summarize_folds(approach_folds, ["Approach", "Imbalance", "Model"]).to_csv(
        PHASE2_DIR / "approach_cv_summary.csv", index=False)

    # ---- 3. Screening threshold (training OOF) + one final check on the test set ---------------
    print("Step 3: thresholds and test-set check")
    test_rows, test_probs = [], {}
    for (approach, name), model in test_models.items():
        strategy = strategy_for(name, chosen)
        if approach == "C":  # OOF predictions of the exact C test model (re-tuned in each fold)
            oof = oof_probabilities(PerClusterClassifier(
                pr_auc_search(approach_pipeline(name, "A", strategy), name, n_jobs=1)), X_tr, y_tr)
        else:
            oof = oof_first_repeat[(approach, strategy, name)]
        threshold = threshold_for_recall(y_tr, oof)
        probs = model.predict_proba(X_te)[:, 1]
        test_probs[(approach, name)] = probs
        test_rows.append({"Approach": approach, "Imbalance": strategy, "Model": name,
                          **screening_metrics(y_te, probs, threshold)})

    dummy = DummyClassifier(strategy="prior").fit(X_tr, y_tr)
    test_rows.append({"Approach": "Baseline", "Imbalance": "-", "Model": "Majority class",
                      **screening_metrics(y_te, dummy.predict_proba(X_te)[:, 1], 1.0)})
    pd.DataFrame(test_rows).to_csv(PHASE2_DIR / "approach_test_metrics.csv", index=False)
    pd.DataFrame({f"{a} | {n}": p for (a, n), p in test_probs.items()}, index=idx_test).assign(
        y_true=y_te.to_numpy()).to_csv(PHASE2_DIR / "test_probabilities.csv")

    # ---- 4. Is the difference real? ---------------------------------------------------------
    cv_tests = paired_cv_tests(approach_folds, len(X_tr))
    cv_tests.to_csv(PHASE2_DIR / "approach_cv_paired_tests.csv", index=False)
    boot = bootstrap_ranking(y_te.to_numpy(), test_probs)
    boot.to_csv(PHASE2_DIR / "approach_bootstrap.csv", index=False)

    baseline_cv = {"ROC_AUC": 0.5, "PR_AUC": float(y_tr.mean()), "Accuracy": float(1 - y_tr.mean())}
    summary = {
        "imbalance_strategy_chosen": chosen,
        "imbalance_strategy_mean_cv_pr_auc": strategy_means.to_dict(),
        "majority_baseline_train": baseline_cv,
        "cv_corrected_ttest_counts": verdict_counts(cv_tests),
        "test_bootstrap_counts": verdict_counts(boot),
        "runtime_minutes": round((time.time() - start) / 60, 1),
    }
    save_json(summary, PHASE2_DIR / "summary.json")
    save_json(best_params, PHASE2_DIR / "best_params.json")
    print(json.dumps(summary, indent=2))


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("mode", choices=["reproduce", "compare"])
    args = parser.parse_args()
    if args.mode == "reproduce":
        reproduce()
    else:
        run_phase2()


if __name__ == "__main__":
    main()
