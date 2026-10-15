"""Turn the saved results into markdown tables (results/tables.md) for the README and study guide.

Run: python scripts/make_results_tables.py
"""

import json

import pandas as pd

from somnoscreen.config import RESULTS_DIR
from somnoscreen.pipeline import MODEL_NAMES

P2 = RESULTS_DIR / "phase2"


def pm(mean: float, sd: float) -> str:
    """'mean ± sd' with 3 decimals."""
    return f"{mean:.3f} ± {sd:.3f}"


def table(rows: list[list[str]], header: list[str]) -> str:
    """Simple markdown table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def main() -> None:
    """Write results/tables.md."""
    out = []
    summary = json.loads((P2 / "summary.json").read_text())
    ev = json.loads((RESULTS_DIR / "final_model_evaluation.json").read_text())

    cv = pd.read_csv(P2 / "approach_cv_summary.csv").set_index(["Approach", "Model"])
    rows = []
    for name in MODEL_NAMES:
        row = [name, cv.loc[("A", name), "Imbalance"]]
        for metric in ["ROC_AUC", "PR_AUC"]:
            for a in "ABC":
                row.append(pm(cv.loc[(a, name), f"{metric}_mean"], cv.loc[(a, name), f"{metric}_std"]))
        rows.append(row)
    base = summary["majority_baseline_train"]
    rows.append(["Majority-class baseline", "-", f"{base['ROC_AUC']:.3f}", "", "",
                 f"{base['PR_AUC']:.3f}", "", ""])
    out.append("## CV results by approach (5 folds x 5 repeats, training set, mean ± SD)\n")
    out.append(table(rows, ["Model", "Imbalance", "ROC-AUC A", "ROC-AUC B", "ROC-AUC C",
                            "PR-AUC A", "PR-AUC B", "PR-AUC C"]))

    imb = pd.read_csv(P2 / "imbalance_cv_summary.csv").set_index(["Imbalance", "Model"])
    rows = []
    for name in MODEL_NAMES:
        row = [name]
        for s in ["smote", "smotenc", "class_weight"]:
            row.append(pm(imb.loc[(s, name), "PR_AUC_mean"], imb.loc[(s, name), "PR_AUC_std"])
                       if (s, name) in imb.index else "n/a")
        rows.append(row)
    means = summary["imbalance_strategy_mean_cv_pr_auc"]
    rows.append(["**Mean of 6 models (no KNN)**"] + [f"**{means[s]:.3f}**"
                                                     for s in ["smote", "smotenc", "class_weight"]])
    out.append("\n\n## Imbalance strategies, Approach A (CV PR-AUC, mean ± SD)\n")
    out.append(table(rows, ["Model", "SMOTE", "SMOTENC", "class_weight='balanced'"]))

    out.append("\n\n## Is clustering better? (number of the 7 models)\n")
    rows = []
    for key in ["B vs A | ROC_AUC", "B vs A | PR_AUC", "C vs A | ROC_AUC", "C vs A | PR_AUC"]:
        c, b = summary["cv_corrected_ttest_counts"][key], summary["test_bootstrap_counts"][key]
        rows.append([key.replace(" | ", ", ").replace("_", "-"), f"{c['better']} / {c['worse']} / {c['no clear difference']}",
                     f"{b['better']} / {b['worse']} / {b['no clear difference']}"])
    out.append(table(rows, ["Comparison", "CV corrected t-test: better / worse / no difference",
                            "Test bootstrap: better / worse / no difference"]))

    cal = pd.read_csv(RESULTS_DIR / "calibration_cv_summary.csv")
    out.append("\n\n## Calibration (final model, 5x5 CV)\n")
    out.append(table([[r.Calibration, pm(r.Brier_mean, r.Brier_std), pm(r.ROC_AUC_mean, r.ROC_AUC_std),
                       pm(r.PR_AUC_mean, r.PR_AUC_std)] for r in cal.itertuples()],
                     ["Calibration", "Brier (lower is better)", "ROC-AUC", "PR-AUC"]))

    t, m, cis = ev["test_at_threshold"], ev["test_majority_baseline"], ev["test_bootstrap_95ci"]
    cvf = ev["cv_5x5_final_model"]
    out.append(f"\n\n## Final model on the held-out test set (n={ev['n_test']}), threshold "
               f"{ev['screening_threshold']:.3f}\n")
    rows = [
        ["ROC-AUC", f"{t['ROC_AUC']:.3f} (95% CI {cis['ROC_AUC'][0]:.3f}-{cis['ROC_AUC'][1]:.3f})",
         pm(cvf["ROC_AUC_mean"], cvf["ROC_AUC_std"]), f"{m['ROC_AUC']:.3f}"],
        ["PR-AUC", f"{t['PR_AUC']:.3f} (95% CI {cis['PR_AUC'][0]:.3f}-{cis['PR_AUC'][1]:.3f})",
         pm(cvf["PR_AUC_mean"], cvf["PR_AUC_std"]), f"{m['PR_AUC']:.3f}"],
        ["Brier score", f"{t['Brier']:.3f}", pm(cvf["Brier_mean"], cvf["Brier_std"]), f"{m['Brier']:.3f}"],
        ["Recall (sensitivity)", f"{t['Recall']:.3f}", "", f"{m['Recall']:.3f}"],
        ["Precision", f"{t['Precision']:.3f}", pm(cvf["Precision@Recall_mean"], cvf["Precision@Recall_std"])
         + " (at 70% recall)", f"{m['Precision']:.3f}"],
        ["F1", f"{t['F1']:.3f}", "", f"{m['F1']:.3f}"],
        ["Balanced accuracy", f"{t['Balanced_Acc']:.3f}", "", f"{m['Balanced_Acc']:.3f}"],
        ["Accuracy", f"{t['Accuracy']:.3f}", "", f"{m['Accuracy']:.3f}"],
    ]
    out.append(table(rows, ["Metric", "Test set (at screening threshold)", "CV 5x5 (training)",
                            "Majority baseline (test)"]))

    fair = pd.read_csv(RESULTS_DIR / "fairness.csv")
    out.append("\n\n## Fairness on the test set at the screening threshold\n")
    out.append(table([[r.Group, str(r.N), f"{r.Positive_Rate:.3f}", f"{r.Flagged_Rate:.3f}",
                       f"{r.Recall:.3f}", f"{r.Precision:.3f}"] for r in fair.itertuples()],
                     ["Group", "N", "Share with sleep trouble", "Share flagged", "Recall", "Precision"]))

    (RESULTS_DIR / "tables.md").write_text("\n".join(out) + "\n")
    print((RESULTS_DIR / "tables.md").read_text())


if __name__ == "__main__":
    main()
