"""Charts saved to figures/ (PNG). Every chart is drawn from numbers saved in results/."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_curve

from somnoscreen import config
from somnoscreen.explain import COLORS, apply_style, plot_global_importance, plt
from somnoscreen.pipeline import MODEL_NAMES

APPROACH_NAMES = {"A": "A: direct", "B": "B: + cluster label", "C": "C: per-cluster models"}


def _save(fig: plt.Figure, name: str) -> Path:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def roc_pr_curves(y_true: np.ndarray, final_probs: np.ndarray, test_prob_csv: Path, model: str) -> Path:
    """ROC and precision-recall curves on the test set: Approach A vs B vs C for the final model."""
    apply_style()
    probs = pd.read_csv(test_prob_csv, index_col=0)
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 4.6))
    for color, approach in zip(COLORS["series"], "ABC", strict=False):
        p = probs[f"{approach} | {model}"]
        fpr, tpr, _ = roc_curve(y_true, p)
        precision, recall, _ = precision_recall_curve(y_true, p)
        ax_roc.plot(fpr, tpr, color=color, label=APPROACH_NAMES[approach])
        ax_pr.plot(recall, precision, color=color,
                   label=f"{APPROACH_NAMES[approach]} (PR-AUC {average_precision_score(y_true, p):.3f})")
    ax_roc.plot([0, 1], [0, 1], "--", color=COLORS["muted"], linewidth=1, label="Chance")
    ax_pr.axhline(y_true.mean(), linestyle="--", color=COLORS["muted"], linewidth=1,
                  label=f"Chance (base rate {y_true.mean():.3f})")
    ax_pr.axvline(config.TARGET_RECALL, linestyle=":", color=COLORS["muted"], linewidth=1)
    ax_pr.text(config.TARGET_RECALL - 0.01, 0.03, "target recall", color=COLORS["muted"], fontsize=8,
               ha="right")
    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate (recall)",
               title=f"ROC curve, test set ({model})")
    ax_pr.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve, test set", ylim=(0, 1))
    ax_roc.legend(loc="lower right")
    ax_pr.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return _save(fig, "roc_pr_curves.png")


def calibration_plot(y_true: np.ndarray, raw_probs: np.ndarray, cal_probs: np.ndarray, method: str) -> Path:
    """Reliability diagram: predicted risk vs observed rate, before and after calibration."""
    apply_style()
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.plot([0, 1], [0, 1], "--", color=COLORS["muted"], linewidth=1, label="Perfect calibration")
    curves = [("Uncalibrated", raw_probs, COLORS["series"][1])]
    if method != "none":
        curves.append((f"Calibrated ({method})", cal_probs, COLORS["series"][0]))
    for name, p, color in curves:
        observed, predicted = calibration_curve(y_true, p, n_bins=10, strategy="quantile")
        ax.plot(predicted, observed, marker="o", markersize=5, color=color,
                label=f"{name}, Brier {brier_score_loss(y_true, p):.3f}")
    ax.set(xlabel="Predicted risk", ylabel="Observed share with sleep trouble",
           title="Calibration on the test set", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    return _save(fig, "calibration.png")


def fairness_plot(fairness: pd.DataFrame) -> Path:
    """Recall and precision per gender and age group at the screening threshold."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(fairness))
    width = 0.36
    for offset, metric, color in [(-width / 2, "Recall", COLORS["series"][0]),
                                  (width / 2, "Precision", COLORS["series"][1])]:
        bars = ax.bar(x + offset, fairness[metric], width - 0.03, color=color, label=metric)
        for bar, value in zip(bars, fairness[metric], strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.2f}", ha="center",
                    fontsize=8, color=COLORS["text"])
    ax.axhline(config.TARGET_RECALL, linestyle=":", color=COLORS["muted"], linewidth=1)
    ax.text(len(fairness) - 0.5, config.TARGET_RECALL + 0.015, "target recall", ha="right",
            fontsize=8, color=COLORS["muted"])
    ax.set_xticks(x, [f"{g}\n(n={n})" for g, n in zip(fairness["Group"], fairness["N"], strict=True)])
    ax.set(ylim=(0, 1), ylabel="Score on test set", title="Fairness check at the screening threshold")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", ncols=2)
    fig.tight_layout()
    return _save(fig, "fairness.png")


def shap_importance_plot(importance: pd.Series, model: str) -> Path:
    """Global SHAP importance of the final model."""
    return _save(plot_global_importance(importance, f"What drives the predicted risk ({model})"),
                 "shap_importance.png")


def _dot_plot(ax, summary: pd.DataFrame, group_col: str, groups: list[str], labels: dict[str, str],
              metric: str) -> None:
    """Mean +/- SD per model as dots with whiskers, one colour per group."""
    y = np.arange(len(MODEL_NAMES))
    offsets = np.linspace(-0.22, 0.22, len(groups))
    for offset, group, color in zip(offsets, groups, COLORS["series"], strict=False):
        rows = summary[summary[group_col] == group].set_index("Model").reindex(MODEL_NAMES)
        ax.errorbar(rows[f"{metric}_mean"], y + offset, xerr=rows[f"{metric}_std"], fmt="o",
                    markersize=6, color=color, ecolor=color, elinewidth=1.2, capsize=0,
                    label=labels[group])
    ax.set_yticks(y, MODEL_NAMES)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)


def imbalance_plot(summary: pd.DataFrame) -> Path:
    """CV PR-AUC (mean +/- SD, 5x5 CV) for each model under each imbalance strategy."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    labels = {"smote": "SMOTE", "smotenc": "SMOTENC", "class_weight": "class_weight='balanced'"}
    _dot_plot(ax, summary, "Imbalance", list(labels), labels, "PR_AUC")
    ax.set(xlabel="PR-AUC (mean ± SD over 25 CV folds)",
           title="Imbalance strategies, Approach A (KNN has no class weights)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncols=3, fontsize=9)
    fig.tight_layout()
    return _save(fig, "imbalance_comparison.png")


def approach_plot(summary: pd.DataFrame) -> Path:
    """CV ROC-AUC and PR-AUC (mean +/- SD) for Approaches A, B, C."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, metric, name in zip(axes, ["ROC_AUC", "PR_AUC"], ["ROC-AUC", "PR-AUC"], strict=True):
        _dot_plot(ax, summary, "Approach", list("ABC"), APPROACH_NAMES, metric)
        ax.set(xlabel=f"{name} (mean ± SD over 25 CV folds)", title=f"Does clustering help? {name}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncols=3, fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, "approach_comparison.png")
