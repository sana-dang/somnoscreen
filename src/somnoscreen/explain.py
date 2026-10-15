"""SHAP explanations shared by the analysis and the Streamlit app, plus the chart style.

We explain the model's final output: the (calibrated) probability of "Yes". SHAP splits one
person's predicted risk minus the average predicted risk into a contribution per feature, so the
bars add up to "your risk - average risk", in percentage points.

The explainer works on the RAW answers (blanks allowed): the saved pipeline does its own
imputing and scaling, so explanations are in the same units the person typed.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # draw without a screen (servers, CI)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from somnoscreen.config import RANDOM_STATE  # noqa: E402
from somnoscreen.features import FEATURE_NAMES, FEATURES_BY_NAME  # noqa: E402

# Chart colours (validated reference palette). Series in fixed order; blue/red are the diverging
# pair used for "lowers risk" / "raises risk".
COLORS = {
    "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"],
    "lowers": "#2a78d6",
    "raises": "#e34948",
    "text": "#0b0b0b",
    "muted": "#52514e",
    "grid": "#e4e3df",
    "surface": "#fcfcfb",
}


def apply_style() -> None:
    """Thin, quiet chart style used by every figure in the project."""
    plt.rcParams.update({
        "figure.facecolor": COLORS["surface"],
        "axes.facecolor": COLORS["surface"],
        "axes.edgecolor": COLORS["grid"],
        "axes.labelcolor": COLORS["muted"],
        "axes.titlecolor": COLORS["text"],
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": COLORS["grid"],
        "grid.linewidth": 0.8,
        "xtick.color": COLORS["muted"],
        "ytick.color": COLORS["muted"],
        "lines.linewidth": 2,
        "legend.frameon": False,
        "font.size": 10,
    })


def label(name: str) -> str:
    """Plain-English name of a feature."""
    return FEATURES_BY_NAME[name].label


def make_explainer(model, background: pd.DataFrame) -> shap.PermutationExplainer:
    """Permutation SHAP explainer on the model's probability of "Yes".

    `background` is a sample of training rows that defines the "average person" baseline.
    """
    def predict(values: np.ndarray) -> np.ndarray:
        return model.predict_proba(pd.DataFrame(values, columns=FEATURE_NAMES))[:, 1]

    masker = shap.maskers.Independent(background[FEATURE_NAMES].to_numpy(dtype=float),
                                      max_samples=len(background))
    return shap.PermutationExplainer(predict, masker, feature_names=FEATURE_NAMES, seed=RANDOM_STATE)


def explain_rows(model, background: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """SHAP values (probability units) for every row of X; columns are the features.

    A fresh seeded explainer is built on every call, so the same input always gets the same answer.
    """
    explanation = make_explainer(model, background)(X[FEATURE_NAMES].to_numpy(dtype=float), silent=True)
    return pd.DataFrame(explanation.values, columns=FEATURE_NAMES, index=X.index)


def explain_one(model, background: pd.DataFrame, row: pd.DataFrame) -> tuple[pd.Series, float]:
    """Per-feature contributions for one person, and the baseline (average) risk."""
    explanation = make_explainer(model, background)(row[FEATURE_NAMES].to_numpy(dtype=float), silent=True)
    contributions = pd.Series(explanation.values[0], index=FEATURE_NAMES)
    return contributions, float(explanation.base_values[0])


def global_importance(shap_values: pd.DataFrame) -> pd.Series:
    """Mean absolute SHAP value per feature (how much each feature moves risk on average)."""
    return shap_values.abs().mean().sort_values(ascending=False)


def plot_contributions(contributions: pd.Series, top: int = 10, title: str = "") -> plt.Figure:
    """Horizontal bars: red = raised this person's risk, blue = lowered it (percentage points)."""
    apply_style()
    top_items = contributions.reindex(contributions.abs().sort_values(ascending=False).index[:top])
    top_items = top_items.iloc[::-1] * 100  # biggest at the top, in percentage points
    fig, ax = plt.subplots(figsize=(7, 0.42 * len(top_items) + 1.2))
    colors = [COLORS["raises"] if v > 0 else COLORS["lowers"] for v in top_items]
    ax.barh([label(n) for n in top_items.index], top_items.to_numpy(), color=colors, height=0.6)
    ax.axvline(0, color=COLORS["muted"], linewidth=1)
    for y, v in enumerate(top_items):
        ax.text(v + (0.15 if v >= 0 else -0.15), y, f"{v:+.1f}", va="center",
                ha="left" if v >= 0 else "right", color=COLORS["text"], fontsize=9)
    span = max(abs(top_items).max() * 1.35, 1)
    ax.set_xlim(-span, span)
    ax.set_xlabel("Change in estimated risk (percentage points)")
    ax.grid(axis="y", visible=False)
    if title:
        ax.set_title(title, loc="left")
    fig.tight_layout()
    return fig


def plot_global_importance(importance: pd.Series, title: str) -> plt.Figure:
    """Bar chart of mean |SHAP| (percentage points) for all features."""
    apply_style()
    values = importance.iloc[::-1] * 100
    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.barh([label(n) for n in values.index], values.to_numpy(), color=COLORS["series"][0], height=0.6)
    for y, v in enumerate(values):
        ax.text(v + 0.05, y, f"{v:.1f}", va="center", color=COLORS["text"], fontsize=8)
    ax.set_xlabel("Average impact on predicted risk (percentage points, mean |SHAP|)")
    ax.grid(axis="y", visible=False)
    ax.set_title(title, loc="left")
    fig.tight_layout()
    return fig
