"""Build and execute notebooks/analysis.ipynb (the short story of the project, with charts).

Every number shown in the notebook is read from results/ or computed by the package; nothing is
typed in by hand. Run after the experiments, evaluation and training:

    python scripts/build_analysis_notebook.py
"""

import nbformat
from nbclient import NotebookClient

from somnoscreen.config import ROOT

md = nbformat.v4.new_markdown_cell
code = nbformat.v4.new_code_cell

CELLS = [
    md("# SomnoScreen: analysis\n\n"
       "A low-cost sleep-trouble risk screener built on NHANES 2017-18. This notebook only *reads* the "
       "saved results and calls the package in `src/somnoscreen/`, so it tells the story without "
       "re-running the long experiments.\n\n"
       "**Target:** SLQ050, *\"Have you ever told a doctor you had trouble sleeping?\"*. It is "
       "self-reported, not a clinical diagnosis."),
    code("import json\n"
         "import pandas as pd\n"
         "from IPython.display import Image, Markdown, display\n"
         "from somnoscreen import config\n"
         "from somnoscreen.data import load_dataset\n"
         "from somnoscreen.features import FEATURE_NAMES, TARGET\n\n"
         "R = config.RESULTS_DIR\n"
         "pd.set_option('display.precision', 4)"),
    md("## 1. Data"),
    code("df = load_dataset()\n"
         "print(f'{len(df):,} adults, {len(FEATURE_NAMES)} features')\n"
         "print(df[TARGET].value_counts().rename({1: 'Yes', 0: 'No'}).to_string())\n"
         "print(f'Positive rate: {df[TARGET].mean():.1%}')\n"
         "df[FEATURE_NAMES].isna().mean().sort_values(ascending=False).head(6).rename('share missing')"),
    md("## 2. Phase 1: the thesis reproduces exactly\n\nThe original notebook logic was ported into the "
       "package and re-run. Every number is compared with the value saved in the original notebook."),
    code("check = pd.read_csv(R / 'reproduction_check.csv')\n"
         "print(f\"{int(check.Match.sum())} of {len(check)} numbers match; largest difference \"\n"
         "      f\"{check.Abs_Diff.max():.4f}\")\n"
         "check.groupby('Section').Match.agg(['sum', 'count'])"),
    md("## 3. Phase 2: imbalance strategies\n\nApproach A, each model tuned on PR-AUC, then 5 folds x "
       "5 repeats of stratified CV on the training set."),
    code("imb = pd.read_csv(R / 'phase2' / 'imbalance_cv_summary.csv')\n"
         "summary = json.loads((R / 'phase2' / 'summary.json').read_text())\n"
         "print('Chosen strategy:', summary['imbalance_strategy_chosen'])\n"
         "print(pd.Series(summary['imbalance_strategy_mean_cv_pr_auc']).round(4).to_string())\n"
         "imb.pivot(index='Model', columns='Imbalance', values='PR_AUC_mean')"),
    code("Image(str(config.FIGURES_DIR / 'imbalance_comparison.png'))"),
    md("## 4. Does clustering help? (thesis question, new evaluation)"),
    code("cv = pd.read_csv(R / 'phase2' / 'approach_cv_summary.csv')\n"
         "cv.pivot(index='Model', columns='Approach', values=['ROC_AUC_mean', 'PR_AUC_mean'])"),
    code("Image(str(config.FIGURES_DIR / 'approach_comparison.png'))"),
    code("print('Corrected paired t-test on the 25 CV folds (count of 7 models):')\n"
         "display(pd.DataFrame(summary['cv_corrected_ttest_counts']).T)\n"
         "print('Paired bootstrap on the test set, 1,000 resamples (count of 7 models):')\n"
         "pd.DataFrame(summary['test_bootstrap_counts']).T"),
    md("## 5. Final model: calibration, screening threshold, test set"),
    code("ev = json.loads((R / 'final_model_evaluation.json').read_text())\n"
         "sel = ev['selection']\n"
         "print(f\"Model: {sel['model']}, imbalance: {sel['imbalance_strategy']}, \"\n"
         "      f\"calibration: {sel['calibration']}\")\n"
         "display(pd.read_csv(R / 'calibration_cv_summary.csv')[['Calibration', 'Brier_mean', 'Brier_std',\n"
         "                                                     'ROC_AUC_mean', 'PR_AUC_mean']])\n"
         "print(f\"Screening threshold (target recall {ev['target_recall']}): \"\n"
         "      f\"{ev['screening_threshold']:.4f}\")\n"
         "pd.read_csv(R / 'final_test_metrics.csv')"),
    code("display(Image(str(config.FIGURES_DIR / 'roc_pr_curves.png')))\n"
         "Image(str(config.FIGURES_DIR / 'calibration.png'))"),
    md("## 6. Fairness at the screening threshold"),
    code("display(pd.read_csv(R / 'fairness.csv'))\n"
         "Image(str(config.FIGURES_DIR / 'fairness.png'))"),
    md("## 7. What drives the prediction (SHAP)"),
    code("Image(str(config.FIGURES_DIR / 'shap_importance.png'))"),
    md("One person, explained the same way the app does it (the deployed model, compared with the "
       "average person in the background sample):"),
    code("import joblib\n"
         "from somnoscreen.explain import explain_one, plot_contributions\n"
         "model = joblib.load(config.MODEL_PATH)\n"
         "background = pd.read_csv(config.SHAP_BACKGROUND_PATH)\n"
         "person = df[FEATURE_NAMES].iloc[[0]]\n"
         "contrib, avg = explain_one(model, background, person)\n"
         "print(f'Predicted risk {model.predict_proba(person)[0, 1]:.1%} vs average {avg:.1%}')\n"
         "plot_contributions(contrib, title='Why this estimate? (row 0)')"),
    md("## Limitations\n\n"
       "- The target is **self-reported** (\"ever told a doctor\"), not a diagnosed sleep disorder.\n"
       "- NHANES **survey weights are not used**, so the numbers describe this sample, not the US "
       "population.\n"
       "- **Cross-sectional** data: associations, not causes.\n"
       "- **US population**, 2017-18 only; the model has not been validated elsewhere."),
]


def main() -> None:
    """Write the notebook and execute it so the outputs are saved inside."""
    nb = nbformat.v4.new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {"name": "somnoscreen", "display_name": "Python 3.11 (somnoscreen)",
                                 "language": "python"}
    path = ROOT / "notebooks" / "analysis.ipynb"
    resources = {"metadata": {"path": str(ROOT)}}
    client = NotebookClient(nb, timeout=600, kernel_name="somnoscreen", resources=resources)
    client.execute()
    nbformat.write(nb, path)
    print(f"Wrote and executed {path}")


if __name__ == "__main__":
    main()
