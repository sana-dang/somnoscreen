"""Copy the ORIGINAL thesis results out of the saved notebook outputs into CSV files.

The numbers are read from the HTML tables stored in notebooks/thesis_original.ipynb (they were
produced when the thesis notebook was run). Nothing is retyped by hand. The CSVs are used by
`python -m somnoscreen.experiments reproduce` to build results/reproduction_check.csv.

Run: python scripts/extract_thesis_numbers.py   (needs lxml, see requirements-dev.txt)
"""

import io
import json

import pandas as pd

from somnoscreen.config import ROOT

NOTEBOOK = ROOT / "notebooks" / "thesis_original.ipynb"
OUT_DIR = ROOT / "results" / "thesis_original"


def html_tables(cell: dict) -> list[pd.DataFrame]:
    """All HTML tables shown as outputs of one notebook cell."""
    tables = []
    for output in cell.get("outputs", []):
        html = output.get("data", {}).get("text/html")
        if html:
            tables.append(pd.read_html(io.StringIO("".join(html)))[0])
    return tables


def flatten(table: pd.DataFrame) -> pd.DataFrame:
    """Turn pandas' multi-level HTML headers into simple column names."""
    if isinstance(table.columns, pd.MultiIndex):
        table.columns = [
            " | ".join(part for part in col if not str(part).startswith("Unnamed")) for col in table.columns
        ]
    return table.loc[:, [c for c in table.columns if not str(c).startswith("Unnamed")]]


def main() -> None:
    """Extract the comparison table, cluster sizes, bootstrap table and fairness table."""
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Cell 60: full A/B/C comparison on the test set + majority-class baseline.
    comparison = flatten(html_tables(cells[60])[0])
    comparison.to_csv(OUT_DIR / "comparison.csv", index=False)

    # Cell 56: cluster sizes for Approach C.
    flatten(html_tables(cells[56])[0]).to_csv(OUT_DIR / "cluster_info.csv", index=False)

    # Cell 64: paired bootstrap, text such as "-0.0270*" (* = 95% interval excludes 0).
    boot = flatten(html_tables(cells[64])[0]).rename(columns={"Model": "Model"})
    model_col = boot.columns[0]
    rows = []
    for _, r in boot.iterrows():
        for col in boot.columns[1:]:
            comparison_name, metric = [part.strip() for part in col.split("|")]
            text = str(r[col])
            rows.append({
                "Model": r[model_col],
                "Comparison": comparison_name,
                "Metric": metric,
                "Diff": float(text.rstrip("*")),
                "Significant": text.endswith("*"),
            })
    pd.DataFrame(rows).to_csv(OUT_DIR / "bootstrap.csv", index=False)

    # Cell 74: fairness of the selected model.
    flatten(html_tables(cells[74])[0]).to_csv(OUT_DIR / "fairness.csv", index=False)
    print(f"Saved original thesis numbers to {OUT_DIR}")


if __name__ == "__main__":
    main()
