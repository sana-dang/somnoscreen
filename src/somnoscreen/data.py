"""Data layer: load the three NHANES files, merge them, decode NHANES codes into model values.

Output: one clean DataFrame with `SEQN`, the 19 feature columns (in `FEATURE_NAMES` order) and the
binary target. Missing values are left as NaN; they are filled later INSIDE the model pipeline,
from training rows only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from somnoscreen import config
from somnoscreen.features import (
    FEATURE_NAMES,
    FEATURES,
    PHQ_ITEMS,
    REFUSED_DONT_KNOW,
    TARGET,
    TARGET_MAP,
    TARGET_NHANES,
    nhanes_columns,
)

SOURCE_FILES = {
    "demographics": "demographics.csv",
    "examination": "examination.csv",
    "questionnaire": "questionnaire.csv",
}


def _read_source(data_dir: Path, source: str) -> pd.DataFrame:
    """Read the needed columns of one NHANES file; fail loudly if a variable is missing."""
    path = data_dir / SOURCE_FILES[source]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Copy the NHANES CSVs into {data_dir} or run scripts/download_nhanes.py."
        )
    wanted = ["SEQN", *nhanes_columns(source)]
    available = pd.read_csv(path, nrows=0).columns
    missing = [c for c in wanted if c not in available]
    if missing:
        raise KeyError(f"{path.name} is missing required NHANES variables: {missing}")
    return pd.read_csv(path, usecols=wanted)[wanted]


def load_raw(data_dir: Path = config.DATA_DIR) -> pd.DataFrame:
    """Merge demographics + examination + questionnaire on SEQN (inner joins, as in the thesis)."""
    demo = _read_source(data_dir, "demographics")
    exam = _read_source(data_dir, "examination")
    ques = _read_source(data_dir, "questionnaire")
    return demo.merge(exam, on="SEQN", how="inner").merge(ques, on="SEQN", how="inner")


def decode(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn raw NHANES codes into model values and keep only rows with a Yes/No target.

    - Target SLQ050: 1 -> 1 (Yes), 2 -> 0 (No); rows with 7/9 (refused / don't know) or no answer
      are dropped.
    - Each feature: its missing codes become NaN, then categorical codes are mapped with the
      feature's `choices` (any unlisted code also becomes NaN).
    - phq_score: sum of 7 PHQ items (DPQ030 excluded, see features.py), only when all 7 are answered.
    """
    df = raw[raw[TARGET_NHANES].isin(TARGET_MAP)].copy()

    out = pd.DataFrame({"SEQN": df["SEQN"]})
    for feature in FEATURES:
        if feature.name == "phq_score":
            items = df[list(PHQ_ITEMS)].replace({code: np.nan for code in REFUSED_DONT_KNOW})
            out["phq_score"] = items.sum(axis=1, min_count=len(PHQ_ITEMS))
            continue
        values = df[feature.nhanes].replace({code: np.nan for code in feature.missing_codes})
        if feature.choices:
            values = values.map(feature.value_map)
        out[feature.name] = values.astype(float)

    out[TARGET] = df[TARGET_NHANES].map(TARGET_MAP).astype(int)
    return out[["SEQN", *FEATURE_NAMES, TARGET]].reset_index(drop=True)


def load_dataset(data_dir: Path = config.DATA_DIR) -> pd.DataFrame:
    """Load, merge and decode the NHANES 2017-18 data into the modelling table."""
    return decode(load_raw(data_dir))


def split_train_test(df: pd.DataFrame) -> tuple[pd.Index, pd.Index]:
    """80/20 stratified split of row labels, identical to the thesis (random_state=42).

    Returns the row labels in the shuffled order produced by `train_test_split`; that order is kept
    everywhere so results match the thesis exactly.
    """
    idx_train, idx_test = train_test_split(
        df.index, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=df[TARGET]
    )
    return idx_train, idx_test
