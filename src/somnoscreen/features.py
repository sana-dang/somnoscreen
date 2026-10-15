"""THE feature specification: the single source of truth for every model input.

Each `Feature` says where the value comes from in NHANES, how NHANES codes are turned into model
values, and how the Streamlit app asks the question. `data.py` (training) and the app both use the
same `choices`, so an answer is always encoded the same way at training time and at prediction
time (no train/serve skew).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# A choice is (NHANES code, answer text shown in the app, value the model sees).
Choice = tuple[int, str, float]


@dataclass(frozen=True)
class Feature:
    """One model input column."""

    name: str  # column name used by the model
    nhanes: str  # NHANES variable code ("derived" for computed features)
    source: str  # which NHANES file: demographics / examination / questionnaire
    kind: str  # "numeric", "ordinal" or "binary"
    label: str  # plain-English name used in charts and SHAP explanations
    section: str  # app form section
    question: str  # question text in the app
    widget: str  # "number", "select", "bmi" (height + weight) or "phq" (7 PHQ items)
    optional: bool = False  # may the app user leave it blank? (filled by the imputer)
    missing_codes: tuple[float, ...] = ()  # NHANES "refused" / "don't know" codes -> missing
    choices: tuple[Choice, ...] = ()  # for categorical features
    min_value: float | None = None  # numeric widget limits
    max_value: float | None = None
    default: float | None = None
    step: float | None = None
    help: str = ""

    @property
    def value_map(self) -> dict[float, float]:
        """NHANES code -> model value. Codes that are not listed become missing."""
        return {code: value for code, _, value in self.choices}

    @property
    def options(self) -> dict[str, float]:
        """App answer text -> model value."""
        return {text: value for _, text, value in self.choices}


REFUSED_DONT_KNOW = (7, 9)

SNORE_CHOICES: tuple[Choice, ...] = (
    (0, "Never", 0),
    (1, "Rarely (1-2 nights a week)", 1),
    (2, "Occasionally (3-4 nights a week)", 2),
    (3, "Frequently (5 or more nights a week)", 3),
)
YES_NO_CHOICES: tuple[Choice, ...] = ((1, "Yes", 1), (2, "No", 0))

FEATURES: list[Feature] = [
    # ---- Demographics ---------------------------------------------------------------------------
    Feature(
        "age", "RIDAGEYR", "demographics", "numeric", "Age", "About you",
        "Age (years)", "number", min_value=16, max_value=80, default=40, step=1,
        help="NHANES records everyone aged 80 or over as 80.",
    ),
    Feature(
        "gender", "RIAGENDR", "demographics", "binary", "Gender (female)", "About you",
        "Gender", "select", choices=((1, "Male", 0), (2, "Female", 1)),
    ),
    Feature(
        "education", "DMDEDUC2", "demographics", "ordinal", "Education level", "About you",
        "Highest level of education completed", "select", missing_codes=REFUSED_DONT_KNOW,
        choices=(
            (1, "Less than 9th grade", 0),
            (2, "9th-11th grade (no diploma)", 1),
            (3, "High school graduate / GED", 2),
            (4, "Some college or associate degree", 3),
            (5, "College graduate or above", 4),
        ),
    ),
    # ---- Body measurements ----------------------------------------------------------------------
    Feature(
        "bmi", "BMXBMI", "examination", "numeric", "Body mass index (BMI)", "Body measurements",
        "Height and weight (BMI is calculated for you)", "bmi", min_value=12, max_value=90,
    ),
    Feature(
        "waist", "BMXWAIST", "examination", "numeric", "Waist size", "Body measurements",
        "Waist circumference (cm)", "number", optional=True, min_value=40, max_value=180, step=0.5,
        help="Measure around your belly at the level of your navel.",
    ),
    Feature(
        "systolic_bp", "BPXSY1", "examination", "numeric", "Blood pressure (systolic)",
        "Body measurements", "Systolic blood pressure (top number, mmHg)", "number", optional=True,
        min_value=70, max_value=240, step=1,
    ),
    Feature(
        "diastolic_bp", "BPXDI1", "examination", "numeric", "Blood pressure (diastolic)",
        "Body measurements", "Diastolic blood pressure (bottom number, mmHg)", "number",
        optional=True, min_value=30, max_value=140, step=1,
    ),
    Feature(
        "pulse", "BPXPLS", "examination", "numeric", "Resting pulse", "Body measurements",
        "Resting pulse (beats per minute)", "number", optional=True, min_value=35, max_value=160,
        step=1,
    ),
    # ---- Sleep symptoms -------------------------------------------------------------------------
    Feature(
        "snoring", "SLQ030", "questionnaire", "ordinal", "Snoring", "Sleep",
        "In the past 12 months, how often did you snore while sleeping?", "select",
        missing_codes=REFUSED_DONT_KNOW, choices=SNORE_CHOICES,
    ),
    Feature(
        "snort_stop_breathing", "SLQ040", "questionnaire", "ordinal", "Snorting / stopping breathing",
        "Sleep", "In the past 12 months, how often did you snort, gasp, or stop breathing while asleep?",
        "select", missing_codes=REFUSED_DONT_KNOW, choices=SNORE_CHOICES,
    ),
    Feature(
        "daytime_sleepiness", "SLQ120", "questionnaire", "ordinal", "Daytime sleepiness", "Sleep",
        "In the past month, how often did you feel overly sleepy during the day?", "select",
        missing_codes=REFUSED_DONT_KNOW,
        choices=(
            (0, "Never", 0),
            (1, "Rarely (once a month)", 1),
            (2, "Sometimes (2-4 times a month)", 2),
            (3, "Often (5-15 times a month)", 3),
            (4, "Almost always (16-30 times a month)", 4),
        ),
    ),
    # ---- Lifestyle ------------------------------------------------------------------------------
    Feature(
        "vigorous_work", "PAQ605", "questionnaire", "binary", "Vigorous activity at work", "Lifestyle",
        "Does your work involve vigorous activity (e.g. heavy lifting) for at least 10 minutes at a time?",
        "select", missing_codes=REFUSED_DONT_KNOW, choices=YES_NO_CHOICES,
    ),
    Feature(
        "vigorous_recreation", "PAQ620", "questionnaire", "binary", "Vigorous exercise / sport",
        "Lifestyle", "Do you do vigorous sports or fitness activities for at least 10 minutes at a time?",
        "select", missing_codes=REFUSED_DONT_KNOW, choices=YES_NO_CHOICES,
    ),
    Feature(
        "smoked_100", "SMQ020", "questionnaire", "binary", "Has smoked (100+ cigarettes)", "Lifestyle",
        "Have you smoked at least 100 cigarettes in your entire life?", "select",
        missing_codes=REFUSED_DONT_KNOW, choices=YES_NO_CHOICES,
    ),
    # ---- Self-report ----------------------------------------------------------------------------
    Feature(
        "sleep_hours_weekday", "SLD012", "questionnaire", "numeric", "Sleep hours (weekdays)", "Sleep",
        "Usual hours of sleep on weekdays or workdays", "number", min_value=2, max_value=14,
        default=7.0, step=0.5,
        # NOTE: no missing codes here, 7 and 9 hours are real answers.
    ),
    Feature(
        "sleep_hours_weekend", "SLD013", "questionnaire", "numeric", "Sleep hours (weekends)", "Sleep",
        "Usual hours of sleep on weekends or non-workdays", "number", min_value=2, max_value=14,
        default=8.0, step=0.5,
    ),
    Feature(
        "phq_score", "derived", "questionnaire", "numeric", "Mood score (PHQ, 7 items)", "Mood",
        "Over the last 2 weeks, how often have you been bothered by the following problems?", "phq",
    ),
    Feature(
        "general_health", "HUQ010", "questionnaire", "ordinal", "Self-rated health (higher = worse)",
        "Lifestyle", "Would you say your health in general is...", "select",
        missing_codes=REFUSED_DONT_KNOW,
        choices=((1, "Excellent", 1), (2, "Very good", 2), (3, "Good", 3), (4, "Fair", 4), (5, "Poor", 5)),
    ),
    Feature(
        "sitting_minutes", "PAD680", "questionnaire", "numeric", "Sitting time per day", "Lifestyle",
        "Minutes spent sitting on a typical day", "number", missing_codes=(7777, 9999),
        min_value=0, max_value=1380, default=360, step=15,
        help="Include sitting at work, at home, travelling and watching TV.",
    ),
]

# Model column order. It is the same order as the thesis notebook's `base_features`.
FEATURE_NAMES: list[str] = [f.name for f in FEATURES]
FEATURES_BY_NAME: dict[str, Feature] = {f.name: f for f in FEATURES}

# Columns that hold categories (ordinal or yes/no) rather than measurements. SMOTENC treats these
# as categorical so synthetic rows never get impossible in-between values like "snoring = 1.37".
CATEGORICAL_FEATURES: list[str] = [f.name for f in FEATURES if f.kind in ("ordinal", "binary")]

# ---- PHQ items --------------------------------------------------------------------------------
# PHQ-9 depression questionnaire items used to build `phq_score` (0-21).
# DPQ030 ("Trouble falling or staying asleep, or sleeping too much") is DELIBERATELY EXCLUDED:
# it asks almost the same thing as the target question (SLQ050, "ever told a doctor you had
# trouble sleeping?"), so including it would leak the answer into the features.
# DPQ090 (thoughts of self-harm) is not part of the PHQ-8 and is not used.
PHQ_ITEMS: dict[str, str] = {
    "DPQ010": "Little interest or pleasure in doing things",
    "DPQ020": "Feeling down, depressed, or hopeless",
    "DPQ040": "Feeling tired or having little energy",
    "DPQ050": "Poor appetite or overeating",
    "DPQ060": "Feeling bad about yourself, or that you are a failure or have let yourself or your "
    "family down",
    "DPQ070": "Trouble concentrating on things, such as reading the newspaper or watching television",
    "DPQ080": "Moving or speaking so slowly that other people could have noticed, or the opposite: "
    "being so fidgety or restless that you have been moving around a lot more than usual",
}
PHQ_CHOICES: tuple[Choice, ...] = (
    (0, "Not at all", 0),
    (1, "Several days", 1),
    (2, "More than half the days", 2),
    (3, "Nearly every day", 3),
)

# ---- Target -----------------------------------------------------------------------------------
# SLQ050: "Have you ever told a doctor or other health professional that you have trouble
# sleeping?" This is SELF-REPORTED, not a clinical diagnosis of a sleep disorder.
TARGET_NHANES = "SLQ050"
TARGET = "sleep_trouble"
TARGET_MAP = {1: 1, 2: 0}  # 1 = Yes, 2 = No; 7 (refused) and 9 (don't know) rows are dropped


def nhanes_columns(source: str) -> list[str]:
    """NHANES variable codes needed from one source file (besides the SEQN id)."""
    cols = [f.nhanes for f in FEATURES if f.source == source and f.nhanes != "derived"]
    if source == "questionnaire":
        cols += [TARGET_NHANES, *PHQ_ITEMS]
    return cols


def compute_bmi(height_cm: float | None, weight_kg: float | None) -> float:
    """BMI = weight (kg) / height (m)^2, or NaN if either value is missing."""
    if not height_cm or not weight_kg:
        return np.nan
    return weight_kg / (height_cm / 100) ** 2


def phq_total(item_values: list[float | None]) -> float:
    """Sum of the 7 PHQ items (0-21). Missing if any item is missing, exactly as in training."""
    values = pd.Series(item_values, dtype=float)
    return values.sum(min_count=len(PHQ_ITEMS)) if len(values) == len(PHQ_ITEMS) else np.nan


def answers_to_frame(answers: dict[str, float | None]) -> pd.DataFrame:
    """Turn {feature name: model value} into a one-row DataFrame in training column order.

    Missing or None answers become NaN, which the saved pipeline's imputer fills in exactly as it
    did during training.
    """
    unknown = set(answers) - set(FEATURE_NAMES)
    if unknown:
        raise ValueError(f"Unknown features: {sorted(unknown)}")
    row = {name: answers.get(name) for name in FEATURE_NAMES}
    return pd.DataFrame([row], columns=FEATURE_NAMES).astype(float)
