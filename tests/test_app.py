"""Tests for the app's input handling and the saved model artifacts."""

import json

import joblib
import numpy as np
import pytest

from somnoscreen import config
from somnoscreen.features import (
    FEATURE_NAMES,
    FEATURES,
    PHQ_CHOICES,
    PHQ_ITEMS,
    answers_to_frame,
    form_to_answers,
    frequent_phq_answers,
)

ARTIFACTS = config.MODEL_PATH.exists() and config.METADATA_PATH.exists()
needs_model = pytest.mark.skipif(not ARTIFACTS, reason="run python -m somnoscreen.train first")


def full_form(**overrides) -> dict:
    """A complete form as the app would collect it (answer texts and numbers)."""
    form = {"height_cm": 170.0, "weight_kg": 80.0,
            "phq": {code: PHQ_CHOICES[1][1] for code in PHQ_ITEMS}}
    for f in FEATURES:
        if f.widget == "select":
            form[f.name] = list(f.options)[0]
        elif f.widget == "number":
            form[f.name] = f.default if f.default is not None else f.min_value + 10
    form.update(overrides)
    return form


def test_form_gives_training_columns_in_order():
    row = answers_to_frame(form_to_answers(full_form()))
    assert list(row.columns) == FEATURE_NAMES
    assert row.shape == (1, 19)
    assert row.notna().all().all()


def test_form_encoding_uses_the_feature_spec():
    form = full_form(gender="Female", smoked_100="No", education="College graduate or above")
    row = answers_to_frame(form_to_answers(form))
    assert row.loc[0, "gender"] == 1
    assert row.loc[0, "smoked_100"] == 0
    assert row.loc[0, "education"] == 4
    assert row.loc[0, "bmi"] == pytest.approx(80 / 1.7**2)
    assert row.loc[0, "phq_score"] == 7  # 7 items x "Several days" (1)


def test_blank_optional_fields_become_missing():
    row = answers_to_frame(form_to_answers(full_form(waist=None, systolic_bp=None, diastolic_bp=None,
                                                     pulse=None)))
    assert row[["waist", "systolic_bp", "diastolic_bp", "pulse"]].isna().all().all()


def test_unanswered_phq_item_makes_score_missing():
    form = full_form()
    form["phq"]["DPQ010"] = None
    assert np.isnan(form_to_answers(form)["phq_score"])


def test_unknown_feature_is_rejected():
    with pytest.raises(ValueError):
        answers_to_frame({"DPQ030": 3})


def test_frequent_phq_count():
    phq = {code: "Nearly every day" for code in list(PHQ_ITEMS)[:2]}
    phq.update({code: "Not at all" for code in list(PHQ_ITEMS)[2:]})
    assert frequent_phq_answers(phq) == 2


@needs_model
def test_metadata_matches_feature_spec():
    metadata = json.loads(config.METADATA_PATH.read_text())
    assert metadata["feature_order"] == FEATURE_NAMES
    low, moderate, high = (metadata["risk_bands"][b] for b in ["Low", "Moderate", "High"])
    assert low[1] == moderate[0] == metadata["screening_threshold"]
    assert moderate[1] == high[0]


@needs_model
def test_model_returns_probabilities_with_blank_optional_fields():
    model = joblib.load(config.MODEL_PATH)
    complete = answers_to_frame(form_to_answers(full_form()))
    blank = answers_to_frame(form_to_answers(full_form(waist=None, systolic_bp=None, diastolic_bp=None,
                                                       pulse=None)))
    for row in (complete, blank):
        p = model.predict_proba(row)[:, 1]
        assert p.shape == (1,)
        assert 0.0 <= p[0] <= 1.0


@needs_model
def test_app_runs_end_to_end():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(config.ROOT / "app" / "streamlit_app.py"), default_timeout=120)
    app.run()
    assert not app.exception
    form = full_form()
    for box in app.selectbox:
        feature = next(f for f in FEATURES if f.question == box.label)
        box.set_value(form[feature.name])
    for radio in app.radio:
        radio.set_value(PHQ_CHOICES[3][1])  # "Nearly every day" -> gentle PHQ note should appear
    heights = [n for n in app.number_input if n.label == "Height (cm)"]
    weights = [n for n in app.number_input if n.label == "Weight (kg)"]
    heights[0].set_value(170.0)
    weights[0].set_value(80.0)
    app.button[0].click()
    app.run()
    assert not app.exception
    assert any("Estimated risk" in m.label for m in app.metric)
    assert any("counsellor" in i.value for i in app.info)
