"""Tests for the feature spec and the data layer."""

import numpy as np
import pandas as pd
import pytest

from somnoscreen.data import decode
from somnoscreen.features import FEATURE_NAMES, FEATURES, FEATURES_BY_NAME, PHQ_ITEMS, TARGET

from .conftest import needs_data


def _raw_row(**overrides) -> dict:
    """One valid raw NHANES row (every variable answered), with optional overrides."""
    row = {"SEQN": 1, "SLQ050": 1, **{code: 1 for code in PHQ_ITEMS}}
    for f in FEATURES:
        if f.nhanes != "derived":
            row[f.nhanes] = f.choices[0][0] if f.choices else 30.0
    row.update(overrides)
    return row


def test_dpq030_is_excluded():
    """DPQ030 (trouble sleeping) nearly duplicates the target, so it must never be used."""
    assert "DPQ030" not in PHQ_ITEMS
    assert len(PHQ_ITEMS) == 7
    assert all(f.nhanes != "DPQ030" for f in FEATURES)


def test_nineteen_features_in_thesis_order():
    assert len(FEATURE_NAMES) == 19
    assert FEATURE_NAMES[:3] == ["age", "gender", "education"]
    assert FEATURE_NAMES[-3:] == ["phq_score", "general_health", "sitting_minutes"]


def test_refused_and_dont_know_codes_become_missing():
    raw = pd.DataFrame([
        _raw_row(SLQ030=7, SLQ040=9, DMDEDUC2=7, HUQ010=9, PAQ605=7, PAD680=7777),
        _raw_row(PAD680=9999, SMQ020=9),
    ])
    out = decode(raw)
    assert out.loc[0, ["snoring", "snort_stop_breathing", "education", "general_health",
                       "vigorous_work", "sitting_minutes"]].isna().all()
    assert out.loc[1, ["sitting_minutes", "smoked_100"]].isna().all()


def test_sleep_hours_seven_and_nine_are_kept():
    """7 and 9 are real sleep-hour answers, not missing codes."""
    out = decode(pd.DataFrame([_raw_row(SLD012=7.0, SLD013=9.0)]))
    assert out.loc[0, "sleep_hours_weekday"] == 7.0
    assert out.loc[0, "sleep_hours_weekend"] == 9.0


def test_value_maps():
    raw = pd.DataFrame([
        _raw_row(RIAGENDR=1, DMDEDUC2=1, PAQ605=2, SMQ020=1, SLQ030=3, SLQ120=4, HUQ010=5),
        _raw_row(RIAGENDR=2, DMDEDUC2=5, PAQ605=1, SMQ020=2, SLQ030=0, SLQ120=0, HUQ010=1),
    ])
    out = decode(raw)
    assert out["gender"].tolist() == [0, 1]  # Male 0, Female 1
    assert out["education"].tolist() == [0, 4]  # NHANES 1..5 -> 0..4
    assert out["vigorous_work"].tolist() == [0, 1]  # No 0, Yes 1
    assert out["smoked_100"].tolist() == [1, 0]
    assert out["snoring"].tolist() == [3, 0]
    assert out["daytime_sleepiness"].tolist() == [4, 0]
    assert out["general_health"].tolist() == [5, 1]


def test_target_mapping_and_dropped_rows():
    raw = pd.DataFrame([_raw_row(SLQ050=1), _raw_row(SLQ050=2), _raw_row(SLQ050=7),
                        _raw_row(SLQ050=9), _raw_row(SLQ050=np.nan)])
    out = decode(raw)
    assert out[TARGET].tolist() == [1, 0]


def test_phq_score_needs_all_seven_items():
    complete = _raw_row(**{code: 2 for code in PHQ_ITEMS})
    one_refused = _raw_row(**{**{code: 2 for code in PHQ_ITEMS}, "DPQ010": 7})
    out = decode(pd.DataFrame([complete, one_refused]))
    assert out.loc[0, "phq_score"] == 14
    assert np.isnan(out.loc[1, "phq_score"])


def test_missing_variable_fails_loudly(tmp_path):
    from somnoscreen.data import load_raw

    pd.DataFrame({"SEQN": [1], "RIAGENDR": [1]}).to_csv(tmp_path / "demographics.csv", index=False)
    with pytest.raises(KeyError, match="missing required NHANES variables"):
        load_raw(tmp_path)


@needs_data
def test_row_and_class_counts(dataset):
    assert len(dataset) == 5824
    assert dataset[TARGET].sum() == 1560
    assert (dataset[TARGET] == 0).sum() == 4264


@needs_data
def test_no_nhanes_codes_left(dataset):
    for name in ["education", "snoring", "snort_stop_breathing", "daytime_sleepiness", "vigorous_work",
                 "vigorous_recreation", "smoked_100", "general_health"]:
        allowed = set(FEATURES_BY_NAME[name].value_map.values())
        assert set(dataset[name].dropna().unique()) <= allowed, name
    assert dataset["sitting_minutes"].max() < 7777
    assert dataset["phq_score"].max() <= 21
