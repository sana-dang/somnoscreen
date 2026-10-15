"""SomnoScreen web app: a 2-minute sleep-trouble risk screener.

The form is generated from the feature spec (features.py), so answers are encoded exactly as in
training. Nothing the user types is stored or logged.

Run locally:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # lets Streamlit Community Cloud import the package

from somnoscreen import config  # noqa: E402
from somnoscreen.explain import explain_one, plot_contributions  # noqa: E402
from somnoscreen.features import (  # noqa: E402
    FEATURES,
    PHQ_CHOICES,
    PHQ_ITEMS,
    answers_to_frame,
    form_to_answers,
    frequent_phq_answers,
)

SECTIONS = ["About you", "Body measurements", "Sleep", "Lifestyle", "Mood"]
BAND_COLORS = {"Low": "#1baf7a", "Moderate": "#eda100", "High": "#e34948"}


@st.cache_resource
def load_artifacts() -> tuple[object, dict, pd.DataFrame]:
    """Saved model, its metadata and the SHAP background sample (loaded once per server)."""
    model = joblib.load(config.MODEL_PATH)
    metadata = json.loads(config.METADATA_PATH.read_text())
    background = pd.read_csv(config.SHAP_BACKGROUND_PATH)
    return model, metadata, background


def risk_band(probability: float, bands: dict[str, list[float]]) -> str:
    """Name of the band whose [low, high) range contains the probability."""
    for name, (low, high) in bands.items():
        if low <= probability < high or (high == 1.0 and probability == 1.0):
            return name
    return "High"


def ask(feature) -> object:
    """Draw the widget for one feature and return the raw answer."""
    if feature.widget == "select":
        return st.selectbox(feature.question, list(feature.options), index=None,
                            placeholder="Choose an answer", help=feature.help or None)
    optional = " (optional)" if feature.optional else ""
    return st.number_input(
        feature.question + optional,
        min_value=float(feature.min_value), max_value=float(feature.max_value),
        value=None if feature.optional or feature.default is None else float(feature.default),
        step=float(feature.step or 1), format="%.0f" if (feature.step or 1) >= 1 else "%.1f",
        placeholder="Leave blank if unknown" if feature.optional else None, help=feature.help or None,
    )


def draw_form() -> tuple[bool, dict]:
    """The questionnaire, grouped into sections. Returns (submitted, form answers)."""
    form: dict = {"phq": {}}
    with st.form("screen"):
        for section in SECTIONS:
            st.subheader(section)
            if section == "Body measurements":
                left, right = st.columns(2)
                form["height_cm"] = left.number_input("Height (cm)", 100.0, 230.0, None, 1.0, "%.0f",
                                                      placeholder="e.g. 165")
                form["weight_kg"] = right.number_input("Weight (kg)", 30.0, 250.0, None, 0.5, "%.1f",
                                                       placeholder="e.g. 70")
                st.caption("Waist, blood pressure and pulse are optional; blanks are filled with typical "
                           "values. A home blood-pressure monitor shows both blood pressure and pulse.")
            if section == "Mood":
                st.markdown("**Over the last 2 weeks, how often have you been bothered by:**")
                for code, text in PHQ_ITEMS.items():
                    form["phq"][code] = st.radio(text, [t for _, t, _ in PHQ_CHOICES], index=None,
                                                 horizontal=True, key=code)
                continue
            for feature in [f for f in FEATURES if f.section == section and f.widget in ("select", "number")]:
                form[feature.name] = ask(feature)
        submitted = st.form_submit_button("Estimate my risk", type="primary", use_container_width=True)
    return submitted, form


def missing_required(form: dict) -> list[str]:
    """Questions that must be answered before a result is shown."""
    missing = [f.question for f in FEATURES
               if f.widget in ("select", "number") and not f.optional and form.get(f.name) is None]
    if not form.get("height_cm") or not form.get("weight_kg"):
        missing.append("Height and weight")
    if any(v is None for v in form["phq"].values()):
        missing.append("All 7 mood questions")
    return missing


def show_result(form: dict, model, metadata: dict, background: pd.DataFrame) -> None:
    """Risk %, band, recommendation and the per-person 'why' chart."""
    row = answers_to_frame(form_to_answers(form))
    probability = float(model.predict_proba(row)[0, 1])
    threshold = metadata["screening_threshold"]
    band = risk_band(probability, metadata["risk_bands"])

    st.divider()
    st.header("Your result")
    left, right = st.columns([1, 2])
    left.metric("Estimated risk", f"{probability:.0%}")
    left.markdown(
        f"<span style='background:{BAND_COLORS[band]};color:white;padding:4px 12px;border-radius:12px;"
        f"font-weight:600'>{band} risk</span>", unsafe_allow_html=True)
    with right:
        flagged = probability >= threshold
        st.markdown(
            f"**{'Above' if flagged else 'Below'} the screening threshold** ({threshold:.0%}). "
            f"For comparison, {metadata['base_rate']:.0%} of adults in the NHANES data said they had told "
            "a doctor about trouble sleeping."
        )
        if flagged:
            st.markdown("Your answers look similar to those of people who had talked to a doctor about "
                        "trouble sleeping. **Consider raising your sleep with a doctor**, especially if "
                        "poor sleep affects your day.")
        else:
            st.markdown("Your answers look less like those of people who reported sleep trouble. If you "
                        "are still worried about your sleep, it is always fine to talk to a doctor.")

    if frequent_phq_answers(form["phq"]) >= 2:
        st.info("Several of your answers about mood and energy were *more than half the days* or *nearly "
                "every day*. That can be worth talking about with a doctor or counsellor. (This is not "
                "a diagnosis, and this tool does not score depression.)")

    st.subheader("Why this estimate?")
    with st.spinner("Working out which answers mattered most..."):
        contributions, average = explain_one(model, background, row)
    st.pyplot(plot_contributions(contributions, top=10))
    st.caption(f"Each bar shows how much one answer moved your estimate compared with an average adult "
               f"in the data (average estimate {average:.0%}). Red raised it, blue lowered it. "
               "Blank optional answers are filled with typical values.")


def main() -> None:
    """Page layout."""
    st.set_page_config(page_title="SomnoScreen", page_icon="🌙", layout="centered")
    model, metadata, background = load_artifacts()

    st.title("SomnoScreen")
    st.markdown("A 2-minute sleep-trouble risk screener: a short questionnaire and a few body "
                "measurements. No lab tests, no sleep study.")
    st.warning("**Screening tool, not a diagnosis.** It estimates how likely someone with your answers "
               "is to have told a doctor about trouble sleeping, based on a US health survey "
               "(NHANES 2017-18). Your answers are not stored.", icon="⚠️")

    submitted, form = draw_form()
    if submitted:
        missing = missing_required(form)
        if missing:
            st.error("Please answer: " + "; ".join(missing))
        else:
            show_result(form, model, metadata, background)

    test = metadata["metrics"]["held_out_test"]
    with st.expander("About the model"):
        st.markdown(
            f"- **Model:** {metadata['model']} ({metadata['imbalance_strategy']} for class imbalance, "
            f"{metadata['calibration']} calibration), trained on {metadata['n_rows']:,} NHANES adults.\n"
            f"- **Held-out test set:** ROC-AUC {test['ROC_AUC']:.3f}, PR-AUC {test['PR_AUC']:.3f}; at the "
            f"screening threshold it caught {test['Recall']:.0%} of 'Yes' cases with precision "
            f"{test['Precision']:.0%}.\n"
            "- **Target:** self-reported 'ever told a doctor you had trouble sleeping'. It is not a "
            "clinical diagnosis of a sleep disorder.\n"
            "- **Limits:** US survey data (2017-18), associations not causes, survey weights not used."
        )


main()
