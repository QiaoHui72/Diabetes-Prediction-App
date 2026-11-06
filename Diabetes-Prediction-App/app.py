"""AI Diabetes Risk Assessment - Streamlit app (scikit-learn).

What it does
- Loads the diabetes dataset (diabetes.csv, exported from diabetes_prediction.ipynb)
- Cleans it (zeros that mean "missing" -> filled with the median) and scales it
- Trains and tunes six models using stratified cross-validation
- Lets the user enter Glucose, Blood Pressure, Height, Weight (BMI) and Age, with validation
- Shows the prediction, the model used, the dataset and the model results

Run:  streamlit run app.py
"""
import os
import pickle
import hashlib
import importlib.metadata
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Diabetes Risk Assessment", page_icon="🩺", layout="centered")

# ----------------------------------------------------------------------------
# Apple-style look: soft grey page, white rounded cards, one blue accent
# ----------------------------------------------------------------------------
st.markdown(
    """
<style>
html, body, .stApp, .stApp input, .stApp button {
    font-family: Arial, Helvetica, sans-serif;
    color: #14233e;
}
.stApp {
    background: radial-gradient(ellipse at 7% 5%, #b9dbfb 0, transparent 22%),
                radial-gradient(ellipse at 95% 98%, #bed6f8 0, transparent 24%),
                linear-gradient(135deg, #e4efff, #f3f7ff);
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1240px; padding: 3.5rem 2rem 6rem; }
h1, h2, h3 { color: #101e35; letter-spacing: -.025em; }
.hero { text-align: center; padding: 0 0 28px; }
.hero h1 {
    font-size: clamp(32px, 4vw, 60px); line-height: 1.15; font-weight: 700;
    padding: 0; margin: 0 0 12px;
    background: linear-gradient(90deg, #1247ba, #0869f6);
    background-clip: text; -webkit-background-clip: text; color: transparent;
}
.hero p { color: #536984; font-size: clamp(17px, 1.6vw, 25px); margin: 0; }
/* Use test IDs as well as BaseWeb attributes across Streamlit versions. */
[data-testid="stTabs"] [role="tablist"] {
    justify-content: center; gap: 12px; border: 0; margin-bottom: 8px;
}
[data-testid="stTabs"] [role="tab"] {
    padding: 12px 30px; height: 60px; border-radius: 999px; color: #536984;
}
[data-testid="stTabs"] [role="tab"] p { font-size: 21px; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: white; color: #075ced; box-shadow: 0 4px 12px #1165df12;
}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none; }
.st-key-card_details, .st-key-card_summary, .card {
    background: white; border: 0 !important; border-radius: 28px;
    padding: 30px 38px; box-shadow: 0 8px 28px #2a70cc05;
}
.st-key-card_details h3 { font-size: 30px; padding: 0 0 8px; }
[data-testid="stWidgetLabel"] p { font-size: 18px; color: #14233e; }
[data-baseweb="input"], [data-baseweb="select"] > div {
    background: white; border: 1px solid #c6d8fb; border-radius: 16px;
    min-height: 60px; box-shadow: none;
}
[data-baseweb="input"] input { font-size: 21px; padding-left: 18px; }
[data-baseweb="select"] > div { font-size: 21px; padding-left: 18px; }
[data-baseweb="input"]:focus-within, [data-baseweb="select"]:focus-within > div {
    border-color: #0866ee; box-shadow: 0 0 0 3px #0866ee15;
}
[data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] { display: none; }
.bmi-note { color: #607493; font-size: 18px; margin: 0; }
.bmi-note b, .result .sub b { color: #0755d9; }
[data-testid="stExpander"] { border-top: 1px solid #e2eafa; }
[data-testid="stExpander"] details { border: 0; border-radius: 0; }
[data-testid="stExpander"] summary p { color: #607493; font-size: 17px; }
.stButton > button {
    background: #0863ed; color: white; border: 0; border-radius: 999px;
    min-height: 65px; width: 100%; box-shadow: 0 9px 19px #0863ed35;
}
.stButton > button p { color: white; font-size: 23px; }
.stButton > button:hover { background: #0756d2; color: white; }
.stButton > button:focus-visible { outline: 3px solid #85b5ff; outline-offset: 3px; }
.card { margin: 8px 0; }
.result { text-align: center; }
.result .label { font-size: clamp(26px, 3vw, 40px); font-weight: 700; }
.result .sub { color: #607493; font-size: 21px; margin-top: 8px; }
.low { color: #198338; } .high { color: #c32b34; }
.bar { height: 11px; border-radius: 99px; background: #dbe7f8; overflow: hidden; margin-top: 25px; }
.bar > div { height: 100%; width: var(--w); border-radius: 99px; background: linear-gradient(90deg,#0863ed,#48a0ff); }
.summary-line { font-size: 21px; margin: 0 0 14px; }
.input-tiles { display: grid; grid-template-columns: repeat(4,1fr); gap: 16px; margin: 20px 0 24px; }
.input-tile { background: #eef5ff; border-radius: 20px; padding: 17px 20px; }
.input-tile span { display: block; color: #607493; font-size: 18px; }
.input-tile b { color: #0953d4; font-size: 27px; }
.small { color: #607493; font-size: 16px; margin: 0; }
@media (max-width: 640px) {
    .block-container { padding: 2rem 1rem 3rem; }
    .st-key-card_details, .st-key-card_summary, .card { padding: 22px 20px; border-radius: 22px; }
    [data-testid="stTabs"] [role="tab"] { padding: 8px 20px; height: 48px; }
    [data-testid="stTabs"] [role="tab"] p { font-size: 17px; }
    .input-tiles { grid-template-columns: repeat(2,1fr); gap: 10px; }
}
</style>
""",
    unsafe_allow_html=True,
)

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diabetes.csv")

from modeling import FEATURES, ZERO_MEANS_MISSING, clean
import charts

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
@st.cache_data
def read_csv(path_or_file):
    return pd.read_csv(path_or_file)


def get_data():
    """Use diabetes.csv next to this file, or let the user upload it."""
    if os.path.exists(DATA_FILE):
        return read_csv(DATA_FILE)
    st.info(
        "I could not find **diabetes.csv** next to the app. "
        "Upload it below (export it from your notebook with "
        "`df.to_csv('diabetes.csv', index=False)`)."
    )
    uploaded = st.file_uploader("Upload diabetes.csv", type="csv")
    if uploaded is None:
        st.stop()
    return read_csv(uploaded)


@st.cache_resource(show_spinner="Loading saved models...")
def load_models(bundle_bytes, dataset_bytes):
    # This bundled file is created by build_models.py, not uploaded by visitors.
    bundle = pickle.loads(bundle_bytes)
    if hashlib.sha256(dataset_bytes).hexdigest() != bundle['dataset_sha256']:
        raise ValueError("The dataset has changed. Run python build_models.py locally and upload the updated trained_models.pkl.")
    for name, version in bundle['versions'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError(f"Install the supplied requirements.txt: {name} must be {version} for these saved models.")
    return bundle['models']

@st.cache_data(show_spinner=False)
def dataset_chart(kind, data, columns=None):
    if kind == "pairs":
        return charts.pairs(data, columns)
    return getattr(charts, kind)(data)

@st.cache_data(show_spinner=False)
def model_chart(kind, bundle_bytes, model_name=None):
    saved_results = pickle.loads(bundle_bytes)['models'][0]
    if kind == "confusion":
        return charts.confusion(saved_results[model_name]['confusion'], model_name)
    return getattr(charts, kind)(saved_results)

# ----------------------------------------------------------------------------
# Load data and models
# ----------------------------------------------------------------------------
st.markdown(
    '<div class="hero"><h1>AI Diabetes Risk Assessment</h1>'
    "<p>Enter a few health details and get an instant risk estimate.</p></div>",
    unsafe_allow_html=True,
)

df = get_data()
target = "Outcome" if "Outcome" in df.columns else df.columns[-1]
needed = ["Glucose", "BloodPressure", "BMI", "Age"]
missing_cols = [c for c in needed if c not in df.columns]
if missing_cols:
    st.error(f"The dataset is missing these columns: {', '.join(missing_cols)}")
    st.stop()

try:
    root = Path(__file__).resolve().parent
    results, best_name, feature_names = load_models(
        (root / "trained_models.pkl").read_bytes(), (root / "diabetes.csv").read_bytes())
except (OSError, ValueError) as exc:
    st.error(f"Could not load saved models: {exc}")
    st.info("Upload trained_models.pkl, diabetes.csv and requirements.txt from the same updated ZIP. Training is not run on the web server.")
    st.stop()
clean_df = clean(df)

tab_predict, tab_data, tab_models = st.tabs(["Predict", "Dataset", "Models"])

# ----------------------------------------------------------------------------
# Predict tab
# ----------------------------------------------------------------------------
with tab_predict:
    with st.container(border=True, key="card_details"):
        st.subheader("Your details")

        model_name = st.selectbox(
            "Model",
            list(results.keys()),
            index=list(results.keys()).index(best_name),
            help="The model with the highest cross-validation accuracy is selected by default.",
        )

        c1, c2 = st.columns(2)
        glucose = c1.number_input("Glucose (mg/dL)", min_value=40.0, max_value=400.0, value=110.0, step=1.0)
        bp = c2.number_input(
            "Blood pressure, diastolic (mmHg)", min_value=30.0, max_value=140.0, value=70.0, step=1.0
        )

        c3, c4 = st.columns(2)
        height_cm = c3.number_input("Height (cm)", min_value=100.0, max_value=230.0, value=170.0, step=1.0, format="%.0f")
        weight_kg = c4.number_input("Weight (kg)", min_value=25.0, max_value=250.0, value=70.0, step=0.5, format="%.1f")
        calculated_bmi = weight_kg / ((height_cm / 100) ** 2)
        st.markdown(f'<p class="bmi-note">Your BMI: <b>{calculated_bmi:.1f}</b></p>', unsafe_allow_html=True)
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=30, step=1)

        with st.expander("Optional details"):
            manual_bmi = st.checkbox("I already know my BMI")
            bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0, step=0.1) if manual_bmi else calculated_bmi
            st.caption("The model uses glucose, blood pressure, BMI and age. Height and weight calculate BMI unless you enter it here.")
        extra_values = {}
        predict_clicked = st.button("Predict", use_container_width=True)

    if predict_clicked:
        # ---- input validation ----
        errors, warnings = [], []
        if not 10 <= bmi <= 70:
            errors.append(f"BMI of {bmi:.1f} is not realistic. Please check height and weight.")
        if not (glucose > 0 and bp > 0 and age > 0):
            errors.append("Glucose, blood pressure and age must be above zero.")
        for label, value, col in [("Glucose", glucose, "Glucose"), ("Blood pressure", bp, "BloodPressure"),
                                  ("BMI", bmi, "BMI"), ("Age", age, "Age")]:
            lo, hi = clean_df[col].min(), clean_df[col].max()
            if value < lo or value > hi:
                warnings.append(
                    f"{label} ({value:.1f}) is outside the range in the training data "
                    f"({lo:.0f} to {hi:.0f}), so the estimate may be less reliable."
                )

        for e in errors:
            st.error(e)
        for w in warnings:
            st.warning(w)

        if not errors:
            values = {"Glucose": glucose, "BloodPressure": bp, "BMI": bmi, "Age": age, **extra_values}
            row = pd.DataFrame([[values[f] for f in feature_names]], columns=feature_names)
            pipe = results[model_name]["pipeline"]
            prob = float(pipe.predict_proba(row)[0][1])
            likely = prob >= 0.5

            css = "high" if likely else "low"
            text = "Likely to have diabetes" if likely else "Not likely to have diabetes"
            pct = min(max(prob, 0.0), 1.0) * 100
            st.markdown(
                f'<div class="card result"><div class="label {css}">{text}</div>'
                f'<div class="sub">Estimated risk: <b>{pct:.0f}%</b></div>'
                f'<div class="bar"><div style="--w:{pct:.0f}%"></div></div></div>',
                unsafe_allow_html=True,
            )

            with st.container(border=True, key="card_summary"):
                st.markdown(
                    f'<p class="summary-line"><b>Model used:</b> {model_name}</p>'
                    f'<p class="summary-line"><b>Model accuracy on test data:</b> about {results[model_name]["test_accuracy"] * 100:.0f}%</p>'
                    '<div class="input-tiles">'
                    f'<div class="input-tile"><span>Glucose</span><b>{glucose:g}</b></div>'
                    f'<div class="input-tile"><span>Blood pressure</span><b>{bp:g}</b></div>'
                    f'<div class="input-tile"><span>BMI</span><b>{bmi:.1f}</b></div>'
                    f'<div class="input-tile"><span>Age</span><b>{age}</b></div></div>'
                    '<p class="small">This is a learning project, not medical advice. '
                    'Please talk to a doctor about your health.</p>',
                    unsafe_allow_html=True,
                )

# ----------------------------------------------------------------------------
# Dataset tab
# ----------------------------------------------------------------------------
with tab_data:
    st.subheader("The dataset")
    m1, m2, m3 = st.columns(3)
    m1.metric("Rows", f"{len(df):,}")
    m2.metric("Columns", df.shape[1])
    m3.metric("With diabetes", f"{df[target].mean() * 100:.0f}%")

    st.markdown("**Data table**")
    st.dataframe(df, height=300)

    st.markdown("**Summary numbers**")
    st.dataframe(df.describe().T.round(2))

    st.markdown("**Missing values**")
    zero_counts = {c: int((df[c] == 0).sum()) for c in ZERO_MEANS_MISSING if c in df.columns}
    st.caption(
        "In this dataset, a 0 in these columns is impossible (for example, a blood pressure of 0), "
        "so it is treated as missing and filled with the median."
    )
    st.dataframe(
        pd.DataFrame({"Zeros treated as missing": zero_counts})
    )

    st.markdown("**Diabetes vs no diabetes**")
    counts = df[target].value_counts().rename({0: "No diabetes", 1: "Diabetes"})
    st.image(dataset_chart("distribution", df), use_container_width=True)
    st.subheader("Correlation heatmap")
    st.caption("Zeros representing missing measurements are excluded. Each correlation uses available pairs of values; correlation does not imply causation.")
    st.image(dataset_chart("correlation", df), use_container_width=True)
    st.subheader("Feature relationships")
    pair_columns = st.multiselect("Features to compare (choose 2–5)",
        [c for c in df.select_dtypes(include="number").columns if c != target],
        default=FEATURES, max_selections=5)
    st.caption("Each point is a record. Blue: no diabetes; orange: diabetes. The diagonal shows distributions. Missing measurements are omitted from the relevant panels.")
    if len(pair_columns) >= 2 and st.checkbox("Show pair plot", value=False):
        st.image(dataset_chart("pairs", df, pair_columns), use_container_width=True)
    elif len(pair_columns) < 2:
        st.info("Choose at least two features to show the pair plot.")

# ----------------------------------------------------------------------------
# Models tab
# ----------------------------------------------------------------------------
with tab_models:
    st.subheader("Model results")
    st.write(
        "Six models use glucose, blood pressure, BMI and age. We reserve 20% of the data "
        "for testing and tune on the other 80% with shuffled, stratified 5-fold cross-validation. "
        "Imputation and scaling are fitted inside each fold. The default model is selected by "
        "cross-validation accuracy, never by its test score."
    )
    st.caption("The original four search spaces are compared on the same folds. Expanded searches retain the baseline if it scores higher. CV scores are tuning scores and may be optimistic; the held-out test provides the separate evaluation. No accuracy improvement is guaranteed on new data.")
    table = pd.DataFrame([{
        "Model": name,
        "Original search CV (%)": round(r['baseline_cv']*100,2) if r['baseline_cv'] is not None else None,
        "Tuned CV (%)": round(r['cv_accuracy']*100,2),
        "CV fold SD (points)": round(r['cv_std']*100,2),
        "Test accuracy (%)": round(r['test_accuracy']*100,2),
        "Recall (%)": round(r['recall']*100,2),
        "Precision (%)": round(r['precision']*100,2),
        "F1": round(r['f1'],3), "ROC AUC": round(r['auc'],3),
    } for name,r in results.items()])
    st.dataframe(table, hide_index=True)
    st.download_button("Download evaluation results (.csv)", table.to_csv(index=False), "model_evaluation.csv", "text/csv")
    st.image(model_chart("comparison", (root / "trained_models.pkl").read_bytes()), use_container_width=True)
    st.success(f"Selected by cross-validation: **{best_name}**")
    st.subheader("Confusion matrix")
    pick = st.selectbox("Show for", list(results), index=list(results).index(best_name), key="cm")
    r = results[pick]
    m1,m2,m3 = st.columns(3)
    m1.metric("Test accuracy", f"{r['test_accuracy']:.1%}")
    m2.metric("Diabetes recall", f"{r['recall']:.1%}")
    m3.metric("Diabetes precision", f"{r['precision']:.1%}")
    st.image(model_chart("confusion", (root / "trained_models.pkl").read_bytes(), pick), use_container_width=True)
    st.caption("Rows are actual outcomes; columns are predictions. False negatives are diabetes records predicted as no diabetes. Recall measures how many actual diabetes records were detected.")
    with st.expander("Selected model settings"):
        st.json(r['best_params'])
    st.subheader("ROC curves")
    st.image(model_chart("roc_chart", (root / "trained_models.pkl").read_bytes()), use_container_width=True)
    st.caption("AUC measures ranking performance across thresholds. The dashed line represents random ranking. Predictions use the original 0.5 probability threshold.")

    st.download_button(
        "Download best model (.pkl)",
        data=pickle.dumps(results[best_name]["pipeline"]),
        file_name="Diabetesmodel.pkl",
        mime="application/octet-stream",
    )
