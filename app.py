"""Heart Disease Prediction & Analytics — Streamlit application.

Run locally: streamlit run app.py
The trained model and feature metadata are stored in ./artifacts.
Educational project only; not medical advice.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
ART_DIR = BASE_DIR / "artifacts"

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
)

@st.cache_resource
def load_model():
    return joblib.load(ART_DIR / "best_heart_disease_model.joblib")

@st.cache_data
def load_json(filename):
    with (ART_DIR / filename).open("r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_metrics():
    return pd.read_csv(ART_DIR / "model_metrics.csv")

st.title("❤️ Heart Disease Prediction")
st.caption("Machine-learning academic demonstration using the supplied Kaggle-style training dataset. Not a medical diagnosis.")

required = [ART_DIR / "best_heart_disease_model.joblib", ART_DIR / "feature_info.json"]
missing = [p.name for p in required if not p.exists()]
if missing:
    st.error("Required model files are missing: " + ", ".join(missing))
    st.info("Upload the complete artifacts folder from this project to the same GitHub repository as app.py.")
    st.stop()

info = load_json("feature_info.json")
page = st.sidebar.radio("Navigate", ["Predict", "Model performance", "About"])

if page == "Predict":
    st.header("Enter the health-record features")
    st.warning("This is an educational prediction based on a synthetic dataset. It must not be used for diagnosis or treatment decisions.")
    values = {}
    cols = st.columns(3)
    for idx, feature in enumerate(info["columns"]):
        with cols[idx % 3]:
            if feature in info["numeric"]:
                meta = info["numeric"][feature]
                min_v, max_v, default = meta["min"], meta["max"], meta["median"]
                if float(min_v).is_integer() and float(max_v).is_integer():
                    values[feature] = st.number_input(
                        feature, min_value=int(min_v), max_value=int(max_v),
                        value=int(round(default)), step=1
                    )
                else:
                    values[feature] = st.number_input(
                        feature, min_value=float(min_v), max_value=float(max_v),
                        value=float(default), step=0.1, format="%.2f"
                    )
            else:
                meta = info["categorical"][feature]
                options = meta["values"]
                default = meta["mode"]
                values[feature] = st.selectbox(
                    feature, options, index=options.index(default) if default in options else 0
                )

    if st.button("Predict", type="primary", use_container_width=True):
        input_df = pd.DataFrame([values], columns=info["columns"])
        try:
            estimator = load_model()
            prediction = int(estimator.predict(input_df)[0])
            probability = None
            if hasattr(estimator, "predict_proba"):
                probability = float(estimator.predict_proba(input_df)[0, 1])
            st.subheader("Prediction result")
            if prediction == 1:
                st.error("Model prediction: Presence")
            else:
                st.success("Model prediction: Absence")
            if probability is not None:
                st.metric("Estimated model probability of Presence", f"{probability:.1%}")
                st.progress(max(0.0, min(1.0, probability)))
            st.caption("Model output is not a clinical risk score. Consult a qualified healthcare professional for medical concerns.")
        except Exception as exc:
            st.exception(exc)

elif page == "Model performance":
    st.header("Model performance")
    st.write("A Logistic Regression pipeline was trained on a reproducible sample of 100,000 rows and evaluated on a separate 50,000-row test sample.")
    try:
        results = load_metrics()
        st.dataframe(results, use_container_width=True, hide_index=True)
        row = results.iloc[0]
        a, b, c, d = st.columns(4)
        a.metric("Accuracy", f"{row['Accuracy']:.3f}")
        b.metric("Precision", f"{row['Precision']:.3f}")
        c.metric("Recall", f"{row['Recall']:.3f}")
        d.metric("ROC-AUC", f"{row['ROC_AUC']:.3f}")
        cm_path = ART_DIR / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Confusion matrix on the held-out evaluation sample", width=500)
    except Exception as exc:
        st.error(f"Could not load model metrics: {exc}")

else:
    st.header("About this project")
    st.markdown(
        """
        **Project:** Heart Disease Prediction using Machine Learning.

        **Model:** Logistic Regression with preprocessing in a scikit-learn pipeline.

        **Input handling:** Numeric features are imputed and standardized; categorical-coded features are imputed and one-hot encoded. The record identifier is excluded.

        **Target labels:** `Absence` = 0 and `Presence` = 1.

        **Data:** The supplied training CSV contains 630,000 labelled records. The deployed model was trained using a reproducible 100,000-row sample and evaluated using a separate 50,000-row sample.

        **Important:** The dataset is synthetic/educational. Predictions are not medical advice and cannot diagnose heart disease.
        """
    )
    st.caption(f"Model artifact created with scikit-learn {info.get('sklearn_version', 'unknown')}.")
