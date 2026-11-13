# Heart Disease Prediction — Streamlit

## Deploy on Streamlit Community Cloud

1. Upload **all files and the `artifacts/` folder** in this project to the root of a GitHub repository.
2. In Streamlit Community Cloud, create a new app and select that repository.
3. Set branch to `main` and main file path to `app.py`.
4. Deploy. Streamlit installs dependencies from the root `requirements.txt`.

Expected structure:

```text
Heart_Disease_Prediction/
├── app.py
├── requirements.txt
├── README.md
└── artifacts/
    ├── best_heart_disease_model.joblib
    ├── feature_info.json
    ├── model_metrics.csv
    ├── metrics.json
    └── confusion_matrix.png
```

## Run locally

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

## Model details

The included scikit-learn Pipeline uses Logistic Regression. It was trained on a reproducible 100,000-row sample from the supplied labelled training CSV and evaluated on a separate 50,000-row sample. The model, preprocessing steps, and feature metadata are bundled in `artifacts/`; the CSV is not needed to run the deployed app.

**Educational use only. Not for medical diagnosis or treatment.**
