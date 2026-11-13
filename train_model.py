"""Rebuild the model artifacts from train.csv.

Usage:
    python train_model.py train.csv
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

if len(sys.argv) < 2:
    raise SystemExit("Usage: python train_model.py path/to/train.csv")

csv_path = Path(sys.argv[1])
out_dir = Path(__file__).resolve().parent / "artifacts"
out_dir.mkdir(exist_ok=True)

df = pd.read_csv(csv_path).drop(columns=["id"], errors="ignore").drop_duplicates().reset_index(drop=True)
target = "Heart Disease"
y = df[target].astype(str).str.strip().str.lower().map({"absence": 0, "presence": 1})
if y.isna().any():
    raise ValueError("Unexpected target labels in 'Heart Disease'.")
y = y.astype(int)
X = df.drop(columns=[target])

cat_cols = [
    "Sex", "Chest pain type", "FBS over 120", "EKG results",
    "Exercise angina", "Slope of ST", "Number of vessels fluro", "Thallium"
]
cat_cols = [c for c in cat_cols if c in X.columns]
num_cols = [c for c in X.columns if c not in cat_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
rng = np.random.RandomState(42)
train_idx = rng.choice(len(X_train), min(100000, len(X_train)), replace=False)
test_idx = rng.choice(len(X_test), min(50000, len(X_test)), replace=False)

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])
model = Pipeline([
    ("preprocess", preprocessor),
    ("model", LogisticRegression(max_iter=500, solver="lbfgs"))
])
model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
pred = model.predict(X_test.iloc[test_idx])
prob = model.predict_proba(X_test.iloc[test_idx])[:, 1]
metrics = {
    "Accuracy": float(accuracy_score(y_test.iloc[test_idx], pred)),
    "Precision": float(precision_score(y_test.iloc[test_idx], pred, zero_division=0)),
    "Recall": float(recall_score(y_test.iloc[test_idx], pred, zero_division=0)),
    "F1": float(f1_score(y_test.iloc[test_idx], pred, zero_division=0)),
    "ROC_AUC": float(roc_auc_score(y_test.iloc[test_idx], prob))
}
feature_info = {
    "target": target,
    "target_mapping": {"Absence": 0, "Presence": 1},
    "columns": X.columns.tolist(),
    "numeric": {
        c: {"min": float(X[c].min()), "max": float(X[c].max()), "median": float(X[c].median())}
        for c in num_cols
    },
    "categorical": {
        c: {"values": sorted(int(v) for v in X[c].dropna().unique()), "mode": int(X[c].mode().iloc[0])}
        for c in cat_cols
    },
    "best_model": "Logistic Regression (100,000-row training sample)",
    "sklearn_version": __import__("sklearn").__version__,
    "training_rows": int(len(train_idx)),
    "evaluation_rows": int(len(test_idx))
}
joblib.dump(model, out_dir / "best_heart_disease_model.joblib", compress=3)
(out_dir / "feature_info.json").write_text(json.dumps(feature_info, indent=2))
(out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
pd.DataFrame([{"Model": "Logistic Regression", "Training rows": len(train_idx),
               "Evaluation rows": len(test_idx), **metrics}]).to_csv(out_dir / "model_metrics.csv", index=False)

cm = confusion_matrix(y_test.iloc[test_idx], pred)
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(cm)
ax.set_xticks([0, 1], ["Absence", "Presence"])
ax.set_yticks([0, 1], ["Absence", "Presence"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
for (i, j), value in np.ndenumerate(cm):
    ax.text(j, i, str(value), ha="center", va="center")
fig.tight_layout()
fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
plt.close(fig)
print("Saved model artifacts to", out_dir)
print(metrics)
