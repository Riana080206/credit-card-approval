"""
Credit Card Approval Prediction - Model Training Script
College Project: Cloud Computing

This script:
1. Loads or downloads the UCI Credit Approval dataset (crx.data).
2. Cleans and preprocesses numerical and categorical features.
3. Builds an end-to-end Scikit-Learn Pipeline (Imputation, Scaling, One-Hot Encoding).
4. Tunes Logistic Regression hyperparameters via GridSearchCV (C, class_weight, solver).
5. Evaluates model performance on a holdout test set (Accuracy, Precision, Recall, F1, Confusion Matrix).
6. Persists the best trained pipeline to models/model.joblib and metrics to models/metrics.json.
"""

import os
import sys
import json
import urllib.request
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Project directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATASET_PATH = os.path.join(DATA_DIR, "crx.data")
MODEL_PATH = os.path.join(MODELS_DIR, "model.joblib")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

# Public UCI Credit Approval Dataset URL
UCI_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data"

# Standard UCI Credit Screening column definitions
COLUMN_NAMES = [
    "Gender",          # A1: 'b', 'a'
    "Age",             # A2: continuous
    "Debt",            # A3: continuous
    "Married",         # A4: 'u', 'y', 'l', 't'
    "BankCustomer",    # A5: 'g', 'p', 'gg'
    "EducationLevel",  # A6: 'c', 'd', 'cc', 'i', 'j', 'k', 'm', 'r', 'q', 'w', 'x', 'e', 'aa', 'ff'
    "Ethnicity",       # A7: 'v', 'h', 'bb', 'j', 'n', 'z', 'dd', 'ff', 'o'
    "YearsEmployed",   # A8: continuous
    "PriorDefault",    # A9: 't', 'f'
    "Employed",        # A10: 't', 'f'
    "CreditScore",     # A11: continuous / delinquency count
    "DriversLicense",  # A12: 't', 'f'
    "Citizen",         # A13: 'g', 'p', 's'
    "ZipCode",         # A14: continuous / postal code
    "Income",          # A15: continuous
    "ApprovalStatus"   # A16: '+' (Approved), '-' (Rejected)
]

NUMERICAL_COLS = ["Age", "Debt", "YearsEmployed", "CreditScore", "ZipCode", "Income"]
CATEGORICAL_COLS = [
    "Gender", "Married", "BankCustomer", "EducationLevel", 
    "Ethnicity", "PriorDefault", "Employed", "DriversLicense", "Citizen"
]


def ensure_dataset():
    """Ensure the dataset exists in data/ directory; download if missing."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(DATASET_PATH) and os.path.getsize(DATASET_PATH) > 0:
        print(f"[INFO] Using existing dataset at: {DATASET_PATH}")
        return True

    print(f"[INFO] Dataset not found locally. Attempting automatic download from UCI...")
    print(f"       URL: {UCI_DATA_URL}")
    try:
        urllib.request.urlretrieve(UCI_DATA_URL, DATASET_PATH)
        print(f"[SUCCESS] Downloaded dataset to {DATASET_PATH}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download dataset automatically: {e}")
        print("\nManual Instructions:")
        print("1. Download 'crx.data' from https://archive.ics.uci.edu/dataset/27/credit+approval")
        print(f"2. Place the file into the project folder at: {DATASET_PATH}")
        print("3. Re-run: python train_model.py\n")
        return False


def load_and_preprocess_data():
    """Load dataset, map target labels, and prepare training/testing sets."""
    print("[INFO] Reading dataset into Pandas DataFrame...")
    # Missing values in UCI Credit Approval are represented by '?'
    df = pd.read_csv(DATASET_PATH, names=COLUMN_NAMES, na_values="?", skipinitialspace=True)
    print(f"[INFO] Raw dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Map target variable: '+' -> 1 (Approved), '-' -> 0 (Rejected)
    df["ApprovalStatus"] = df["ApprovalStatus"].map({"+": 1, "-": 0})
    df = df.dropna(subset=["ApprovalStatus"])

    # Separate feature matrix X and target y
    X = df.drop(columns=["ApprovalStatus"])
    y = df["ApprovalStatus"].astype(int)

    # Convert numeric columns to numeric dtype
    for col in NUMERICAL_COLS:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    # Stratified Train/Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[INFO] Split: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    return X_train, X_test, y_train, y_test


def build_pipeline():
    """Construct Scikit-Learn ColumnTransformer and Pipeline."""
    # Numerical pipeline: impute missing with median, then standardize
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical pipeline: impute missing with most frequent, then one-hot encode
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, NUMERICAL_COLS),
        ("cat", cat_pipeline, CATEGORICAL_COLS)
    ])

    # Full modeling pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])

    return pipeline


def train_and_optimize(X_train, y_train):
    """Perform Hyperparameter Optimization using GridSearchCV."""
    pipeline = build_pipeline()

    # Hyperparameter search space
    param_grid = {
        "classifier__C": [0.01, 0.1, 1.0, 10.0],
        "classifier__class_weight": [None, "balanced"],
        "classifier__solver": ["liblinear", "lbfgs"]
    }

    print("[INFO] Starting GridSearchCV hyperparameter optimization...")
    print(f"       Grid parameters: {param_grid}")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=1  # Safe cross-platform execution (Windows/Linux/Cloud)
    )

    grid_search.fit(X_train, y_train)
    print(f"[SUCCESS] Grid search complete.")
    print(f"          Best Parameters: {grid_search.best_params_}")
    print(f"          Best 5-Fold CV Accuracy: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def evaluate_model(model, X_test, y_test, best_params, cv_score):
    """Compute comprehensive test metrics and confusion matrix."""
    print("\n[INFO] Evaluating model on unseen test set...")
    y_pred = model.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    report = classification_report(y_test, y_pred, target_names=["Rejected (0)", "Approved (1)"], output_dict=True)

    metrics = {
        "model_name": "Logistic Regression with GridSearchCV",
        "dataset": "UCI Credit Approval (crx.data)",
        "test_samples": int(len(y_test)),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "cv_score": round(float(cv_score), 4),
        "best_hyperparameters": best_params,
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp
        },
        "classification_report": report
    }

    print("=" * 60)
    print("           MODEL TEST EVALUATION METRICS           ")
    print("=" * 60)
    print(f" Accuracy:          {acc * 100:.2f}%")
    print(f" Precision:         {prec * 100:.2f}%")
    print(f" Recall:            {rec * 100:.2f}%")
    print(f" F1-Score:          {f1 * 100:.2f}%")
    print(f" 5-Fold CV Score:   {cv_score * 100:.2f}%")
    print("-" * 60)
    print(f" Confusion Matrix:  [[TN={tn}, FP={fp}], [FN={fn}, TP={tp}]]")
    print("=" * 60)

    return metrics


def save_artifacts(model, metrics):
    """Save trained model pipeline and evaluation metrics."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Save model pipeline with Joblib
    joblib.dump(model, MODEL_PATH)
    print(f"[SUCCESS] Trained model pipeline saved to: {MODEL_PATH}")

    # Save metrics JSON
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[SUCCESS] Model metrics saved to: {METRICS_PATH}")


def main():
    print("\n" + "=" * 60)
    print(" CREDIT CARD APPROVAL PREDICTION - MODEL TRAINING PIPELINE ")
    print("=" * 60)

    if not ensure_dataset():
        sys.exit(1)

    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    best_model, best_params, cv_score = train_and_optimize(X_train, y_train)
    metrics = evaluate_model(best_model, X_test, y_test, best_params, cv_score)
    save_artifacts(best_model, metrics)

    print("\n[ALL DONE] Model training pipeline completed successfully!\n")


if __name__ == "__main__":
    main()
