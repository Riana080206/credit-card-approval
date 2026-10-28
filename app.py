"""
Credit Card Approval Prediction - Flask Web Application
College Project: Cloud Computing

This backend service provides:
- GET /        : Interactive Dashboard with live model performance & prediction form
- POST /predict: Validates input, executes trained ML pipeline, returns prediction
- GET /health  : Health-check endpoint for Cloud Run container monitoring
- GET /metrics : Model evaluation metrics as JSON
"""

import os
import sys
import time
import uuid
import json
import logging
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

# Set up clean standard output logging (Compatible with Google Cloud Logging)
class RequestIdFormatter(logging.Formatter):
    """Custom Formatter that guarantees a request_id is present on all log records."""
    def format(self, record):
        if not hasattr(record, "request_id"):
            record.request_id = "SYSTEM"
        return super().format(record)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(
    RequestIdFormatter(
        "%(asctime)s [%(levelname)s] [ReqID: %(request_id)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.handlers = [handler]
logger = logging.getLogger("credit_approval_app")

# Directory & Artifact paths (relative paths for cloud portability)
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

app = Flask(__name__)

# Expected feature schema (matches UCI crx.data features)
NUMERICAL_COLS = ["Age", "Debt", "YearsEmployed", "CreditScore", "ZipCode", "Income"]
CATEGORICAL_COLS = [
    "Gender", "Married", "BankCustomer", "EducationLevel",
    "Ethnicity", "PriorDefault", "Employed", "DriversLicense", "Citizen"
]
ALL_COLS = [
    "Gender", "Age", "Debt", "Married", "BankCustomer", "EducationLevel",
    "Ethnicity", "YearsEmployed", "PriorDefault", "Employed", "CreditScore",
    "DriversLicense", "Citizen", "ZipCode", "Income"
]

# Global cache for trained pipeline and metrics
_trained_model = None
_cached_metrics = None


def get_model():
    """Load model pipeline from disk if not already in memory."""
    global _trained_model
    if _trained_model is None and MODEL_PATH.exists():
        try:
            _trained_model = joblib.load(MODEL_PATH)
            logger.info("Successfully loaded trained ML model pipeline.", extra={"request_id": "INIT"})
        except Exception as e:
            logger.error(f"Error loading model from {MODEL_PATH}: {e}", extra={"request_id": "INIT"})
            _trained_model = None
    return _trained_model


def get_metrics():
    """Load evaluation metrics JSON if available."""
    global _cached_metrics
    if _cached_metrics is None and METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r") as f:
                _cached_metrics = json.load(f)
        except Exception as e:
            logger.error(f"Error reading metrics JSON: {e}", extra={"request_id": "INIT"})
            _cached_metrics = None
    return _cached_metrics


@app.route("/", methods=["GET"])
def index():
    """Render main application dashboard."""
    model = get_model()
    metrics = get_metrics()
    model_loaded = model is not None

    return render_template(
        "index.html",
        model_loaded=model_loaded,
        metrics=metrics
    )


@app.route("/health", methods=["GET"])
def health():
    """Health-check endpoint for Cloud Run monitoring."""
    model = get_model()
    return jsonify({
        "status": "healthy",
        "service": "credit-card-approval-prediction",
        "model_loaded": model is not None,
        "cloud_ready": True,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }), 200


@app.route("/metrics", methods=["GET"])
def metrics():
    """Return model evaluation metrics as JSON."""
    metrics_data = get_metrics()
    if metrics_data:
        return jsonify(metrics_data), 200
    return jsonify({
        "error": "Model has not been trained yet. Run 'python train_model.py' to generate metrics."
    }), 503


@app.route("/predict", methods=["POST"])
def predict():
    """Validate input features, run prediction, and log sanitized request stats."""
    start_time = time.time()
    req_id = str(uuid.uuid4())[:8]

    # Verify model readiness
    model = get_model()
    if model is None:
        duration_ms = round((time.time() - start_time) * 1000, 2)
        logger.warning(
            f"Prediction failed: Model not trained or missing | Status: 503 | Duration: {duration_ms}ms",
            extra={"request_id": req_id}
        )
        return jsonify({
            "error": "The model is not available. Please run 'python train_model.py' first.",
            "request_id": req_id
        }), 503

    # Support JSON or form data payloads
    if request.is_json:
        data = request.get_json(silent=True) or {}
    else:
        data = request.form.to_dict()

    # Input validation
    missing_fields = []
    parsed_features = {}

    # Validate numerical fields
    for col in NUMERICAL_COLS:
        val = data.get(col)
        if val is None or str(val).strip() == "":
            missing_fields.append(col)
        else:
            try:
                num_val = float(val)
                if num_val < 0:
                    return jsonify({
                        "error": f"Value for '{col}' cannot be negative ({num_val}).",
                        "request_id": req_id
                    }), 400
                parsed_features[col] = num_val
            except (ValueError, TypeError):
                return jsonify({
                    "error": f"Invalid number format for '{col}': received '{val}'.",
                    "request_id": req_id
                }), 400

    # Validate categorical fields
    for col in CATEGORICAL_COLS:
        val = data.get(col)
        if val is None or str(val).strip() == "":
            missing_fields.append(col)
        else:
            parsed_features[col] = str(val).strip()

    if missing_fields:
        duration_ms = round((time.time() - start_time) * 1000, 2)
        logger.warning(
            f"Validation failed: Missing fields: {missing_fields} | Status: 400 | Duration: {duration_ms}ms",
            extra={"request_id": req_id}
        )
        return jsonify({
            "error": f"Missing required fields: {', '.join(missing_fields)}",
            "missing_fields": missing_fields,
            "request_id": req_id
        }), 400

    # Create single-row DataFrame matching training feature schema
    input_df = pd.DataFrame([parsed_features])[ALL_COLS]

    try:
        # Run Scikit-Learn Pipeline (Imputation -> Scaling/Encoding -> LogisticRegression)
        pred_class = int(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
        prob_approved = float(probabilities[1])
        prob_rejected = float(probabilities[0])

        is_approved = bool(pred_class == 1)
        result_label = "Likely Approved" if is_approved else "Likely Rejected"
        confidence = prob_approved if is_approved else prob_rejected

        duration_ms = round((time.time() - start_time) * 1000, 2)

        # Log prediction result (No PII or sensitive applicant data)
        logger.info(
            f"Prediction: {result_label} (Confidence: {confidence * 100:.1f}%) | "
            f"Status: 200 | Duration: {duration_ms}ms",
            extra={"request_id": req_id}
        )

        return jsonify({
            "status": "success",
            "request_id": req_id,
            "prediction": result_label,
            "is_approved": is_approved,
            "confidence": round(confidence, 4),
            "confidence_percent": f"{confidence * 100:.1f}%",
            "probability_approved": round(prob_approved, 4),
            "probability_rejected": round(prob_rejected, 4),
            "duration_ms": duration_ms,
            "disclaimer": "Educational estimates only, not real financial lending decisions."
        }), 200

    except Exception as e:
        duration_ms = round((time.time() - start_time) * 1000, 2)
        logger.error(
            f"Internal error during prediction: {e} | Status: 500 | Duration: {duration_ms}ms",
            extra={"request_id": req_id}
        )
        return jsonify({
            "error": "An internal error occurred during prediction processing.",
            "request_id": req_id
        }), 500


if __name__ == "__main__":
    # Cloud Run reads PORT environment variable, defaults to 8080
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting Credit Card Approval Web Application on 0.0.0.0:{port}", extra={"request_id": "STARTUP"})
    app.run(host="0.0.0.0", port=port, debug=False)
