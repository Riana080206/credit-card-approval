# Credit Card Approval Prediction

A beginner-friendly, end-to-end Machine Learning web application built with **Python**, **Scikit-learn**, and **Flask** for a college project in **Cloud Computing**.

The application predicts whether a credit card application is **Likely Approved** or **Likely Rejected** using **Logistic Regression with hyperparameter optimization (GridSearchCV)** trained on the real-world **UCI Credit Approval Dataset (`crx.data`)**.

---

## 1. Project Objectives
- **Machine Learning Core**: Clean, preprocess, and train a Logistic Regression model using an automated `Pipeline` (Median/Frequent Imputation, Standard Scaling, One-Hot Encoding).
- **Hyperparameter Optimization**: Systematically evaluate combinations of regularization strength `C`, penalty solvers, and class weighting using `GridSearchCV` with 5-fold cross-validation.
- **Web Interface**: Deliver an intuitive dashboard displaying test evaluation metrics, confusion matrix, and an interactive prediction form with demo profile presets.
- **Cloud Readiness**: Architect the application for containerized deployment on **Google Cloud Run** (dynamic `$PORT`, `0.0.0.0` binding, stdout logging without PII, stateless execution).

---

## 2. Directory Structure

```
credit-card-approval/
├── app.py                  # Flask web service, routes, API endpoints, and cloud logging
├── train_model.py          # ML pipeline: data ingestion, preprocessing, GridSearchCV, evaluation, model export
├── requirements.txt        # Python dependency manifest
├── README.md               # Setup guide, execution commands, and project documentation
├── data/
│   ├── .gitkeep            # Version control placeholder
│   └── crx.data            # UCI Credit Approval dataset (auto-downloaded by train_model.py)
├── models/
│   ├── .gitkeep            # Version control placeholder
│   ├── model.joblib        # Best trained Scikit-learn Pipeline persisted with Joblib
│   └── metrics.json        # Test evaluation metrics (Accuracy, F1, Confusion Matrix, etc.)
├── templates/
│   └── index.html          # Modern frontend dashboard
└── static/
    └── style.css           # Vanilla CSS design system
```

---

## 3. Machine Learning Details & Actual Performance

The model is trained on the **UCI Credit Approval dataset (`crx.data`)** with an 80/20 stratified train/test split (552 training samples, 138 holdout test samples).

### Data Preprocessing Pipeline
- **Numerical Features** (`Age`, `Debt`, `YearsEmployed`, `CreditScore`, `ZipCode`, `Income`):
  - Missing values imputed using the **median**.
  - Normalized using `StandardScaler` ($\mu = 0, \sigma = 1$).
- **Categorical Features** (`Gender`, `Married`, `BankCustomer`, `EducationLevel`, `Ethnicity`, `PriorDefault`, `Employed`, `DriversLicense`, `Citizen`):
  - Missing values imputed using the **most frequent** value.
  - Encoded with `OneHotEncoder(handle_unknown='ignore')`.

### Hyperparameter Optimization (GridSearchCV)
- **Tuned Parameters**:
  - `classifier__C`: `[0.01, 0.1, 1.0, 10.0]`
  - `classifier__class_weight`: `[None, 'balanced']`
  - `classifier__solver`: `['liblinear', 'lbfgs']`
- **Optimal Hyperparameters Selected**:
  - `C`: `0.1`
  - `class_weight`: `'balanced'`
  - `solver`: `'liblinear'`
- **5-Fold Cross-Validation Accuracy**: `85.50%`

### Holdout Test Set Evaluation Metrics (138 Test Samples)
- **Test Accuracy**: `87.68%`
- **Precision**: `83.33%`
- **Recall**: `90.16%`
- **F1-Score**: `86.61%`
- **Confusion Matrix**:
  - **True Negatives (TN)**: `66` (Correctly predicted rejections)
  - **False Positives (FP)**: `11` (Incorrectly predicted as approved)
  - **False Negatives (FN)**: `6` (Incorrectly predicted as rejected)
  - **True Positives (TP)**: `55` (Correctly predicted approvals)

---

## 4. Windows PowerShell Setup & Run Instructions

Follow these exact commands in **Windows PowerShell**:

### Step 1: Open Terminal in Project Folder
Navigate to your project folder:
```powershell
cd "c:\Users\mites\OneDrive\Documents\JC CC"
```

### Step 2: Create and Activate a Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(Note: If PowerShell displays an `Execution_Policies` restriction, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and then run the activate command again).*

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Train and Evaluate the Model
Run the automated training script:
```powershell
python train_model.py
```
*What this does:*
1. Automatically checks for `data/crx.data` (downloads it from the UCI repository if missing).
2. Cleans missing values, builds the Scikit-learn Pipeline, and runs `GridSearchCV`.
3. Evaluates test accuracy, precision, recall, and F1-score.
4. Saves the trained pipeline to `models/model.joblib` and metrics to `models/metrics.json`.

### Step 5: Launch the Flask Web Application
```powershell
python app.py
```

### Step 6: Open the Dashboard in your Browser
Open your browser and navigate to:
```
http://127.0.0.1:8080/
```

---

## 5. API Endpoints

| Method | Endpoint | Description | Sample Response |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | Web application dashboard | HTML UI |
| `GET` | `/health` | Container health check | `{"status": "healthy", "model_loaded": true}` |
| `GET` | `/metrics` | Holdout evaluation metrics | `{"accuracy": 0.8768, "precision": 0.8333, ...}` |
| `POST` | `/predict` | Evaluates applicant profile | `{"prediction": "Likely Approved", "confidence": 0.8995, ...}` |

---

## 6. How a Prediction Request Flows Through the System

```
[ User Browser / Client ]
          │
          │ 1. Submits form (or clicks Demo Preset)
          ▼
[ POST /predict (Flask Backend) ]
          │
          │ 2. Generates unique 8-char Request ID
          │ 3. Validates required fields, data types, and non-negative constraints
          ▼
[ Scikit-learn Preprocessing Pipeline ]
          │
          │ 4. SimpleImputer (median / most frequent)
          │ 5. StandardScaler (numerical) + OneHotEncoder (categorical)
          ▼
[ Optimized Logistic Regression Model ]
          │
          │ 6. Executes model.predict() and model.predict_proba()
          ▼
[ Sanitized Logging & Response ]
          │
          │ 7. Logs [Timestamp] [ReqID] [Result] [Status] [Duration ms] to stdout
          │    (No personal identification or financial numbers logged)
          ▼
[ JSON Output to Browser ]
          │
          └─► UI updates with "Likely Approved" or "Likely Rejected" badge & confidence bar.
```

---

## 7. Cloud Readiness for Google Cloud Run

This application was engineered with the 12-factor cloud principles:
1. **Dynamic Port Binding**: Reads the `PORT` environment variable (`os.environ.get("PORT", 8080)`) and binds to `0.0.0.0`.
2. **Stateless Service**: No local database or session affinity required. The model is loaded into memory on container startup.
3. **Structured Cloud Logging**: Application logs print directly to standard output (`sys.stdout`) with timestamps and request IDs for Google Cloud Logging ingestion.
4. **Health Check Endpoint**: `GET /health` provides a lightweight endpoint for Cloud Run container liveness probes.
5. **Portable File Paths**: Uses `pathlib.Path(__file__).parent` relative referencing, preventing hardcoded local machine paths.

---

## 8. Troubleshooting & Common Errors

### 1. `Execution of scripts is disabled on this system` (PowerShell)
- **Cause**: Windows PowerShell security policy blocks virtual environment activation.
- **Fix**: Run this in PowerShell before activating:
  ```powershell
  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
  .\venv\Scripts\Activate.ps1
  ```

### 2. `Model has not been trained yet` or `503 Service Unavailable`
- **Cause**: The web app was launched before running the training script.
- **Fix**: Run `python train_model.py` to create `models/model.joblib`.

### 3. `Failed to download dataset automatically`
- **Cause**: Firewall or temporary internet outage during initial download.
- **Fix**: Download `crx.data` directly from:
  `https://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data`
  Place the file inside the `data/` folder as `data/crx.data`, then run `python train_model.py`.

### 4. `Port 8080 is already in use`
- **Cause**: Another service or previous Python instance is using port 8080.
- **Fix**: Specify an alternate port using the `PORT` environment variable:
  ```powershell
  $env:PORT="5000"
  python app.py
  ```
  Then access `http://127.0.0.1:5000/`.

---

## 9. Academic & Ethical Disclaimer

This application is developed strictly for **educational and college project demonstration purposes**.
- Real credit card numbers, CVVs, bank passwords, or sensitive PII are **never** collected or stored.
- Predictions represent statistical estimates of machine learning classifiers and must **not** be interpreted as genuine financial or credit decisions.
