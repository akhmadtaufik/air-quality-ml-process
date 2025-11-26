# Air Quality Intelligence Platform

![Python](https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=ffdd54)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit_learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Build](https://img.shields.io/badge/Build-Passing-2EA44F?style=flat-square&logo=github-actions&logoColor=white)
![Code Style](https://img.shields.io/badge/Code_Style-Black-000000?style=flat-square&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square&logo=open-source-initiative&logoColor=white)

## About the Project

This repository contains an end-to-end machine learning workflow that predicts Jakarta’s air quality categories from chemical pollutant readings (PM10, PM2.5, SO₂, CO, O₃, NO₂). The solution automates preprocessing, feature engineering, model training (XGBoost + classical baselines), deployment via FastAPI, and regression testing so it can be promoted to production with confidence.

## Project Structure

```
.
├── config/                 # YAML configuration (paths, predictors, label schema)
├── data/
│   ├── raw/               # ISPU CSV dumps
│   └── processed/         # Train/valid/test splits + feature artifacts
├── models/
│   ├── experiment/        # Latest experiment outputs (best_model.pkl)
│   └── production/        # Deployed model + feature artifacts
├── src/
│   ├── pipeline/          # Preprocessing, training, deployment orchestrators
│   ├── features/          # Feature engineering utilities
│   ├── modeling/          # Baseline estimators, training helpers
│   ├── serving/           # FastAPI app + Streamlit UI
│   ├── preprocessing/     # Domain-specific cleaning logic
│   ├── evaluation/        # Metrics helpers
│   └── utils/             # Test runner, config helpers
├── tests/                 # Pytest suites for API/modeling/preprocessing
├── notebooks/             # Exploratory notebooks
├── main.py                # CLI entry point for pipelines
├── Makefile               # Developer shortcuts (lint/test/pipeline/serve)
└── requirements.txt       # Python dependencies
```

## Tech Stack

- **Language / Runtime:** Python 3.10, Conda, venv
- **Data & Modeling:** Pandas, NumPy, Scikit-Learn, XGBoost
- **Serving:** FastAPI, Uvicorn, Pydantic, Streamlit
- **Tooling:** Makefile, Docker-ready workflows, Flake8
- **Testing:** Pytest (+ pytest-cov optional)

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Option A — Conda (replace `ml-air-env` with your environment name)
conda create -n ml-air-env python=3.10 -y
conda activate ml-air-env

# Option B — Python venv (replace `.venv-ml-air` with your preferred folder)
python3 -m venv .venv-ml-air
source .venv-ml-air/bin/activate

# Install dependencies (works for both options)
make install
```

### 2. Check Status

```bash
python main.py status
# or
make status
```

### 3. Run Complete Pipeline

```bash
# Option A: Run everything (preprocess -> train -> deploy)
python main.py all

# Option B: Step by step
python main.py preprocess  # Preprocess raw data
python main.py train       # Train models
python main.py deploy      # Deploy to production
```

### 4. Start Services

```bash
# Start API
python main.py serve
# or
make serve-api

# Start UI (new terminal session)
make serve-ui
```

### 5. Access

- **API**: <http://localhost:8000>
- **API Docs**: <http://localhost:8000/docs>
- **UI**: <http://localhost:8501>

## 📋 Command Reference

### Main Commands (Recommended)

```bash
python main.py status       # Check project status
python main.py all          # Run complete pipeline
python main.py preprocess   # Preprocess data only
python main.py train        # Train models only
python main.py deploy       # Deploy to production
python main.py serve        # Start API server
```

### Makefile Shortcuts

```bash
# Pipeline
make pipeline-all    # Run complete pipeline
make preprocess      # Preprocess data
make train           # Train models
make deploy-model    # Deploy to production
make status          # Check status

# Servers
make serve-api       # Start API
make serve-ui        # Start UI

# Development
make test            # Run tests
make lint            # Check code quality
make clean           # Clean temp files
```

## 🔄 ML Pipeline Lifecycle

### Full Lifecycle

```bash
# Step 1: Preprocess raw data
python main.py preprocess
# Output: data/processed/*.pkl

# Step 2: Train models
python main.py train
# Output: models/experiment/best_model.pkl, log/training_log.json

# Step 3: Deploy to production
python main.py deploy
# Output: models/production/model.pkl

# Step 4: Start serving
python main.py serve
# Running at: http://localhost:8000
```

### Quick Iteration

If the processed datasets already exist, retrain and deploy with:

```bash
# Train with new hyperparameters
python main.py train --experiment-name experiment_v2

# Deploy new model
python main.py deploy

# Restart API (auto-reload picks up the new model)
```

## 📊 Preprocessing Pipeline

File: `src/pipeline/preprocessing_pipeline.py`

**Input:** CSV files under `data/raw/`

**Steps:**

1. Load every monthly ISPU CSV file.
2. Merge intermediate label categories (`SEDANG` + `TIDAK SEHAT` → `TIDAK BAIK`).
3. Handle missing values and enforce numeric types.
4. Split the dataset into train/validation/test partitions.
5. Apply imbalance treatments (RandomUnderSampler, etc.) for balanced training sets.

**Outputs:**

- `X_train.pkl`, `y_train.pkl`
- `X_valid.pkl`, `y_valid.pkl`
- `X_test.pkl`, `y_test.pkl`
- `X_rus.pkl`, `y_rus.pkl` (undersampled)

## 🤖 Training Pipeline

File: `src/pipeline/training_pipeline.py`

**Input:** Processed data from `data/processed/`

**Models Trained:**

1. LogisticRegression
2. DecisionTreeClassifier
3. RandomForestClassifier
4. KNeighborsClassifier

**Outputs:**

- Best model saved in `models/experiment/best_model.pkl`
- Training metadata appended to `log/training_log.json`

## 🚀 Deployment Pipeline

File: `src/pipeline/deployment_pipeline.py`

**Input:** Model artifacts from `models/experiment/`

**Steps:**

1. Copy the best performing model into `models/production/`.
2. Copy encoder artifacts (e.g., `ohe_stasiun.pkl`, `le_categori.pkl`).

**Output:** Production-ready model bundle.

## 🌐 API Reference

### Endpoints

**GET /**

- Root endpoint exposing API metadata.

**GET /health**

- Health check reporting `model_loaded`.
- Response: `{"status": "healthy", "model_loaded": true}`

**POST /predict**

- Prediction endpoint.
- Request body:

```json
{
  "stasiun": "DKI1 (Bunderan HI)",
  "pm10": 50.5,
  "pm25": 30.2,
  "so2": 15.0,
  "co": 5.5,
  "o3": 45.0,
  "no2": 25.0
}
```

- Response:

```json
{
  "prediction": "BAIK",
  "confidence": 0.95,
  "model_info": "DecisionTreeClassifier"
}
```

**GET /model-info**

- Returns metadata about the currently loaded model.

## 🧪 Testing

```bash
# Run all tests
make test

# Quick test (no coverage)
make test-quick

# Test specific module
pytest tests/test_preprocessing.py -v
```

## 🔧 Development

### Add New Model

1. Edit `src/modeling/baseline.py`:

```python
def load_models():
    return {
        # ... existing models ...
        'GradientBoosting': GradientBoostingClassifier()
    }
```

2. Retrain:

```bash
python main.py train
python main.py deploy
```

### Modify Preprocessing

1. Edit `src/pipeline/preprocessing_pipeline.py`
2. Rerun the pipeline:

```bash
python main.py all
```

## 📝 Configuration

Update `config/config.yaml` to manage:

- Dataset paths
- Feature definitions
- Valid ranges per pollutant
- Label categories

## 🐛 Troubleshooting

### Data Not Found

```bash
# Check status
python main.py status

# Preprocess if needed
python main.py preprocess
```

### Model Not Loading

```bash
# Retrain and deploy
python main.py train
python main.py deploy

# Restart API
python main.py serve
```

### Port Already in Use

```bash
# Kill existing process
lsof -i :8000
kill -9 <PID>

# Or use different port
uvicorn src.serving.api:app --port 8080
```
