import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier


@pytest.fixture
def sample_data():
    """
    Fixture providing sample training and test data.

    Returns:
        dict: Dictionary with X_train, X_test, y_train, y_test
    """
    np.random.seed(42)

    n_samples = 100
    n_features = 7

    X = np.random.randn(n_samples, n_features)
    y = np.random.choice(["BAIK", "TIDAK BAIK"], size=n_samples)

    split_idx = 80

    return {
        "X_train": X[:split_idx],
        "X_test": X[split_idx:],
        "y_train": y[:split_idx],
        "y_test": y[split_idx:],
    }


@pytest.fixture
def sample_dataframe():
    """
    Fixture providing a sample pandas DataFrame with air quality data.

    Returns:
        pd.DataFrame: DataFrame with sample air quality measurements
    """
    np.random.seed(42)

    data = {
        "stasiun": np.random.choice(
            ["DKI1 (Bunderan HI)", "DKI2 (Kelapa Gading)"], size=50
        ),
        "pm10": np.random.uniform(10, 100, 50),
        "pm25": np.random.uniform(5, 50, 50),
        "so2": np.random.uniform(1, 30, 50),
        "co": np.random.uniform(1, 10, 50),
        "o3": np.random.uniform(10, 60, 50),
        "no2": np.random.uniform(5, 40, 50),
        "categori": np.random.choice(["BAIK", "TIDAK BAIK"], size=50),
    }

    return pd.DataFrame(data)


@pytest.fixture
def sample_model():
    """
    Fixture providing a simple trained model.

    Returns:
        RandomForestClassifier: A fitted model
    """
    np.random.seed(42)

    X = np.random.randn(100, 7)
    y = np.random.choice(["BAIK", "TIDAK BAIK"], size=100)

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    return model


@pytest.fixture
def config_dict():
    """
    Fixture providing a sample configuration dictionary.

    Returns:
        dict: Configuration parameters
    """
    return {
        "base_dir": Path("."),
        "dataset_dir": Path("./data/raw"),
        "dataset_processed_dir": Path("./data/processed"),
        "label": "categori",
        "label_categories": ["BAIK", "SEDANG", "TIDAK SEHAT"],
        "label_categories_new": ["BAIK", "TIDAK BAIK"],
        "predictors": ["stasiun", "pm10", "pm25", "so2", "co", "o3", "no2"],
        "int32_columns": ["pm10", "pm25", "so2", "co", "o3", "no2"],
        "object_columns": ["stasiun", "critical", "categori"],
        "range_pm10": [-1, 800],
        "range_pm25": [-1, 400],
        "range_so2": [-1, 500],
        "range_co": [-1, 100],
        "range_o3": [-1, 160],
        "range_no2": [-1, 100],
    }


@pytest.fixture
def temp_log_file(tmp_path):
    """
    Fixture providing a temporary log file path.

    Args:
        tmp_path: pytest's temporary directory fixture

    Returns:
        Path: Path to temporary log file
    """
    log_file = tmp_path / "test_training_log.json"
    return log_file


@pytest.fixture
def sample_training_log():
    """
    Fixture providing sample training log data.

    Returns:
        dict: Training log template with sample data
    """
    return {
        "model_name": ["baseline-LogisticRegression", "baseline-RandomForest"],
        "model_uid": ["abc123", "def456"],
        "training_time": [1.5, 3.2],
        "training_date": ["2025-11-24 10:00:00", "2025-11-24 10:05:00"],
        "performance": [
            {
                "accuracy": 0.85,
                "macro avg": {
                    "precision": 0.84,
                    "recall": 0.83,
                    "f1-score": 0.83,
                },
            },
            {
                "accuracy": 0.88,
                "macro avg": {
                    "precision": 0.87,
                    "recall": 0.86,
                    "f1-score": 0.87,
                },
            },
        ],
        "f1_score_avg": [0.83, 0.87],
        "data_configurations": ["undersampling", "undersampling"],
    }


@pytest.fixture
def api_test_payload():
    """
    Fixture providing a valid API request payload.

    Returns:
        dict: Valid prediction request data
    """
    return {
        "stasiun": "DKI1 (Bunderan HI)",
        "pm10": 50.5,
        "pm25": 30.2,
        "so2": 15.0,
        "co": 5.5,
        "o3": 45.0,
        "no2": 25.0,
    }
