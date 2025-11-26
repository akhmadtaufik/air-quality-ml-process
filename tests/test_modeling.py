import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.modeling.baseline import load_models, get_model_list_for_experiment
from src.modeling.train import train_single_model
from src.modeling.hyperparameter import tune_model


def test_load_models():
    """Test that load_models returns correct model dictionary."""
    models = load_models()

    assert isinstance(models, dict), "load_models should return a dictionary"
    assert len(models) > 0, "Should return at least one model"

    expected_models = [
        "LogisticRegression",
        "DecisionTreeClassifier",
        "RandomForestClassifier",
        "KNeighborsClassifier",
        "XGBClassifier",
    ]

    for model_name in expected_models:
        assert model_name in models, f"{model_name} should be in models dict"


def test_get_model_list_for_experiment():
    """Test that model list creation works correctly."""
    models = get_model_list_for_experiment("test_experiment")

    assert isinstance(models, list), "Should return a list"
    assert len(models) == 5, "Should return 5 models"

    for model_dict in models:
        assert "model_name" in model_dict
        assert "model_object" in model_dict
        assert "model_uid" in model_dict
        assert model_dict["model_uid"] == ""


def test_model_training_shape_output(sample_data):
    """Test that model prediction output has correct shape."""
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(sample_data["X_train"], sample_data["y_train"])

    predictions = model.predict(sample_data["X_test"])

    assert (
        predictions.shape[0] == sample_data["X_test"].shape[0]
    ), "Prediction output should have same number of samples as test data"
    assert len(predictions.shape) == 1, "Predictions should be 1D array"


def test_model_probability_range(sample_data):
    """Test that prediction probabilities are within [0, 1] range."""
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(sample_data["X_train"], sample_data["y_train"])

    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(sample_data["X_test"])

        assert np.all(probas >= 0.0), "All probabilities should be >= 0"
        assert np.all(probas <= 1.0), "All probabilities should be <= 1"
        assert np.allclose(
            probas.sum(axis=1), 1.0
        ), "Probabilities should sum to 1"


def test_train_single_model(sample_data):
    """Test single model training function."""
    model = LogisticRegression(random_state=42, max_iter=1000)

    trained_model, performance, training_time = train_single_model(
        model=model,
        x_train=sample_data["X_train"],
        y_train=sample_data["y_train"],
        x_valid=sample_data["X_test"],
        y_valid=sample_data["y_test"],
    )

    assert trained_model is not None, "Trained model should not be None"
    assert isinstance(performance, dict), "Performance should be a dictionary"
    assert isinstance(training_time, float), "Training time should be a float"
    assert training_time >= 0, "Training time should be non-negative"

    assert "accuracy" in performance
    assert "macro avg" in performance
    assert "f1-score" in performance["macro avg"]


def test_model_predictions_are_valid_classes(sample_data):
    """Test that model predictions are valid class labels."""
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(sample_data["X_train"], sample_data["y_train"])

    predictions = model.predict(sample_data["X_test"])

    valid_classes = set(sample_data["y_train"])
    prediction_classes = set(predictions)

    assert prediction_classes.issubset(
        valid_classes
    ), "Predictions should only contain classes seen during training"


def test_tune_model_grid_search(sample_data):
    """Test hyperparameter tuning with GridSearchCV."""
    model = RandomForestClassifier(random_state=42)
    param_grid = {"n_estimators": [10, 20], "max_depth": [3, 5]}

    grid_search = tune_model(
        model=model,
        param_grid=param_grid,
        X=sample_data["X_train"],
        y=sample_data["y_train"],
        cv=3,
    )

    assert hasattr(
        grid_search, "best_params_"
    ), "Should have best_params_ attribute"
    assert hasattr(
        grid_search, "best_estimator_"
    ), "Should have best_estimator_ attribute"
    assert (
        grid_search.best_estimator_ is not None
    ), "Best estimator should not be None"

    predictions = grid_search.predict(sample_data["X_test"])
    assert predictions.shape[0] == sample_data["X_test"].shape[0]


def test_model_fit_does_not_fail(sample_data):
    """Test that model fitting does not raise exceptions."""
    models = load_models()

    for model_name, model in models.items():
        try:
            model.fit(sample_data["X_train"], sample_data["y_train"])
            predictions = model.predict(sample_data["X_test"])
            assert len(predictions) > 0
        except Exception as e:
            pytest.fail(f"Model {model_name} failed to train: {e}")
