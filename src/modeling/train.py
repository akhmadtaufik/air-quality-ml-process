import uuid
import copy
from typing import Any, Dict, List, Tuple
from sklearn.base import BaseEstimator
from sklearn.metrics import classification_report
from tqdm import tqdm

from src.helper.logger import (
    time_stamp,
    create_log_template,
    training_log_updater,
)


def train_eval_model(
    list_of_model: List[Dict[str, Any]],
    prefix_model_name: str,
    x_train: Any,
    y_train: Any,
    data_configuration_name: str,
    x_valid: Any,
    y_valid: Any,
    log_path: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Train and evaluate a list of models, record logs, and update persistent training history.

    Args:
        list_of_model (List[Dict[str, Any]]):
            List of models with metadata. Each dict must contain:
              - "model_name" (str): model label.
              - "model_object" (estimator): sklearn-compatible model with `.fit()` and `.predict()`.
        prefix_model_name (str):
            Prefix string added to each model name (e.g., experiment ID).
        x_train (array-like | pandas.DataFrame):
            Training features.
        y_train (array-like | pandas.Series):
            Training target labels.
        data_configuration_name (str):
            Identifier or config name describing data setup.
        x_valid (array-like | pandas.DataFrame):
            Validation features.
        y_valid (array-like | pandas.Series):
            Validation target labels.
        log_path (str):
            Path to JSON file where logs are persisted.

    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
            - training_log: updated list of all training logs (loaded from `log_path`).
            - list_of_model: updated list of models, each enriched with a "model_uid" key.

    Notes:
        - Each run gets a unique model ID (`model_uid`) generated via `uuid4().hex`.
        - Training time is measured in seconds.
        - Macro-average F1-score is extracted separately and logged.
        - Evaluation uses `classification_report` (scikit-learn) with `output_dict=True`.

    Example:
        >>> models = [
        ...     {"model_name": "LogReg", "model_object": LogisticRegression()},
        ...     {"model_name": "DecisionTree", "model_object": DecisionTreeClassifier()}
        ... ]
        >>> training_log, trained_models = train_eval_model(
        ...     list_of_model=models,
        ...     prefix_model_name="exp1",
        ...     x_train=X_train, y_train=y_train,
        ...     data_configuration_name="dataset_v1",
        ...     x_valid=X_val, y_valid=y_val,
        ...     log_path="training_log.json"
        ... )
        >>> training_log[-1]["model_name"]
        'exp1-DecisionTree'
        >>> trained_models[0]["model_uid"]  # unique ID
        '3c0a64d68a0a4565a8d7d2c8f354b2df'
    """
    list_of_model = copy.deepcopy(list_of_model)

    logger = create_log_template()

    for model in tqdm(list_of_model, desc="Training models"):
        model_name = prefix_model_name + "-" + model["model_name"]

        x_train_input = x_train
        y_train_input = y_train
        x_valid_input = x_valid

        if model["model_name"] == "XGBClassifier":
            if hasattr(x_train, "to_numpy"):
                x_train_input = x_train.to_numpy()
            if hasattr(y_train, "to_numpy"):
                y_train_input = y_train.to_numpy()
            if hasattr(x_valid, "to_numpy"):
                x_valid_input = x_valid.to_numpy()

        start_time = time_stamp()
        model["model_object"].fit(x_train_input, y_train_input)
        finished_time = time_stamp()

        elapsed_time = (finished_time - start_time).total_seconds()

        y_pred = model["model_object"].predict(x_valid_input)
        performance = classification_report(y_valid, y_pred, output_dict=True)

        chiper_id = uuid.uuid4().hex
        model["model_uid"] = chiper_id

        logger["model_name"].append(model_name)
        logger["model_uid"].append(chiper_id)
        logger["training_time"].append(elapsed_time)
        logger["training_date"].append(str(start_time))
        logger["performance"].append(performance)
        logger["f1_score_avg"].append(performance["macro avg"]["f1-score"])
        logger["data_configurations"].append(data_configuration_name)

    training_log = training_log_updater(logger, log_path)

    return training_log, list_of_model


def train_single_model(
    model: BaseEstimator,
    x_train: Any,
    y_train: Any,
    x_valid: Any,
    y_valid: Any,
) -> Tuple[BaseEstimator, Dict[str, Any], float]:
    """
    Train and evaluate a single model without logging.

    This is a simplified version for quick training and evaluation
    without persistent logging to JSON files.

    Args:
        model (BaseEstimator):
            Sklearn-compatible model instance.
        x_train (array-like):
            Training features.
        y_train (array-like):
            Training labels.
        x_valid (array-like):
            Validation features.
        y_valid (array-like):
            Validation labels.

    Returns:
        Tuple[BaseEstimator, Dict[str, Any], float]:
            - trained_model: The fitted model
            - performance: Classification report dict
            - training_time: Training duration in seconds

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier()
        >>> trained, metrics, time = train_single_model(
        ...     model, X_train, y_train, X_val, y_val
        ... )
        >>> print(f"Training took {time:.2f} seconds")
        >>> print(f"F1-score: {metrics['macro avg']['f1-score']:.3f}")
    """
    start_time = time_stamp()
    model.fit(x_train, y_train)
    finished_time = time_stamp()

    elapsed_time = (finished_time - start_time).total_seconds()

    y_pred = model.predict(x_valid)
    performance = classification_report(y_valid, y_pred, output_dict=True)

    return model, performance, elapsed_time
