import json
from datetime import datetime
from typing import Dict, List, Union
import pandas as pd


def time_stamp() -> datetime:
    """
    Get the current local date and time.

    Returns:
        datetime.datetime:
            The current local datetime at the moment of function call.

    Example:
        >>> ts = time_stamp()
        >>> print(ts)
        2025-09-09 18:42:13.123456
    """
    return datetime.now()


def create_log_template() -> (
    Dict[str, List[Union[str, float, dict, datetime]]]
):
    """
    Create a template dictionary for experiment or model training logs.

    Each key maps to a list, where every list entry corresponds to one
    training run / experiment.

    Returns:
        Dict[str, List[...]]:
            A dictionary with the following keys and expected list element types:

            - "model_name": List[str]
                Model names (e.g., "RandomForestClassifier", "LogisticRegression").

            - "model_uid": List[str]
                Unique identifier per model run (e.g., UUID, hash, or timestamp string).

            - "training_time": List[float]
                Training duration per run (unit: seconds or minutes).

            - "training_date": List[datetime.datetime | str]
                Date and time when training occurred. Can be stored as a datetime
                object or ISO-format string.

            - "performance": List[dict]
                Dictionary of performance metrics per run.
                Example: {"accuracy": 0.92, "precision": 0.90, "recall": 0.88}

            - "f1_score_avg": List[float]
                Average F1-score per run (useful for classification tasks).

            - "data_configurations": List[dict]
                Dictionary describing dataset or preprocessing settings.
                Example:
                    {
                        "train_size": 0.8,
                        "validation_size": 0.2,
                        "scaler": "StandardScaler",
                        "imputer": "SimpleImputer(strategy='median')"
                    }

    Example:
        >>> logs = create_log_template()
        >>> logs["model_name"].append("RandomForestClassifier")
        >>> logs["model_uid"].append("rf_20250909_001")
        >>> logs["training_time"].append(32.57)
        >>> logs["training_date"].append("2025-09-09 19:00:23")
        >>> logs["performance"].append({"accuracy": 0.92, "f1_score": 0.89})
        >>> logs["f1_score_avg"].append(0.89)
        >>> logs["data_configurations"].append({
        ...     "train_size": 0.8,
        ...     "validation_size": 0.2,
        ...     "scaler": "StandardScaler"
        ... })
    """
    logger = {
        "model_name": [],
        "model_uid": [],
        "training_time": [],
        "training_date": [],
        "performance": [],
        "f1_score_avg": [],
        "data_configurations": [],
    }
    return logger


def training_log_updater(current_log: dict, log_path: str) -> list[dict]:
    """
    Update a persistent training log file with a new log entry.

    This function:
      1. Loads an existing training log from disk (JSON file).
      2. If the file does not exist or is empty/invalid JSON, initializes it as an empty list.
      3. Appends the new log (`current_log`) to the list.
      4. Writes the updated log list back to disk.
      5. Returns the updated log list.

    Args:
        current_log (dict):
            Dictionary representing the current run's training log.
            Example:
                {
                    "model_name": "RandomForestClassifier",
                    "training_time": 32.57,
                    "performance": {"accuracy": 0.92, "f1_score": 0.89},
                    "training_date": "2025-09-09 19:00:23",
                    ...
                }
        log_path (str):
            Path to the JSON file where logs are stored.

    Returns:
        list[dict]:
            Updated list of all training logs (including the newly appended one).

    Raises:
        json.JSONDecodeError:
            If the log file exists but is not valid JSON.

    Notes:
        - If `log_path` does not exist, the function creates a new JSON file
          containing an empty list (`[]`) before appending the new log.
        - The function overwrites the file with the updated log list.
        - Each call appends exactly one `current_log`.

    Example:
        >>> new_log = {
        ...     "model_name": "LogisticRegression",
        ...     "model_uid": "lr_20250909_002",
        ...     "training_time": 12.14,
        ...     "training_date": "2025-09-09 19:15:42",
        ...     "performance": {"accuracy": 0.88, "f1_score": 0.85},
        ...     "f1_score_avg": 0.85,
        ...     "data_configurations": {"train_size": 0.7, "validation_size": 0.3}
        ... }
        >>> updated_logs = training_log_updater(new_log, "training_log.json")
        >>> len(updated_logs)
        5   # (assuming there were 4 previous logs)
    """
    current_log = current_log.copy()

    try:
        with open(log_path, "r") as file:
            last_log = json.load(file)

    except (FileNotFoundError, json.JSONDecodeError):
        with open(log_path, "w") as file:
            file.write("[]")

        with open(log_path, "r") as file:
            last_log = json.load(file)

    last_log.append(current_log)

    with open(log_path, "w") as file:
        json.dump(last_log, file)

    return last_log


def training_log_to_df(log_path: str) -> pd.DataFrame:
    """
    Convert a training log JSON file to a pandas DataFrame.

    This function loads the training log from disk and converts it to a DataFrame
    for easier analysis and visualization. It handles empty or missing log files
    gracefully by returning an empty DataFrame.

    Args:
        log_path (str):
            Path to the JSON file where logs are stored.

    Returns:
        pd.DataFrame:
            DataFrame containing all training logs with columns:
            - model_name, model_uid, training_time, training_date,
              performance, f1_score_avg, data_configurations

    Example:
        >>> df = training_log_to_df("log/training_log.json")
        >>> print(df[['model_name', 'f1_score_avg']].head())
    """
    try:
        with open(log_path, "r") as file:
            logs = json.load(file)

        if not logs:
            return pd.DataFrame()

        return pd.DataFrame(logs)

    except (FileNotFoundError, json.JSONDecodeError):
        return pd.DataFrame()
