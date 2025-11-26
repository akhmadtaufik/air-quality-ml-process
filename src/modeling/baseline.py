from typing import Dict
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier

    class LabelEncodedXGBClassifier(XGBClassifier):
        """XGBClassifier variant that encodes string labels internally."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._label_encoder = LabelEncoder()
            self._classes_fitted = False

        def fit(self, X, y, *args, **kwargs):  # type: ignore[override]
            y_encoded = self._label_encoder.fit_transform(list(y))
            result = super().fit(X, y_encoded, *args, **kwargs)
            self.classes_ = self._label_encoder.classes_
            self._classes_fitted = True
            return result

        def predict(self, X, *args, **kwargs):  # type: ignore[override]
            encoded_preds = super().predict(X, *args, **kwargs)
            if not self._classes_fitted:
                return encoded_preds
            encoded_preds = [int(p) for p in encoded_preds]
            return self._label_encoder.inverse_transform(encoded_preds)

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def load_models() -> Dict[str, BaseEstimator]:
    """
    Return a dictionary of baseline model instances.

    This function initializes a set of common classification models
    with their default parameters. The models are ready to be trained.

    Returns:
        Dict[str, BaseEstimator]:
            Dictionary mapping model names to instantiated model objects.
            Keys are model class names (str).
            Values are sklearn-compatible estimator instances.

    Example:
        >>> models = load_models()
        >>> print(models.keys())
        dict_keys(['LogisticRegression', 'DecisionTreeClassifier',
                   'RandomForestClassifier', 'KNeighborsClassifier'])
        >>> lr_model = models['LogisticRegression']
        >>> lr_model.fit(X_train, y_train)
    """
    models_dict = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTreeClassifier": DecisionTreeClassifier(random_state=42),
        "RandomForestClassifier": RandomForestClassifier(
            random_state=42, n_estimators=100
        ),
        "KNeighborsClassifier": KNeighborsClassifier(),
    }

    if XGBOOST_AVAILABLE:
        models_dict["XGBClassifier"] = LabelEncodedXGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
        )

    return models_dict


def get_model_list_for_experiment(
    experiment_name: str,
) -> list[Dict[str, any]]:
    """
    Create a list of model dictionaries for a specific experiment.

    Each dictionary contains model metadata including model_name, model_object,
    and an empty model_uid field to be filled during training.

    Args:
        experiment_name (str):
            Name/identifier for the experiment (e.g., "undersampling", "oversampling").

    Returns:
        list[Dict[str, any]]:
            List of dictionaries, each containing:
            - "model_name" (str): Class name of the model
            - "model_object" (BaseEstimator): Instantiated model
            - "model_uid" (str): Empty string, to be filled during training

    Example:
        >>> models = get_model_list_for_experiment("baseline")
        >>> print(len(models))
        5
        >>> print(models[0].keys())
        dict_keys(['model_name', 'model_object', 'model_uid'])
    """
    base_models = load_models()

    return [
        {"model_name": model_name, "model_object": model_obj, "model_uid": ""}
        for model_name, model_obj in base_models.items()
    ]
