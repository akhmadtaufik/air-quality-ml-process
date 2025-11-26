from typing import Any, Dict
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def tune_model(
    model: BaseEstimator,
    param_grid: Dict[str, Any],
    X: Any,
    y: Any,
    cv: int = 5,
    scoring: str = "f1_macro",
    n_jobs: int = -1,
) -> GridSearchCV:
    """
    Wrapper for GridSearchCV with standard configuration.

    This function performs exhaustive hyperparameter tuning using grid search
    with cross-validation.

    Args:
        model (BaseEstimator):
            Sklearn-compatible model instance to tune.
        param_grid (Dict[str, Any]):
            Dictionary with parameter names (str) as keys and lists of
            parameter settings to try as values.
            Example: {'max_depth': [3, 5, 7], 'min_samples_split': [2, 5, 10]}
        X (array-like):
            Training features.
        y (array-like):
            Training labels.
        cv (int, optional):
            Number of cross-validation folds. Defaults to 5.
        scoring (str, optional):
            Scoring metric for evaluation. Defaults to 'f1_macro'.
        n_jobs (int, optional):
            Number of parallel jobs. -1 uses all processors. Defaults to -1.

    Returns:
        GridSearchCV:
            Fitted GridSearchCV object. Access best parameters via
            `.best_params_` and best estimator via `.best_estimator_`.

    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier()
        >>> param_grid = {
        ...     'n_estimators': [50, 100, 200],
        ...     'max_depth': [5, 10, None],
        ...     'min_samples_split': [2, 5]
        ... }
        >>> grid_search = tune_model(model, param_grid, X_train, y_train)
        >>> print(f"Best params: {grid_search.best_params_}")
        >>> best_model = grid_search.best_estimator_
    """
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1,
    )

    grid_search.fit(X, y)

    return grid_search


def tune_model_randomized(
    model: BaseEstimator,
    param_distributions: Dict[str, Any],
    X: Any,
    y: Any,
    n_iter: int = 10,
    cv: int = 5,
    scoring: str = "f1_macro",
    n_jobs: int = -1,
    random_state: int = 42,
) -> RandomizedSearchCV:
    """
    Wrapper for RandomizedSearchCV for faster hyperparameter tuning.

    This function samples a fixed number of parameter settings from
    specified distributions, which is faster than GridSearchCV for
    large parameter spaces.

    Args:
        model (BaseEstimator):
            Sklearn-compatible model instance to tune.
        param_distributions (Dict[str, Any]):
            Dictionary with parameter names and distributions or lists.
            Example: {'max_depth': [3, 5, 7, 9], 'learning_rate': uniform(0.01, 0.3)}
        X (array-like):
            Training features.
        y (array-like):
            Training labels.
        n_iter (int, optional):
            Number of parameter settings sampled. Defaults to 10.
        cv (int, optional):
            Number of cross-validation folds. Defaults to 5.
        scoring (str, optional):
            Scoring metric for evaluation. Defaults to 'f1_macro'.
        n_jobs (int, optional):
            Number of parallel jobs. -1 uses all processors. Defaults to -1.
        random_state (int, optional):
            Random seed for reproducibility. Defaults to 42.

    Returns:
        RandomizedSearchCV:
            Fitted RandomizedSearchCV object.

    Example:
        >>> from sklearn.ensemble import GradientBoostingClassifier
        >>> from scipy.stats import uniform, randint
        >>> model = GradientBoostingClassifier()
        >>> param_dist = {
        ...     'n_estimators': randint(50, 200),
        ...     'max_depth': randint(3, 10),
        ...     'learning_rate': uniform(0.01, 0.2)
        ... }
        >>> random_search = tune_model_randomized(
        ...     model, param_dist, X_train, y_train, n_iter=20
        ... )
        >>> print(f"Best params: {random_search.best_params_}")
    """
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1,
        random_state=random_state,
    )

    random_search.fit(X, y)

    return random_search
