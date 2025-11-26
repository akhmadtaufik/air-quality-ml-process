import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
)

from src.helper.logger import training_log_to_df


def get_best_model(
    log_path: str, metric: str = "f1_score_avg", ascending: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Extract the best model from training log based on a specified metric.

    This function reads the training log, sorts by the metric, and returns
    the best model's information.

    Args:
        log_path (str):
            Path to the training log JSON file.
        metric (str, optional):
            Column name to sort by. Defaults to "f1_score_avg".
        ascending (bool, optional):
            Sort order. False means higher is better. Defaults to False.

    Returns:
        Optional[Dict[str, Any]]:
            Dictionary containing the best model's log entry, or None if log is empty.
            Contains keys: model_name, model_uid, training_time, training_date,
            performance, f1_score_avg, data_configurations.

    Example:
        >>> best = get_best_model("log/training_log.json")
        >>> if best:
        ...     print(f"Best model: {best['model_name']}")
        ...     print(f"F1-score: {best['f1_score_avg']:.3f}")
    """
    df = training_log_to_df(log_path)

    if df.empty:
        print("Warning: Training log is empty.")
        return None

    if metric not in df.columns:
        raise ValueError(
            f"Metric '{metric}' not found in log. Available: {df.columns.tolist()}"
        )

    df_sorted = df.sort_values(by=metric, ascending=ascending)
    best_row = df_sorted.iloc[0]

    return best_row.to_dict()


def generate_classification_report(
    y_true: Any,
    y_pred: Any,
    target_names: Optional[list] = None,
    output_dict: bool = False,
) -> Any:
    """
    Wrapper for sklearn's classification_report with standard formatting.

    Args:
        y_true (array-like):
            Ground truth (correct) target values.
        y_pred (array-like):
            Estimated targets as returned by a classifier.
        target_names (list, optional):
            Display names matching the labels (same order).
        output_dict (bool, optional):
            If True, return output as dict. Defaults to False (string).

    Returns:
        str | dict:
            Classification report as string or dictionary.

    Example:
        >>> report = generate_classification_report(
        ...     y_test, y_pred, target_names=['BAIK', 'TIDAK BAIK']
        ... )
        >>> print(report)
    """
    return classification_report(
        y_true, y_pred, target_names=target_names, output_dict=output_dict
    )


def plot_confusion_matrix(
    y_true: Any,
    y_pred: Any,
    display_labels: Optional[list] = None,
    cmap: str = "Blues",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[str] = None,
) -> plt.Figure:  # type: ignore
    """
    Plot confusion matrix with customizable display.

    Args:
        y_true (array-like):
            Ground truth (correct) target values.
        y_pred (array-like):
            Estimated targets as returned by a classifier.
        display_labels (list, optional):
            Target names for display.
        cmap (str, optional):
            Colormap. Defaults to 'Blues'.
        figsize (Tuple[int, int], optional):
            Figure size. Defaults to (8, 6).
        save_path (str, optional):
            If provided, save figure to this path.

    Returns:
        plt.Figure:
            Matplotlib figure object.

    Example:
        >>> fig = plot_confusion_matrix(
        ...     y_test, y_pred,
        ...     display_labels=['BAIK', 'TIDAK BAIK'],
        ...     save_path='plots/confusion_matrix.png'
        ... )
        >>> plt.show()
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=display_labels
    )

    disp.plot(cmap=cmap, ax=ax)
    plt.title("Confusion Matrix")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def compare_models(
    log_path: str, metric: str = "f1_score_avg"
) -> pd.DataFrame:
    """
    Create a comparison table of all models from training log.

    Args:
        log_path (str):
            Path to the training log JSON file.
        metric (str, optional):
            Primary metric for sorting. Defaults to "f1_score_avg".

    Returns:
        pd.DataFrame:
            DataFrame with models sorted by metric, showing:
            model_name, f1_score_avg, training_time, data_configurations.

    Example:
        >>> comparison = compare_models("log/training_log.json")
        >>> print(comparison[['model_name', 'f1_score_avg', 'training_time']])
    """
    df = training_log_to_df(log_path)

    if df.empty:
        print("Warning: Training log is empty.")
        return pd.DataFrame()

    cols_to_show = [
        "model_name",
        "f1_score_avg",
        "training_time",
        "data_configurations",
        "model_uid",
    ]
    available_cols = [col for col in cols_to_show if col in df.columns]

    df_comparison = df[available_cols].sort_values(by=metric, ascending=False)

    return df_comparison.reset_index(drop=True)


def extract_performance_metrics(
    performance_dict: Dict[str, Any],
) -> Dict[str, float]:
    """
    Extract key metrics from sklearn classification_report dictionary.

    Args:
        performance_dict (Dict[str, Any]):
            Output from classification_report with output_dict=True.

    Returns:
        Dict[str, float]:
            Simplified dict with accuracy, macro avg precision, recall, f1-score.

    Example:
        >>> report = classification_report(y_true, y_pred, output_dict=True)
        >>> metrics = extract_performance_metrics(report)
        >>> print(metrics)
        {'accuracy': 0.87, 'precision': 0.86, 'recall': 0.85, 'f1_score': 0.85}
    """
    return {
        "accuracy": performance_dict.get("accuracy", 0.0),
        "precision": performance_dict.get("macro avg", {}).get(
            "precision", 0.0
        ),
        "recall": performance_dict.get("macro avg", {}).get("recall", 0.0),
        "f1_score": performance_dict.get("macro avg", {}).get("f1-score", 0.0),
    }
