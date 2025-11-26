import pandas as pd
import numpy as np
from typing import Any


def nan_detector(
    set_data: pd.DataFrame, placeholder: Any = -1
) -> pd.DataFrame:
    """
    Replaces a specified placeholder value throughout a DataFrame with NumPy's NaN.

    This data cleaning utility scans the entire DataFrame for a specific
    value that represents missing or invalid data (e.g., -1, 999, "N/A")
    and converts it to `np.nan`. This standardization is crucial for many
    data analysis and machine learning libraries like pandas and scikit-learn,
    which are optimized to handle `np.nan` correctly.

    Args:
        set_data (pd.DataFrame):
            The input DataFrame to be processed.
        placeholder (Any, optional):
            The value to be replaced with `np.nan`. Defaults to -1.

    Returns:
        pd.DataFrame:
            A copy of the input DataFrame with the placeholder values
            replaced by `np.nan`.

    Notes:
        - The function operates on and returns a copy of the original
          DataFrame, ensuring the original data remains unchanged.
        - Using `np.nan` is standard practice for representing missing data
          as it is automatically ignored in many numerical computations.

    Example:
        >>> import pandas as pd
        >>> import numpy as np
        >>> data = {'age': [25, 30, -1, 45], 'score': [88, -1, 95, 76]}
        >>> df = pd.DataFrame(data)
        >>> print("Original DataFrame:")
        >>> print(df)
        Original DataFrame:
           age  score
        0   25     88
        1   30     -1
        2   -1     95
        3   45     76

        >>> cleaned_df = replace_with_nan(df)
        >>> print("\\nCleaned DataFrame:")
        >>> print(cleaned_df)
        Cleaned DataFrame:
            age  score
        0  25.0   88.0
        1  30.0    NaN
        2   NaN   95.0
        3  45.0   76.0
    """
    # Create a copy to ensure the original DataFrame is not modified
    set_data = set_data.copy()

    # Replace the placeholder value with np.nan
    set_data.replace(placeholder, np.nan, inplace=True)

    return set_data
