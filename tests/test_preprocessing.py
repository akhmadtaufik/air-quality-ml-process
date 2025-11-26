import pandas as pd
from src.preprocessing.categories import join_categories


def test_data_shape_consistency(sample_dataframe):
    """Test that X and y have consistent number of samples."""
    df = sample_dataframe

    X = df[["pm10", "pm25", "so2", "co", "o3", "no2"]]
    y = df["categori"]

    assert len(X) == len(y), "X and y must have the same number of samples"
    assert X.shape[0] == y.shape[0], "Number of rows must match"


def test_no_null_values_after_preprocessing(sample_dataframe):
    """Test that there are no null values in the processed data."""
    df = sample_dataframe.dropna()

    assert (
        not df.isnull().any().any()
    ), "No null values should remain after preprocessing"


def test_join_categories(config_dict):
    """Test category merging functionality."""
    df = pd.DataFrame(
        {"categori": ["BAIK", "SEDANG", "TIDAK SEHAT", "BAIK", "SEDANG"]}
    )

    result = join_categories(df, config_dict)

    assert "SEDANG" not in result["categori"].unique()
    assert "TIDAK SEHAT" not in result["categori"].unique()
    assert "TIDAK BAIK" in result["categori"].unique()

    assert result["categori"].nunique() == 2


def test_join_categories_preserves_good_category(config_dict):
    """Test that BAIK category is preserved after merging."""
    df = pd.DataFrame({"categori": ["BAIK", "SEDANG", "TIDAK SEHAT"]})

    result = join_categories(df, config_dict)

    assert "BAIK" in result["categori"].unique()

    baik_rows = result[result["categori"] == "BAIK"]
    assert len(baik_rows) == 1


def test_data_types_correct(sample_dataframe, config_dict):
    """Test that data types are correct after preprocessing."""
    df = sample_dataframe

    for col in config_dict["int32_columns"]:
        if col in df.columns:
            assert pd.api.types.is_numeric_dtype(
                df[col]
            ), f"{col} should be numeric"

    for col in config_dict["object_columns"]:
        if col in df.columns:
            assert pd.api.types.is_object_dtype(
                df[col]
            ) or pd.api.types.is_string_dtype(
                df[col]
            ), f"{col} should be object/string type"


def test_feature_ranges(sample_dataframe, config_dict):
    """Test that features are within expected ranges."""
    df = sample_dataframe

    ranges = {
        "pm10": config_dict["range_pm10"],
        "pm25": config_dict["range_pm25"],
        "so2": config_dict["range_so2"],
        "co": config_dict["range_co"],
        "o3": config_dict["range_o3"],
        "no2": config_dict["range_no2"],
    }

    for col, (min_val, max_val) in ranges.items():
        if col in df.columns:
            assert df[col].min() >= min_val, f"{col} min value out of range"
            assert df[col].max() <= max_val, f"{col} max value out of range"


def test_dataframe_not_empty(sample_dataframe):
    """Test that DataFrame is not empty after preprocessing."""
    df = sample_dataframe

    assert not df.empty, "DataFrame should not be empty"
    assert len(df) > 0, "DataFrame should have at least one row"


def test_required_columns_exist(sample_dataframe, config_dict):
    """Test that all required predictor columns exist."""
    df = sample_dataframe

    required_cols = config_dict["predictors"]

    for col in required_cols:
        assert col in df.columns, f"Required column '{col}' is missing"
