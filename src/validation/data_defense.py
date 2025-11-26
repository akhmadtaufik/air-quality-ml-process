def check_data(input_data, params):
    """
    Validate a dataset against expected schema and value ranges.

    This function performs a series of assertions to ensure the input dataset
    matches the expected data types, column definitions, and value ranges
    defined in `params`. If a check fails, an `AssertionError` is raised
    with a descriptive message.

    Args:
        input_data (pandas.DataFrame):
            Dataset to validate. Must contain columns corresponding to
            those defined in `params`.
        params (dict):
            Dictionary containing validation rules. Expected keys include:
            - "datetime_columns": list of datetime column names.
            - "object_columns": list of categorical (object) column names.
            - "int32_columns": list of integer column names.
            - "range_stasiun": list of valid station names (for `stasiun` column).
            - "range_pm10": tuple (min, max) for pm10 values.
            - "range_pm25": tuple (min, max) for pm25 values.
            - "range_so2": tuple (min, max) for so2 values.
            - "range_co": tuple (min, max) for co values.
            - "range_o3": tuple (min, max) for o3 values.
            - "range_no2": tuple (min, max) for no2 values.

    Returns:
        None:
            The function raises an `AssertionError` if any check fails.
            If all checks pass, nothing is returned.

    Raises:
        AssertionError:
            If any column types or value ranges do not match expectations.

    Notes:
        - Ensures data types (`datetime`, `object`, `int`) match specification.
        - Ensures categorical values (`stasiun`) are within allowed set.
        - Ensures pollutant values (pm10, pm25, so2, co, o3, no2) lie within
          specified numeric ranges.
        - Designed for use in a data-cleaning pipeline to catch schema or
          range violations early.

    Example:
        >>> params = {
        ...     "datetime_columns": ["tanggal"],
        ...     "object_columns": ["stasiun"],
        ...     "int32_columns": ["pm10", "pm25", "so2", "co", "o3", "no2"],
        ...     "range_stasiun": ["Jakarta", "Bandung"],
        ...     "range_pm10": (0, 500),
        ...     "range_pm25": (0, 500),
        ...     "range_so2": (0, 100),
        ...     "range_co": (0, 50),
        ...     "range_o3": (0, 200),
        ...     "range_no2": (0, 200)
        ... }
        >>> check_data(df, params)  # Raises AssertionError if invalid
    """
    # check data types
    assert (
        input_data.select_dtypes("datetime").columns.to_list()
        == params["datetime_columns"]
    ), "an error occurs in datetime column(s)."
    assert (
        input_data.select_dtypes("object").columns.to_list()
        == params["object_columns"]
    ), "an error occurs in object column(s)."
    assert (
        input_data.select_dtypes("int").columns.to_list()
        == params["int32_columns"]
    ), "an error occurs in int32 column(s)."

    # check range of data
    assert set(input_data.stasiun).issubset(
        set(params["range_stasiun"])
    ), "an error occurs in stasiun range."
    assert input_data.pm10.between(
        params["range_pm10"][0], params["range_pm10"][1]
    ).sum() == len(input_data), "an error occurs in pm10 range."
    assert input_data.pm25.between(
        params["range_pm25"][0], params["range_pm25"][1]
    ).sum() == len(input_data), "an error occurs in pm25 range."
    assert input_data.so2.between(
        params["range_so2"][0], params["range_so2"][1]
    ).sum() == len(input_data), "an error occurs in so2 range."
    assert input_data.co.between(
        params["range_co"][0], params["range_co"][1]
    ).sum() == len(input_data), "an error occurs in co range."
    assert input_data.o3.between(
        params["range_o3"][0], params["range_o3"][1]
    ).sum() == len(input_data), "an error occurs in o3 range."
    assert input_data.no2.between(
        params["range_no2"][0], params["range_no2"][1]
    ).sum() == len(input_data), "an error occurs in no2 range."
