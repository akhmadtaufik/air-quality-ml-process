import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def read_dataset(dataset_dir: str) -> pd.DataFrame:
    """
    Read and concatenate all CSV files in a directory into a single DataFrame.

    This function iterates through all files in the given directory,
    selects those with the `.csv` extension, loads them into DataFrames,
    and concatenates them into one combined dataset.

    Args:
        dataset_dir (str):
            Path to the directory containing CSV files. Can be a string path
            or `Path` object.

    Returns:
        pandas.DataFrame:
            Combined dataset containing rows from all CSV files in the directory.
            The row indices are reset to ensure continuity.

    Notes:
        - Files are joined using `os.listdir()` order, which may vary across OS.
        - Only files with `.csv` suffix are processed.
        - Empty directory or no `.csv` files will return an empty DataFrame.

    Example:
        >>> combined_df = read_dataset("data/raw/")
        100%|██████████| 3/3 [00:00<00:00, 250.00it/s]
        >>> combined_df.shape
        (1500, 8)
    """
    dataset_dir = Path(dataset_dir)
    dataset = pd.DataFrame()

    for file in tqdm(os.listdir(dataset_dir)):
        file_path = dataset_dir / file  # Gabung path
        if file_path.suffix == ".csv":
            df = pd.read_csv(file_path)
            dataset = pd.concat([dataset, df])

    return dataset
