"""
Preprocessing Pipeline - Data Loading & Preprocessing
"""

import sys
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

# Add numpy compatibility fix
import numpy as np

if not hasattr(np, "_core"):
    sys.modules["numpy._core"] = np.core  # type: ignore
    sys.modules["numpy._core._multiarray_umath"] = np.core._multiarray_umath  # type: ignore


def load_raw_data(data_dir: Path) -> pd.DataFrame:
    """
    Load and concatenate all raw CSV files from data/raw directory.

    Args:
        data_dir: Path to raw data directory

    Returns:
        Concatenated DataFrame from all CSV files
    """
    print(f"\n📊 Loading raw data from {data_dir}...")

    csv_files = list(data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    print(f"   Found {len(csv_files)} CSV files")

    dfs = []
    for csv_file in sorted(csv_files):
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
            print(f"   ✓ Loaded: {csv_file.name} ({len(df)} rows)")
        except Exception as e:
            print(f"   ⚠️  Error loading {csv_file.name}: {e}")

    if not dfs:
        raise ValueError("No data loaded successfully")

    combined_df = pd.concat(dfs, ignore_index=True)
    print(
        f"\n   ✅ Total data loaded: {len(combined_df)} rows, {len(combined_df.columns)} columns"
    )

    return combined_df


def preprocess_data(
    df: pd.DataFrame, config: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Basic preprocessing: cleaning, type conversion, category merging.

    Args:
        df: Raw DataFrame
        config: Configuration dictionary

    Returns:
        Tuple of (features_df, labels_series)
    """
    print("\n🔧 Preprocessing data...")

    # Import preprocessing modules
    from src.preprocessing.categories import join_categories

    # 1. Handle datetime columns
    if "datetime_columns" in config:
        for col in config["datetime_columns"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                print(f"   ✓ Converted {col} to datetime")

    # 2. Handle numeric columns
    if "int32_columns" in config:
        for col in config["int32_columns"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                print(f"   ✓ Converted {col} to numeric")

    # 3. Merge categories (SEDANG + TIDAK SEHAT → TIDAK BAIK)
    label_col = config.get("label", "categori")
    if label_col in df.columns:
        df = join_categories(df, config)
        print(f"   ✓ Merged categories in {label_col}")
        print(f"      Categories: {df[label_col].unique().tolist()}")

    # 4. Handle missing values (simple drop for now)
    print(f"\n   Missing values before: {df.isnull().sum().sum()}")
    df = df.dropna()
    print(f"   Missing values after: {df.isnull().sum().sum()}")
    print(f"   Rows after cleaning: {len(df)}")

    # 5. Split features and labels
    predictors = config.get("predictors", [])
    if not all(col in df.columns for col in predictors):
        missing = [col for col in predictors if col not in df.columns]
        raise ValueError(f"Missing predictor columns: {missing}")

    X = df[predictors].copy()
    y = df[label_col].copy()

    print("\n   ✅ Preprocessing complete!")
    print(f"      Features shape: {X.shape}")
    print(f"      Labels shape: {y.shape}")
    print(f"      Label distribution: {y.value_counts().to_dict()}")

    return X, y


def _detect_feature_types(X: pd.DataFrame, config: dict) -> Tuple[list, list]:
    """Split predictors into numeric vs categorical lists using config hint."""

    numeric_candidates = config.get("int32_columns", [])
    numeric_features = [col for col in numeric_candidates if col in X.columns]
    categorical_features = [col for col in X.columns if col not in numeric_features]
    return numeric_features, categorical_features


def _summarize_saved_files(files: Dict[str, pd.DataFrame]) -> None:
    for filename, data in files.items():
        shape = getattr(data, "shape", (len(data),))
        if isinstance(shape, tuple):
            shape_repr = shape
        else:
            shape_repr = (len(data),)
        print(f"   ✓ Saved: {filename} (shape: {shape_repr})")


def run_preprocessing(config: dict, output_dir: Path) -> bool:
    """
    Complete preprocessing pipeline.

    Args:
        config: Configuration dictionary
        output_dir: Directory to save processed data

    Returns:
        True if successful
    """
    try:
        # Load raw data
        raw_data_dir = Path(config["dataset_dir"])
        df = load_raw_data(raw_data_dir)

        # Preprocess
        X, y = preprocess_data(df, config)

        # Save preprocessed data (basic split for now)
        output_dir.mkdir(parents=True, exist_ok=True)

        import joblib
        from sklearn.model_selection import train_test_split
        from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC
        from imblearn.under_sampling import RandomUnderSampler
        from src.features.transformation import (
            fit_feature_engineer,
            transform_with_artifacts,
            save_feature_engineering_artifacts,
        )

        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )

        numeric_features, categorical_features = _detect_feature_types(X_train, config)
        print("\n🧮 Feature engineering setup:")
        print(f"   Numeric features: {numeric_features}")
        print(f"   Categorical features: {categorical_features}")

        # Sampling strategies
        rus = RandomUnderSampler(random_state=42)
        ros = RandomOverSampler(random_state=42)
        if categorical_features:
            categorical_indices = [
                X_train.columns.get_loc(col)
                for col in categorical_features
                if col in X_train.columns
            ]
            smote_sampler = SMOTENC(
                categorical_features=categorical_indices,
                random_state=42,
            )
        else:
            smote_sampler = SMOTE(random_state=42)

        X_rus, y_rus = rus.fit_resample(X_train, y_train)
        X_ros, y_ros = ros.fit_resample(X_train, y_train)
        X_sm, y_sm = smote_sampler.fit_resample(X_train, y_train)

        # Feature engineering
        artifacts, X_train_feng = fit_feature_engineer(
            X_train,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )
        X_valid_feng = transform_with_artifacts(X_valid, artifacts)
        X_test_feng = transform_with_artifacts(X_test, artifacts)
        X_rus_feng = transform_with_artifacts(X_rus, artifacts)
        X_ros_feng = transform_with_artifacts(X_ros, artifacts)
        X_sm_feng = transform_with_artifacts(X_sm, artifacts)

        print("   ✓ Feature engineering artifacts fitted and applied")
        artifact_path = save_feature_engineering_artifacts(artifacts, output_dir)
        print(f"   ✓ Artifacts saved to {artifact_path}")

        # Save all versions
        print(f"\n💾 Saving processed data to {output_dir}...")

        datasets: Dict[str, pd.DataFrame] = {
            "X_train.pkl": X_train,
            "y_train.pkl": y_train,
            "X_valid.pkl": X_valid,
            "y_valid.pkl": y_valid,
            "X_test.pkl": X_test,
            "y_test.pkl": y_test,
            "X_rus.pkl": X_rus,
            "y_rus.pkl": y_rus,
            "X_ros.pkl": X_ros,
            "y_ros.pkl": y_ros,
            "X_sm.pkl": X_sm,
            "y_sm.pkl": y_sm,
            "X_train_feng.pkl": X_train_feng,
            "y_train_feng.pkl": y_train,
            "X_valid_feng.pkl": X_valid_feng,
            "y_valid_feng.pkl": y_valid,
            "X_test_feng.pkl": X_test_feng,
            "y_test_feng.pkl": y_test,
            "X_rus_feng.pkl": X_rus_feng,
            "y_rus_feng.pkl": y_rus,
            "X_ros_feng.pkl": X_ros_feng,
            "y_ros_feng.pkl": y_ros,
            "X_sm_feng.pkl": X_sm_feng,
            "y_sm_feng.pkl": y_sm,
        }

        for filename, data in datasets.items():
            filepath = output_dir / filename
            joblib.dump(data, filepath)

        _summarize_saved_files(datasets)

        print("\n   ✅ All data saved successfully!")
        return True

    except Exception as e:
        print(f"\n   ❌ Preprocessing failed: {e}")
        import traceback

        traceback.print_exc()
        return False
