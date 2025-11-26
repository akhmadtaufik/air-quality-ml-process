import joblib
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURE_ARTIFACT_FILENAME = "feature_engineering_artifacts.pkl"


@dataclass
class FeatureEngineeringArtifacts:
    """Container that keeps every component needed for feature engineering."""

    scaler: Optional[StandardScaler]
    encoder: Optional[OneHotEncoder]
    numeric_features: List[str]
    categorical_features: List[str]
    encoded_feature_names: List[str]


def _validate_columns(data: pd.DataFrame, columns: List[str]) -> None:
    missing = [col for col in columns if col not in data.columns]
    if missing:
        raise ValueError(f"Missing columns for transformation: {missing}")


def fit_feature_engineer(
    data: pd.DataFrame,
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> Tuple[FeatureEngineeringArtifacts, pd.DataFrame]:
    """Fit scaler/encoder on data and return transformed dataframe plus artifacts."""

    numeric_features = list(numeric_features or [])
    categorical_features = list(categorical_features or [])

    _validate_columns(data, numeric_features + categorical_features)

    scaler = None
    if numeric_features:
        scaler = StandardScaler()
        scaler.fit(data[numeric_features])

    encoder = None
    encoded_names: List[str] = []
    if categorical_features:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoder.fit(data[categorical_features])
        encoded_names = encoder.get_feature_names_out(
            categorical_features
        ).tolist()

    artifacts = FeatureEngineeringArtifacts(
        scaler=scaler,
        encoder=encoder,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        encoded_feature_names=encoded_names,
    )

    transformed = transform_with_artifacts(data, artifacts)
    return artifacts, transformed


def transform_with_artifacts(
    data: pd.DataFrame, artifacts: FeatureEngineeringArtifacts
) -> pd.DataFrame:
    """Apply previously fitted artifacts to a dataframe."""

    numeric_df = pd.DataFrame(index=data.index)
    categorical_df = pd.DataFrame(index=data.index)

    if artifacts.numeric_features:
        _validate_columns(data, artifacts.numeric_features)
        if artifacts.scaler is None:
            raise ValueError(
                "Scaler artifact is missing but numeric features were provided."
            )
        numeric_values = artifacts.scaler.transform(
            data[artifacts.numeric_features]
        )
        numeric_df = pd.DataFrame(
            numeric_values,
            columns=artifacts.numeric_features,
            index=data.index,
        )

    if artifacts.categorical_features:
        _validate_columns(data, artifacts.categorical_features)
        if artifacts.encoder is None:
            raise ValueError(
                "Encoder artifact is missing but categorical features were provided."
            )
        encoded_values = artifacts.encoder.transform(
            data[artifacts.categorical_features]
        )
        categorical_df = pd.DataFrame(
            encoded_values,
            columns=artifacts.encoded_feature_names,
            index=data.index,
        )

    if not numeric_df.empty and not categorical_df.empty:
        return pd.concat([numeric_df, categorical_df], axis=1)
    if not numeric_df.empty:
        return numeric_df
    if not categorical_df.empty:
        return categorical_df

    return pd.DataFrame(index=data.index)


def save_feature_engineering_artifacts(
    artifacts: FeatureEngineeringArtifacts, processed_dir: Path
) -> Path:
    """Persist feature engineering artifacts for later reuse."""

    processed_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = processed_dir / FEATURE_ARTIFACT_FILENAME
    joblib.dump(artifacts, artifact_path)
    return artifact_path


def load_feature_engineering_artifacts(
    processed_dir: Path,
) -> Optional[FeatureEngineeringArtifacts]:
    """Load persisted artifacts if they exist."""

    artifact_path = processed_dir / FEATURE_ARTIFACT_FILENAME
    if not artifact_path.exists():
        return None
    return joblib.load(artifact_path)
