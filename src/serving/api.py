import sys
import numpy as np

if not hasattr(np, "_core"):
    sys.modules["numpy._core"] = np.core
    sys.modules["numpy._core._multiarray_umath"] = np.core._multiarray_umath

import joblib
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict, field_validator

from src.features.transformation import (
    FeatureEngineeringArtifacts,
    load_feature_engineering_artifacts,
    transform_with_artifacts,
)


model_artifacts = {}
RAW_FEATURE_COLUMNS = ["stasiun", "pm10", "pm25", "so2", "co", "o3", "no2"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for loading model artifacts at startup.
    """
    project_root = Path(__file__).parent.parent.parent
    production_dir = project_root / "models" / "production"

    production_model_path = production_dir / "model.pkl"
    if production_model_path.exists():
        try:
            model_artifacts["model"] = joblib.load(production_model_path)
            print(f"✓ Model loaded successfully from {production_model_path}")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            model_artifacts["model"] = None
    else:
        print(f"⚠ Warning: Model file not found at {production_model_path}")
        model_artifacts["model"] = None

    artifacts = load_feature_engineering_artifacts(production_dir)
    if artifacts is not None:
        model_artifacts["feature_engineering"] = artifacts
        print(f"✓ Feature engineering artifacts loaded from {production_dir}")
    else:
        model_artifacts["feature_engineering"] = None
        print(
            "⚠ Warning: Feature engineering artifacts not found. "
            "API will use raw features."
        )

    yield

    model_artifacts.clear()


app = FastAPI(
    title="Air Quality Prediction API",
    description="API for predicting air quality category based on sensor data",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    """
    Request model for air quality prediction.

    Based on the predictors defined in config.yaml:
    - stasiun, pm10, pm25, so2, co, o3, no2
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "stasiun": "DKI1 (Bunderan HI)",
                "pm10": 50.5,
                "pm25": 30.2,
                "so2": 15.0,
                "co": 5.5,
                "o3": 45.0,
                "no2": 25.0,
            }
        }
    )

    stasiun: str = Field(
        ...,
        description="Station name (e.g., 'DKI1 (Bunderan HI)')",
        json_schema_extra={"example": "DKI1 (Bunderan HI)"},
    )
    pm10: float = Field(
        ...,
        ge=-1,
        le=800,
        description="PM10 concentration",
        json_schema_extra={"example": 50.5},
    )
    pm25: float = Field(
        ...,
        ge=-1,
        le=400,
        description="PM2.5 concentration",
        json_schema_extra={"example": 30.2},
    )
    so2: float = Field(
        ...,
        ge=-1,
        le=500,
        description="SO2 concentration",
        json_schema_extra={"example": 15.0},
    )
    co: float = Field(
        ...,
        ge=-1,
        le=100,
        description="CO concentration",
        json_schema_extra={"example": 5.5},
    )
    o3: float = Field(
        ...,
        ge=-1,
        le=160,
        description="O3 concentration",
        json_schema_extra={"example": 45.0},
    )
    no2: float = Field(
        ...,
        ge=-1,
        le=100,
        description="NO2 concentration",
        json_schema_extra={"example": 25.0},
    )

    @field_validator("stasiun")
    @classmethod
    def validate_station(cls, v):
        """Validate station name against allowed values."""
        valid_stations = [
            "DKI1 (Bunderan HI)",
            "DKI2 (Kelapa Gading)",
            "DKI3 (Jagakarsa)",
            "DKI4 (Lubang Buaya)",
            "DKI5 (Kebon Jeruk) Jakarta Barat",
        ]
        if v not in valid_stations:
            raise ValueError(
                f"Invalid station. Must be one of: {', '.join(valid_stations)}"
            )
        return v


class PredictionResponse(BaseModel):
    """Response model for prediction."""

    prediction: str = Field(..., description="Predicted air quality category")
    confidence: Optional[float] = Field(
        None, description="Prediction confidence (if available)"
    )
    model_info: Optional[str] = Field(None, description="Model information")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Air Quality Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "docs": "/docs",
        },
    }


@app.get("/health")
async def health_check():
    """
    Health check endpoint.

    Returns the health status of the API and whether the model is loaded.
    """
    model_loaded = model_artifacts.get("model") is not None

    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "message": (
            "Model ready for predictions"
            if model_loaded
            else "Model not loaded"
        ),
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict air quality category based on sensor data.

    Args:
        request: PredictionRequest with sensor measurements

    Returns:
        PredictionResponse with predicted category

    Raises:
        HTTPException: If model is not loaded or prediction fails
    """
    if model_artifacts.get("model") is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Please check server logs.",
        )

    try:
        import pandas as pd

        model = model_artifacts["model"]
        artifacts: Optional[FeatureEngineeringArtifacts] = model_artifacts.get(
            "feature_engineering"
        )

        raw_df = pd.DataFrame(
            [
                {
                    "stasiun": request.stasiun,
                    "pm10": request.pm10,
                    "pm25": request.pm25,
                    "so2": request.so2,
                    "co": request.co,
                    "o3": request.o3,
                    "no2": request.no2,
                }
            ],
            columns=RAW_FEATURE_COLUMNS,
        )

        if artifacts is not None:
            engineered_df = transform_with_artifacts(raw_df, artifacts)
            features = engineered_df.to_numpy()
        else:
            features = raw_df.to_numpy()

        prediction = model.predict(features)[0]

        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            confidence = float(max(proba))

        return PredictionResponse(
            prediction=str(prediction),
            confidence=confidence,
            model_info=str(type(model).__name__),
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {str(e)}"
        )


@app.get("/model-info")
async def model_info():
    """
    Get information about the loaded model.
    """
    if model_artifacts.get("model") is None:
        return {"model_loaded": False, "message": "No model loaded"}

    model = model_artifacts["model"]

    artifacts: Optional[FeatureEngineeringArtifacts] = model_artifacts.get(
        "feature_engineering"
    )
    if artifacts:
        feature_list = (
            artifacts.numeric_features + artifacts.encoded_feature_names
        )
    else:
        feature_list = RAW_FEATURE_COLUMNS

    return {
        "model_loaded": True,
        "model_type": type(model).__name__,
        "has_predict_proba": hasattr(model, "predict_proba"),
        "features": feature_list,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
