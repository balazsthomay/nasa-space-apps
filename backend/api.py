"""
FastAPI backend for Exoplanet System Detective.
Serves ML predictions, SHAP explanations, and physics calculations.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging

from ensemble_predictor import EnsemblePredictor
from physics_engine import PhysicsEngine

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Exoplanet System Detective API",
    description="ML-powered exoplanet detection with physics-based validation",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global objects (loaded at startup)
predictor = None
physics_engine = None
feature_names = []
model_metrics = {}
demo_targets = []


# Pydantic models for request/response
class FeatureVector(BaseModel):
    """Feature vector for prediction."""
    features: Dict[str, float] = Field(..., description="Dictionary of feature name -> value")


class TransitData(BaseModel):
    """Transit signal data."""
    period_days: float = Field(..., gt=0, description="Orbital period in days")
    transit_depth_ppm: float = Field(..., gt=0, description="Transit depth in parts per million")
    transit_duration_hrs: float = Field(..., gt=0, description="Transit duration in hours")
    stellar_teff: float = Field(..., gt=0, description="Stellar effective temperature (K)")
    stellar_radius: float = Field(..., gt=0, description="Stellar radius (solar radii)")
    stellar_logg: float = Field(..., description="Stellar surface gravity log10(cm/s²)")
    impact_parameter: Optional[float] = Field(0.5, ge=0, le=1, description="Impact parameter")


class PredictionResponse(BaseModel):
    """Response from prediction endpoint."""
    is_planet: bool
    probability: float
    confidence: float
    individual_models: Dict[str, float]
    ensemble_method: str


class SystemPrediction(BaseModel):
    """Complete system prediction with physics."""
    prediction: PredictionResponse
    system_parameters: Dict[str, Any]
    validation: Dict[str, Any]
    top_features: List[Dict[str, Any]]


class TargetInfo(BaseModel):
    """Information about a demo target."""
    id: str
    name: str
    koi_name: str
    disposition: str
    period_days: float
    radius_rearth: float


# Startup event: load models
@app.on_event("startup")
async def startup_event():
    """Load models and data at startup."""
    global predictor, physics_engine, feature_names, model_metrics, demo_targets

    logger.info("Loading models and data...")

    try:
        # Load ensemble predictor
        predictor = EnsemblePredictor()
        predictor.load_models()

        # Load training data for SHAP background
        train_df = pd.read_csv("data/processed/features_train.csv")
        exclude_cols = ['target', 'disposition', 'kepid', 'koi_name']
        feature_cols = [col for col in train_df.columns if col not in exclude_cols]

        X_train = train_df[feature_cols].copy()
        X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in feature_cols:
            if X_train[col].isnull().any():
                X_train[col].fillna(X_train[col].median(), inplace=True)

        # Create SHAP explainers
        predictor.create_shap_explainers(X_train, sample_size=100)

        # Load feature names
        with open("models/feature_names.json", 'r') as f:
            feature_names = json.load(f)

        # Load model metrics
        with open("models/metrics.json", 'r') as f:
            model_metrics = json.load(f)

        # Initialize physics engine
        physics_engine = PhysicsEngine()

        # Load demo targets
        test_df = pd.read_csv("data/processed/features_test.csv")
        demo_targets = [
            {
                "id": str(idx),
                "name": row.get('kepler_name', row['koi_name']),
                "koi_name": row['koi_name'],
                "disposition": row['disposition'],
                "period_days": float(row['orbital_period_days']),
                "radius_rearth": float(row['planet_radius_rearth'])
            }
            for idx, row in test_df.head(20).iterrows()
        ]

        logger.info("✓ Models and data loaded successfully!")

    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Exoplanet System Detective API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "POST /api/analyze": "Analyze features and predict if exoplanet",
            "POST /api/predict-system": "Predict complete system parameters",
            "GET /api/models/performance": "Get model performance metrics",
            "GET /api/targets": "Get demo target list",
            "GET /api/features": "Get feature names"
        }
    }


@app.post("/api/analyze", response_model=PredictionResponse)
async def analyze_features(feature_vector: FeatureVector):
    """
    Analyze feature vector and predict if it's a planet.

    Args:
        feature_vector: Dictionary of features

    Returns:
        Prediction with probabilities and confidence
    """
    try:
        # Convert to DataFrame
        features_dict = feature_vector.features

        # Check if all required features are present
        missing_features = set(feature_names) - set(features_dict.keys())
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing features: {list(missing_features)}"
            )

        # Create DataFrame in correct order
        X = pd.DataFrame([features_dict])[feature_names]

        # Handle inf and missing values
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in feature_names:
            if X[col].isnull().any():
                X[col].fillna(0, inplace=True)  # Simple fallback

        # Get predictions
        results = predictor.predict_ensemble(X, voting='soft')

        return PredictionResponse(
            is_planet=bool(results['predictions'][0] == 1),
            probability=float(results['probabilities'][0]),
            confidence=float(results['confidence'][0]),
            individual_models={
                model_name: float(proba[0])
                for model_name, proba in results['individual_probabilities'].items()
            },
            ensemble_method=results['voting_method']
        )

    except Exception as e:
        logger.error(f"Error in analyze endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict-system")
async def predict_system(transit_data: TransitData):
    """
    Predict complete exoplanet system from transit data.

    Combines ML prediction with physics calculations.

    Args:
        transit_data: Transit signal measurements

    Returns:
        Complete system parameters with validation
    """
    try:
        # Calculate system parameters using physics engine
        system_params = physics_engine.calculate_system_parameters(
            period_days=transit_data.period_days,
            transit_depth_ppm=transit_data.transit_depth_ppm,
            transit_duration_hrs=transit_data.transit_duration_hrs,
            stellar_teff=transit_data.stellar_teff,
            stellar_radius=transit_data.stellar_radius,
            stellar_logg=transit_data.stellar_logg,
            impact_parameter=transit_data.impact_parameter
        )

        # Validate physics
        validation = physics_engine.validate_system(system_params)

        # For ML prediction, we'd need to construct full feature vector
        # For now, return physics-based analysis
        # In a full implementation, you'd extract features from transit_data

        response = {
            "system_parameters": system_params,
            "validation": validation,
            "habitable_zone": system_params['habitability'],
            "planet_type": _classify_planet_type(system_params['planet']['radius_rearth']),
            "temperature_category": _classify_temperature(system_params['planet']['equilibrium_temp_K'])
        }

        return response

    except Exception as e:
        logger.error(f"Error in predict-system endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models/performance")
async def get_model_performance():
    """Get performance metrics for all models."""
    return {
        "models": model_metrics,
        "summary": {
            "num_models": len(model_metrics),
            "average_test_accuracy": np.mean([
                m['test']['accuracy'] for m in model_metrics.values()
            ]),
            "average_test_auc": np.mean([
                m['test']['roc_auc'] for m in model_metrics.values()
            ])
        }
    }


@app.get("/api/targets")
async def get_targets():
    """Get list of demo targets."""
    return {
        "targets": demo_targets,
        "count": len(demo_targets)
    }


@app.get("/api/features")
async def get_features():
    """Get list of feature names."""
    return {
        "features": feature_names,
        "count": len(feature_names)
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": predictor is not None,
        "physics_engine_loaded": physics_engine is not None,
        "num_features": len(feature_names)
    }


# Helper functions
def _classify_planet_type(radius_rearth: float) -> str:
    """Classify planet by size."""
    if radius_rearth < 1.5:
        return "Rocky Planet"
    elif radius_rearth < 2.0:
        return "Super-Earth"
    elif radius_rearth < 4.0:
        return "Mini-Neptune"
    elif radius_rearth < 10:
        return "Neptune-like"
    else:
        return "Gas Giant"


def _classify_temperature(temp_k: float) -> str:
    """Classify by temperature."""
    if temp_k < 200:
        return "Cold"
    elif temp_k < 300:
        return "Temperate"
    elif temp_k < 1000:
        return "Warm"
    else:
        return "Hot"


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Exoplanet System Detective API...")
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
