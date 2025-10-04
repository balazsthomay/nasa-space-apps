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
        # Get project root (parent of backend directory)
        project_root = Path(__file__).parent.parent

        # Load ensemble predictor
        predictor = EnsemblePredictor(models_dir=str(project_root / "models"))
        predictor.load_models()

        # Load training data for SHAP background
        train_df = pd.read_csv(project_root / "data/processed/features_train.csv")
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
        with open(project_root / "models/feature_names.json", 'r') as f:
            feature_names = json.load(f)

        # Load model metrics
        with open(project_root / "models/metrics.json", 'r') as f:
            model_metrics = json.load(f)

        # Initialize physics engine
        physics_engine = PhysicsEngine()

        # Load demo targets
        test_df = pd.read_csv(project_root / "data/processed/features_test.csv")
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

        # Construct feature vector from transit data and calculated params
        features = _construct_features_from_transit(transit_data, system_params)

        # Create DataFrame
        X = pd.DataFrame([features])[feature_names]

        # Handle inf and missing values
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in feature_names:
            if X[col].isnull().any():
                X[col].fillna(0, inplace=True)

        # Get ML prediction from ensemble
        ml_results = predictor.predict_ensemble(X, voting='soft')

        # Get SHAP explanations
        top_features = []
        try:
            shap_explanations = predictor.explain_predictions(X)
            # Get ensemble feature importance
            ensemble_importance = predictor.get_ensemble_feature_importance(shap_explanations)
            # Top 10 features
            top_features = [
                {"feature": name, "importance": float(importance)}
                for name, importance in ensemble_importance[:10]
            ]
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")

        # Format prediction
        prediction = PredictionResponse(
            is_planet=bool(ml_results['predictions'][0] == 1),
            probability=float(ml_results['probabilities'][0]),
            confidence=float(ml_results['confidence'][0]),
            individual_models={
                model_name: float(proba[0])
                for model_name, proba in ml_results['individual_probabilities'].items()
            },
            ensemble_method=ml_results['voting_method']
        )

        response = {
            "prediction": prediction.dict(),
            "system_parameters": system_params,
            "validation": validation,
            "habitable_zone": system_params['habitability'],
            "planet_type": _classify_planet_type(system_params['planet']['radius_rearth']),
            "temperature_category": _classify_temperature(system_params['planet']['equilibrium_temp_K']),
            "top_features": top_features
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
def _construct_features_from_transit(transit_data: TransitData, system_params: Dict) -> Dict[str, float]:
    """
    Construct 46-feature vector matching the exact features from training.
    """
    # Extract calculated params
    planet = system_params['planet']
    orbit = system_params['orbit']
    hab = system_params['habitability']

    # Basic params
    period_days = transit_data.period_days
    depth_ppm = transit_data.transit_depth_ppm
    duration_hrs = transit_data.transit_duration_hrs
    stellar_teff = transit_data.stellar_teff
    stellar_radius = transit_data.stellar_radius
    stellar_logg = transit_data.stellar_logg
    impact_param = transit_data.impact_parameter

    # Derived values
    planet_radius = planet['radius_rearth']
    planet_teq = planet['equilibrium_temp_K']
    insolation = planet['insolation_earth']
    sma = orbit['semi_major_axis_au']

    # Estimate SNR
    snr_estimate = np.sqrt(depth_ppm) / 10.0

    # Radius ratio
    radius_ratio = planet_radius / (stellar_radius * 109.2)  # Convert to same units
    expected_depth = (radius_ratio ** 2) * 1e6
    depth_anomaly = abs(depth_ppm - expected_depth) / depth_ppm if depth_ppm > 0 else 0.0

    features = {
        # Transit features
        'transit_depth_ppm': depth_ppm,
        'transit_duration_hrs': duration_hrs,
        'orbital_period_days': period_days,
        'impact_parameter': impact_param,
        'transit_snr': snr_estimate,
        'log_period': np.log10(period_days),
        'log_depth': np.log10(depth_ppm),
        'duration_period_ratio': duration_hrs / (period_days * 24),
        'transit_shape_factor': impact_param * duration_hrs,

        # Stellar features
        'stellar_teff_K': stellar_teff,
        'stellar_radius_rsun': stellar_radius,
        'stellar_logg': stellar_logg,
        'stellar_teff_normalized': stellar_teff / 5778.0,
        'stellar_density_proxy': (10 ** stellar_logg) / (stellar_radius ** 2),
        'stellar_luminosity_proxy': (stellar_radius ** 2) * ((stellar_teff / 5778) ** 4),

        # Planetary features
        'planet_radius_rearth': planet_radius,
        'planet_teq_K': planet_teq,
        'insolation_flux_earth': insolation,
        'log_planet_radius': np.log10(planet_radius),
        'log_insolation': np.log10(insolation) if insolation > 0 else 0,

        # Planet type flags
        'is_rocky_size': float(planet_radius < 1.5),
        'is_super_earth_size': float(1.5 <= planet_radius < 2.0),
        'is_mini_neptune_size': float(2.0 <= planet_radius < 4.0),
        'is_gas_giant_size': float(planet_radius >= 4.0),

        # Temperature flags
        'is_hot': float(planet_teq > 1000),
        'is_warm': float(300 <= planet_teq <= 1000),
        'is_temperate': float(200 <= planet_teq < 300),
        'is_cold': float(planet_teq < 200),

        # Orbital features
        'semi_major_axis_au': sma,
        'orbital_velocity_kms': orbit['orbital_velocity_km_s'],
        'density_proxy': planet['mass_mearth'] / (planet_radius ** 3),
        'in_habitable_zone': float(hab['in_hz_conservative']),
        'hz_distance_normalized': sma / hab['hz_center_au'],
        'transit_probability': orbit['transit_probability'],

        # Transit consistency checks
        'radius_ratio': radius_ratio,
        'expected_transit_depth_ppm': expected_depth,
        'depth_anomaly': depth_anomaly,

        # Uncertainty estimates (10% default)
        'period_uncertainty': period_days * 0.1,
        'radius_uncertainty': planet_radius * 0.1,
        'depth_uncertainty': depth_ppm * 0.1,
        'measurement_quality': snr_estimate / 10.0,  # Normalized quality

        # False positive flags
        'fp_flag_not_transit_like': float(depth_anomaly > 0.5 or impact_param > 0.95),
        'fp_flag_stellar_eclipse': float(depth_ppm > 10000),  # Very deep = binary eclipse
        'fp_flag_centroid_offset': 0.0,  # Would need centroid data
        'fp_flag_ephemeris_match': 0.0,  # Would need multiple transits
        'total_fp_flags': 0.0
    }

    # Calculate total FP flags
    features['total_fp_flags'] = (
        features['fp_flag_not_transit_like'] +
        features['fp_flag_stellar_eclipse'] +
        features['fp_flag_centroid_offset'] +
        features['fp_flag_ephemeris_match']
    )

    return features


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
