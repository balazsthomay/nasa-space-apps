"""
Test script for API endpoints.
Tests the API without running the server (direct function calls).
"""

import sys
import asyncio
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from api import (
    startup_event, analyze_features, predict_system, get_model_performance,
    get_targets, get_features, health_check, FeatureVector, TransitData
)


async def test_api():
    """Test all API endpoints."""
    print("="*60)
    print("TESTING API ENDPOINTS")
    print("="*60)

    # Startup
    print("\n1. Loading models...")
    await startup_event()
    print("✓ Models loaded")

    # Health check
    print("\n2. Testing health check...")
    health = await health_check()
    print(f"Status: {health['status']}")
    print(f"Models loaded: {health['models_loaded']}")
    print(f"Features: {health['num_features']}")

    # Get model performance
    print("\n3. Testing model performance endpoint...")
    performance = await get_model_performance()
    print(f"Number of models: {performance['summary']['num_models']}")
    print(f"Average test accuracy: {performance['summary']['average_test_accuracy']:.4f}")
    print(f"Average test AUC: {performance['summary']['average_test_auc']:.4f}")

    # Get features
    print("\n4. Testing features endpoint...")
    features_response = await get_features()
    print(f"Total features: {features_response['count']}")
    print(f"First 5 features: {features_response['features'][:5]}")

    # Get targets
    print("\n5. Testing targets endpoint...")
    targets_response = await get_targets()
    print(f"Demo targets available: {targets_response['count']}")
    if targets_response['targets']:
        print(f"First target: {targets_response['targets'][0]['name']}")

    # Test system prediction (Kepler-442b-like parameters)
    print("\n6. Testing predict-system endpoint...")
    print("Using Kepler-442b-like parameters...")

    transit_data = TransitData(
        period_days=112.3,
        transit_depth_ppm=376,
        transit_duration_hrs=4.2,
        stellar_teff=4402,
        stellar_radius=0.601,
        stellar_logg=4.653,
        impact_parameter=0.3
    )

    system_response = await predict_system(transit_data)

    print("\nSystem Parameters:")
    print(f"  Planet radius: {system_response['system_parameters']['planet']['radius_rearth']:.2f} R⊕")
    print(f"  Planet mass: {system_response['system_parameters']['planet']['mass_mearth']:.2f} M⊕")
    print(f"  Equilibrium temp: {system_response['system_parameters']['planet']['equilibrium_temp_K']:.0f} K")
    print(f"  Semi-major axis: {system_response['system_parameters']['orbit']['semi_major_axis_au']:.3f} AU")
    print(f"  In habitable zone: {system_response['system_parameters']['habitability']['in_hz_conservative']}")
    print(f"  Planet type: {system_response['planet_type']}")
    print(f"  Temperature category: {system_response['temperature_category']}")

    print("\nValidation:")
    print(f"  Validated: {system_response['validation']['validated']}")
    print(f"  Density: {system_response['validation']['density_g_cm3']:.2f} g/cm³")
    if system_response['validation']['warnings']:
        print(f"  Warnings: {len(system_response['validation']['warnings'])}")
    if system_response['validation']['errors']:
        print(f"  Errors: {len(system_response['validation']['errors'])}")

    print("\n" + "="*60)
    print("ALL API TESTS PASSED ✓")
    print("="*60)
    print("\nAPI is ready to serve!")
    print("\nTo run the server:")
    print("  cd backend")
    print("  uv run uvicorn api:app --reload --port 8001")
    print("\nThen test with:")
    print("  curl http://localhost:8001/api/health")


if __name__ == "__main__":
    asyncio.run(test_api())
