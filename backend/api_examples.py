"""
Example API calls using requests library.
Shows how to interact with the API from Python.
"""

import requests
import json

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health check."""
    print("\n=== Health Check ===")
    response = requests.get(f"{BASE_URL}/api/health")
    print(json.dumps(response.json(), indent=2))


def test_model_performance():
    """Get model performance metrics."""
    print("\n=== Model Performance ===")
    response = requests.get(f"{BASE_URL}/api/models/performance")
    data = response.json()
    print(f"Average Test Accuracy: {data['summary']['average_test_accuracy']:.4f}")
    print(f"Average Test AUC: {data['summary']['average_test_auc']:.4f}")


def test_predict_system():
    """Predict a complete exoplanet system."""
    print("\n=== Predict System (Kepler-442b) ===")

    # Transit data for Kepler-442b (known habitable zone planet)
    transit_data = {
        "period_days": 112.3,
        "transit_depth_ppm": 376,
        "transit_duration_hrs": 4.2,
        "stellar_teff": 4402,
        "stellar_radius": 0.601,
        "stellar_logg": 4.653,
        "impact_parameter": 0.3
    }

    response = requests.post(
        f"{BASE_URL}/api/predict-system",
        json=transit_data
    )

    if response.status_code == 200:
        data = response.json()

        print(f"\nPlanet Type: {data['planet_type']}")
        print(f"Temperature: {data['temperature_category']}")

        planet = data['system_parameters']['planet']
        print(f"\nPlanet Parameters:")
        print(f"  Radius: {planet['radius_rearth']:.2f} Earth radii")
        print(f"  Mass: {planet['mass_mearth']:.2f} Earth masses")
        print(f"  Equilibrium Temp: {planet['equilibrium_temp_K']:.0f} K")

        orbit = data['system_parameters']['orbit']
        print(f"\nOrbital Parameters:")
        print(f"  Semi-major axis: {orbit['semi_major_axis_au']:.3f} AU")
        print(f"  Velocity: {orbit['orbital_velocity_km_s']:.1f} km/s")

        hz = data['habitable_zone']
        print(f"\nHabitability:")
        print(f"  In habitable zone: {hz['in_hz_conservative']}")
        print(f"  HZ boundaries: {hz['hz_inner_au']:.3f} - {hz['hz_outer_au']:.3f} AU")

        validation = data['validation']
        print(f"\nValidation:")
        print(f"  Status: {'✓ PASS' if validation['validated'] else '✗ FAIL'}")
        print(f"  Density: {validation['density_g_cm3']:.2f} g/cm³")

        if validation['warnings']:
            print("\n  Warnings:")
            for warning in validation['warnings']:
                print(f"    - {warning}")

        if validation['errors']:
            print("\n  Errors:")
            for error in validation['errors']:
                print(f"    - {error}")

    else:
        print(f"Error: {response.status_code}")
        print(response.json())


def test_hot_jupiter():
    """Predict a hot Jupiter."""
    print("\n=== Predict System (Hot Jupiter) ===")

    # Typical hot Jupiter parameters
    transit_data = {
        "period_days": 3.5,
        "transit_depth_ppm": 15000,  # Large planet
        "transit_duration_hrs": 2.5,
        "stellar_teff": 6000,
        "stellar_radius": 1.2,
        "stellar_logg": 4.4,
        "impact_parameter": 0.2
    }

    response = requests.post(
        f"{BASE_URL}/api/predict-system",
        json=transit_data
    )

    if response.status_code == 200:
        data = response.json()

        print(f"\nPlanet Type: {data['planet_type']}")
        print(f"Temperature: {data['temperature_category']}")

        planet = data['system_parameters']['planet']
        print(f"  Radius: {planet['radius_rearth']:.2f} Earth radii")
        print(f"  Temp: {planet['equilibrium_temp_K']:.0f} K")

        hz = data['habitable_zone']
        print(f"  In HZ: {hz['in_hz_conservative']}")

    else:
        print(f"Error: {response.status_code}")


def main():
    """Run all examples."""
    print("="*60)
    print("API USAGE EXAMPLES")
    print("="*60)
    print("\nMake sure the server is running:")
    print("  ./start_server.sh")
    print("\nOr:")
    print("  cd backend && uv run uvicorn api:app --port 8000")

    try:
        test_health()
        test_model_performance()
        test_predict_system()
        test_hot_jupiter()

        print("\n" + "="*60)
        print("✓ All examples completed successfully!")
        print("="*60)

    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API server")
        print("Make sure the server is running on http://localhost:8000")
        print("\nStart it with: ./start_server.sh")


if __name__ == "__main__":
    main()
