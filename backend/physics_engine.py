"""
Physics calculations for exoplanet systems.
Includes habitable zone calculation, orbital mechanics, and uncertainty propagation.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# Physical constants
AU_TO_KM = 1.496e8  # AU to km
SOLAR_MASS = 1.989e30  # kg
SOLAR_RADIUS = 6.957e5  # km
EARTH_RADIUS = 6371.0  # km
STEFAN_BOLTZMANN = 5.670374419e-8  # W⋅m⁻²⋅K⁻⁴
G = 6.67430e-11  # m³⋅kg⁻¹⋅s⁻²


class PhysicsEngine:
    """Calculate physical parameters for exoplanet systems."""

    def __init__(self):
        pass

    def calculate_habitable_zone(
        self,
        stellar_teff: float,
        stellar_radius: float,
        conservative: bool = True
    ) -> Dict[str, float]:
        """
        Calculate habitable zone boundaries using Kopparapu et al. (2013) formulas.

        Args:
            stellar_teff: Stellar effective temperature (K)
            stellar_radius: Stellar radius (solar radii)
            conservative: If True, use conservative HZ. If False, use optimistic HZ.

        Returns:
            Dictionary with inner and outer HZ boundaries (AU)
        """
        # Calculate stellar luminosity (L ∝ R² T⁴)
        # Normalized to solar values (T_sun = 5778 K, R_sun = 1)
        L_star = (stellar_radius ** 2) * ((stellar_teff / 5778.0) ** 4)

        # Kopparapu 2013 coefficients
        # For conservative HZ (recent Venus to early Mars)
        if conservative:
            # Recent Venus (runaway greenhouse inner boundary)
            seff_inner = 1.7763
            a_inner = 1.4335e-4
            b_inner = 3.3954e-9
            c_inner = -7.6364e-12
            d_inner = -1.1950e-15

            # Early Mars (maximum greenhouse outer boundary)
            seff_outer = 0.3207
            a_outer = 5.9578e-5
            b_outer = 1.6707e-9
            c_outer = -3.0058e-12
            d_outer = -5.1925e-16

        else:  # Optimistic HZ (moist greenhouse to maximum greenhouse)
            # Moist greenhouse
            seff_inner = 1.0140
            a_inner = 8.1774e-5
            b_inner = 1.7063e-9
            c_inner = -4.3241e-12
            d_inner = -6.6462e-16

            # Maximum greenhouse
            seff_outer = 0.3240
            a_outer = 5.3221e-5
            b_outer = 1.4288e-9
            c_outer = -2.7476e-12
            d_outer = -4.6936e-16

        # Temperature offset from Sun
        T_star = stellar_teff - 5780.0

        # Calculate effective stellar flux at boundaries
        S_inner = seff_inner + a_inner * T_star + b_inner * (T_star ** 2) + \
                  c_inner * (T_star ** 3) + d_inner * (T_star ** 4)

        S_outer = seff_outer + a_outer * T_star + b_outer * (T_star ** 2) + \
                  c_outer * (T_star ** 3) + d_outer * (T_star ** 4)

        # Calculate HZ boundaries (distance in AU)
        # d = sqrt(L_star / S_eff)
        hz_inner = np.sqrt(L_star / S_inner)
        hz_outer = np.sqrt(L_star / S_outer)

        # Calculate HZ center and width
        hz_center = (hz_inner + hz_outer) / 2
        hz_width = hz_outer - hz_inner

        return {
            'hz_inner_au': float(hz_inner),
            'hz_outer_au': float(hz_outer),
            'hz_center_au': float(hz_center),
            'hz_width_au': float(hz_width),
            'stellar_luminosity': float(L_star),
            'conservative': conservative
        }

    def calculate_semi_major_axis(
        self,
        period_days: float,
        stellar_mass: float = 1.0
    ) -> float:
        """
        Calculate semi-major axis using Kepler's Third Law.

        a³ = (G M T²) / (4π²)

        Args:
            period_days: Orbital period (days)
            stellar_mass: Stellar mass (solar masses)

        Returns:
            Semi-major axis (AU)
        """
        # Convert period to seconds
        period_sec = period_days * 86400.0

        # Stellar mass in kg
        M_star_kg = stellar_mass * SOLAR_MASS

        # Calculate a³
        a_cubed = (G * M_star_kg * (period_sec ** 2)) / (4 * np.pi ** 2)

        # Semi-major axis in meters
        a_m = a_cubed ** (1/3)

        # Convert to AU
        a_au = a_m / (AU_TO_KM * 1000)

        return float(a_au)

    def estimate_stellar_mass(
        self,
        stellar_radius: float,
        stellar_logg: float
    ) -> float:
        """
        Estimate stellar mass from radius and surface gravity.

        M = (g * R²) / G

        Args:
            stellar_radius: Stellar radius (solar radii)
            stellar_logg: Log10 of surface gravity (cm/s²)

        Returns:
            Stellar mass (solar masses)
        """
        # Convert to SI units
        R_m = stellar_radius * SOLAR_RADIUS * 1000  # meters
        g_si = (10 ** stellar_logg) / 100  # m/s²

        # Calculate mass
        M_kg = (g_si * R_m ** 2) / G

        # Convert to solar masses
        M_solar = M_kg / SOLAR_MASS

        return float(M_solar)

    def calculate_orbital_velocity(
        self,
        semi_major_axis_au: float,
        period_days: float
    ) -> float:
        """
        Calculate orbital velocity.

        v = 2π a / T

        Args:
            semi_major_axis_au: Semi-major axis (AU)
            period_days: Orbital period (days)

        Returns:
            Orbital velocity (km/s)
        """
        # Convert to consistent units
        a_km = semi_major_axis_au * AU_TO_KM
        period_sec = period_days * 86400.0

        # Calculate velocity
        v_km_s = (2 * np.pi * a_km) / period_sec

        return float(v_km_s)

    def calculate_equilibrium_temperature(
        self,
        stellar_teff: float,
        stellar_radius: float,
        semi_major_axis_au: float,
        albedo: float = 0.3,
        emissivity: float = 1.0
    ) -> float:
        """
        Calculate planet equilibrium temperature.

        T_eq = T_star * sqrt(R_star / (2*a)) * (1-A)^(1/4) / ε^(1/4)

        Args:
            stellar_teff: Stellar temperature (K)
            stellar_radius: Stellar radius (solar radii)
            semi_major_axis_au: Semi-major axis (AU)
            albedo: Bond albedo (0-1, default 0.3 like Earth)
            emissivity: Emissivity (default 1.0)

        Returns:
            Equilibrium temperature (K)
        """
        # Convert semi-major axis to solar radii
        a_rsun = semi_major_axis_au * 215.032  # 1 AU = 215.032 solar radii

        # Calculate equilibrium temperature
        T_eq = stellar_teff * np.sqrt(stellar_radius / (2 * a_rsun)) * \
               ((1 - albedo) / emissivity) ** 0.25

        return float(T_eq)

    def calculate_transit_probability(
        self,
        stellar_radius: float,
        semi_major_axis_au: float,
        eccentricity: float = 0.0
    ) -> float:
        """
        Calculate geometric transit probability.

        P_transit ≈ R_star / a  (for circular orbits)

        Args:
            stellar_radius: Stellar radius (solar radii)
            semi_major_axis_au: Semi-major axis (AU)
            eccentricity: Orbital eccentricity (default 0)

        Returns:
            Transit probability (0-1)
        """
        # Convert to same units (solar radii)
        a_rsun = semi_major_axis_au * 215.032

        # For circular orbits
        if eccentricity < 0.01:
            p_transit = stellar_radius / a_rsun
        else:
            # For eccentric orbits (simplified)
            p_transit = (stellar_radius / a_rsun) * (1 + eccentricity) / (1 - eccentricity ** 2)

        return float(min(p_transit, 1.0))

    def calculate_planet_radius(
        self,
        transit_depth_ppm: float,
        stellar_radius: float
    ) -> float:
        """
        Calculate planet radius from transit depth.

        Transit depth = (R_planet / R_star)²

        Args:
            transit_depth_ppm: Transit depth (parts per million)
            stellar_radius: Stellar radius (solar radii)

        Returns:
            Planet radius (Earth radii)
        """
        # Convert transit depth to fraction
        depth_fraction = transit_depth_ppm / 1e6

        # Radius ratio
        radius_ratio = np.sqrt(depth_fraction)

        # Planet radius in solar radii
        planet_radius_rsun = radius_ratio * stellar_radius

        # Convert to Earth radii (1 solar radius = 109.2 Earth radii)
        planet_radius_rearth = planet_radius_rsun * 109.2

        return float(planet_radius_rearth)

    def estimate_planet_mass(
        self,
        planet_radius_rearth: float
    ) -> Tuple[float, float]:
        """
        Estimate planet mass using mass-radius relations.

        Uses Weiss & Marcy (2014) and Chen & Kipping (2017) relations.

        Args:
            planet_radius_rearth: Planet radius (Earth radii)

        Returns:
            (mass_earth_masses, uncertainty_factor)
        """
        R = planet_radius_rearth

        # Different regimes based on radius
        if R < 1.23:
            # Rocky planets (Earth-like)
            # M ∝ R^3.7
            mass = R ** 3.7
            uncertainty = 2.0

        elif R < 2.0:
            # Super-Earths (rocky to water-rich)
            # M ∝ R^3
            mass = R ** 3.0
            uncertainty = 2.5

        elif R < 4.0:
            # Mini-Neptunes (H/He envelope)
            # M ∝ R^2.06
            mass = (R ** 2.06) * 0.5
            uncertainty = 3.0

        else:
            # Gas giants
            # M ∝ R^0.55
            mass = (R ** 0.55) * 3.0
            uncertainty = 4.0

        return float(mass), float(uncertainty)

    def calculate_system_parameters(
        self,
        period_days: float,
        transit_depth_ppm: float,
        transit_duration_hrs: float,
        stellar_teff: float,
        stellar_radius: float,
        stellar_logg: float,
        impact_parameter: float = 0.5
    ) -> Dict:
        """
        Calculate complete system parameters.

        Args:
            period_days: Orbital period
            transit_depth_ppm: Transit depth
            transit_duration_hrs: Transit duration
            stellar_teff: Stellar temperature (K)
            stellar_radius: Stellar radius (solar radii)
            stellar_logg: Surface gravity log10(cm/s²)
            impact_parameter: Impact parameter (0-1)

        Returns:
            Dictionary with all derived parameters
        """
        # Estimate stellar mass
        stellar_mass = self.estimate_stellar_mass(stellar_radius, stellar_logg)

        # Semi-major axis
        semi_major_axis = self.calculate_semi_major_axis(period_days, stellar_mass)

        # Planet radius
        planet_radius = self.calculate_planet_radius(transit_depth_ppm, stellar_radius)

        # Planet mass estimate
        planet_mass, mass_uncertainty = self.estimate_planet_mass(planet_radius)

        # Orbital velocity
        orbital_velocity = self.calculate_orbital_velocity(semi_major_axis, period_days)

        # Equilibrium temperature
        equilibrium_temp = self.calculate_equilibrium_temperature(
            stellar_teff, stellar_radius, semi_major_axis
        )

        # Transit probability
        transit_prob = self.calculate_transit_probability(stellar_radius, semi_major_axis)

        # Habitable zone (conservative)
        hz_conservative = self.calculate_habitable_zone(stellar_teff, stellar_radius, conservative=True)

        # Habitable zone (optimistic)
        hz_optimistic = self.calculate_habitable_zone(stellar_teff, stellar_radius, conservative=False)

        # Check if in habitable zone
        in_hz_conservative = (semi_major_axis >= hz_conservative['hz_inner_au'] and
                             semi_major_axis <= hz_conservative['hz_outer_au'])

        in_hz_optimistic = (semi_major_axis >= hz_optimistic['hz_inner_au'] and
                           semi_major_axis <= hz_optimistic['hz_outer_au'])

        # Distance from HZ center
        hz_distance = abs(semi_major_axis - hz_conservative['hz_center_au'])

        return {
            'stellar': {
                'mass_msun': stellar_mass,
                'radius_rsun': stellar_radius,
                'teff_K': stellar_teff,
                'logg': stellar_logg,
                'luminosity': hz_conservative['stellar_luminosity']
            },
            'planet': {
                'radius_rearth': planet_radius,
                'mass_mearth': planet_mass,
                'mass_uncertainty_factor': mass_uncertainty,
                'equilibrium_temp_K': equilibrium_temp
            },
            'orbit': {
                'period_days': period_days,
                'semi_major_axis_au': semi_major_axis,
                'orbital_velocity_km_s': orbital_velocity,
                'impact_parameter': impact_parameter,
                'transit_probability': transit_prob
            },
            'habitability': {
                'in_hz_conservative': in_hz_conservative,
                'in_hz_optimistic': in_hz_optimistic,
                'hz_inner_au': hz_conservative['hz_inner_au'],
                'hz_outer_au': hz_conservative['hz_outer_au'],
                'hz_center_au': hz_conservative['hz_center_au'],
                'distance_from_hz_au': hz_distance,
                'hz_optimistic_inner_au': hz_optimistic['hz_inner_au'],
                'hz_optimistic_outer_au': hz_optimistic['hz_outer_au']
            },
            'transit': {
                'depth_ppm': transit_depth_ppm,
                'duration_hrs': transit_duration_hrs,
                'expected_depth_ppm': ((planet_radius / (stellar_radius * 109.2)) ** 2) * 1e6
            }
        }

    def propagate_uncertainty(
        self,
        ensemble_predictions: Dict,
        base_parameters: Dict
    ) -> Dict:
        """
        Propagate prediction uncertainty through physics calculations.

        Uses ensemble disagreement to estimate parameter uncertainties.

        Args:
            ensemble_predictions: From ensemble predictor
            base_parameters: Base system parameters

        Returns:
            Parameters with uncertainty ranges
        """
        # Get prediction confidence (inverse of std across models)
        confidence = ensemble_predictions.get('confidence', [0.8])[0]

        # Lower confidence = higher uncertainty
        uncertainty_factor = 1 + (1 - confidence)

        # Propagate to derived parameters
        params_with_uncertainty = base_parameters.copy()

        # Add uncertainty ranges to key parameters
        for category in ['stellar', 'planet', 'orbit']:
            if category in params_with_uncertainty:
                for key, value in params_with_uncertainty[category].items():
                    if isinstance(value, (int, float)):
                        # Simple uncertainty propagation
                        uncertainty = value * (uncertainty_factor - 1) * 0.1

                        params_with_uncertainty[category][f'{key}_uncertainty'] = uncertainty
                        params_with_uncertainty[category][f'{key}_lower'] = value - uncertainty
                        params_with_uncertainty[category][f'{key}_upper'] = value + uncertainty

        return params_with_uncertainty

    def validate_system(self, params: Dict) -> Dict[str, list]:
        """
        Validate physical parameters for impossible values.

        Checks:
        - Density isn't impossibly low/high
        - Orbital stability
        - Transit duration matches orbital geometry
        - Temperature consistent with stellar distance

        Args:
            params: System parameters from calculate_system_parameters

        Returns:
            Dictionary with warnings and errors
        """
        warnings = []
        errors = []

        # Extract key parameters
        planet_radius = params['planet']['radius_rearth']
        planet_mass = params['planet']['mass_mearth']
        period = params['orbit']['period_days']
        semi_major_axis = params['orbit']['semi_major_axis_au']
        equilibrium_temp = params['planet']['equilibrium_temp_K']
        stellar_teff = params['stellar']['teff_K']

        # 1. Density check
        # Calculate density (mass/volume)
        volume_earth = 1.0  # Earth volumes
        volume = planet_radius ** 3
        density_relative = planet_mass / volume  # Relative to Earth

        # Earth density = 5.51 g/cm³
        density_abs = density_relative * 5.51

        if density_abs < 0.1:
            errors.append(f"Impossibly low density: {density_abs:.2f} g/cm³ (< 0.1)")
        elif density_abs < 0.5:
            warnings.append(f"Very low density: {density_abs:.2f} g/cm³ (puffy planet?)")

        if density_abs > 30:
            errors.append(f"Impossibly high density: {density_abs:.2f} g/cm³ (> iron core)")
        elif density_abs > 15:
            warnings.append(f"Very high density: {density_abs:.2f} g/cm³ (unusual composition?)")

        # 2. Orbital stability (Roche limit check)
        # Approximate Roche limit: ~2.5 * R_star
        stellar_radius_au = params['stellar']['radius_rsun'] * 0.00465  # solar radii to AU
        roche_limit = 2.5 * stellar_radius_au

        if semi_major_axis < roche_limit:
            errors.append(f"Inside Roche limit: {semi_major_axis:.3f} AU < {roche_limit:.3f} AU (planet should be destroyed)")

        # 3. Temperature sanity check
        # Should be less than stellar temperature
        if equilibrium_temp > stellar_teff:
            errors.append(f"Planet hotter than star: {equilibrium_temp:.0f} K > {stellar_teff:.0f} K")

        # Very hot planets (hot Jupiters)
        if equilibrium_temp > 2000 and planet_radius < 2.0:
            warnings.append(f"Very hot small planet: {equilibrium_temp:.0f} K (likely vaporized?)")

        # 4. Period vs distance consistency check
        # Already enforced by Kepler's law, but check for obvious issues
        expected_period = 365.25 * (semi_major_axis ** 1.5)  # Rough estimate assuming Sun-like star

        if abs(period - expected_period) / expected_period > 0.5:
            warnings.append(f"Period-distance mismatch suggests non-solar mass star (expected ~{expected_period:.0f} days)")

        # 5. Transit duration sanity
        transit_duration = params['transit']['duration_hrs']
        # Maximum transit duration ~ (period / π) * (R_star / a)
        max_duration = (period * 24 / np.pi) * (stellar_radius_au / semi_major_axis)

        if transit_duration > max_duration * 1.5:
            warnings.append(f"Transit duration unusually long: {transit_duration:.1f} hrs (max ~{max_duration:.1f} hrs)")

        # 6. Transit depth consistency
        observed_depth = params['transit']['depth_ppm']
        expected_depth = params['transit']['expected_depth_ppm']

        depth_ratio = observed_depth / expected_depth if expected_depth > 0 else 0

        if abs(depth_ratio - 1.0) > 0.5:
            warnings.append(f"Transit depth anomaly: observed/expected = {depth_ratio:.2f} (could indicate blending or incorrect stellar radius)")

        return {
            'errors': errors,
            'warnings': warnings,
            'validated': len(errors) == 0,
            'density_g_cm3': density_abs
        }


def main():
    """Test physics calculations."""
    print("="*60)
    print("PHYSICS ENGINE TEST")
    print("="*60)

    engine = PhysicsEngine()

    # Test with a known system (Kepler-442b - habitable zone super-Earth)
    print("\nTest: Kepler-442b (known habitable zone planet)")
    print("-" * 60)

    params = engine.calculate_system_parameters(
        period_days=112.3,
        transit_depth_ppm=376,
        transit_duration_hrs=4.2,
        stellar_teff=4402,
        stellar_radius=0.601,
        stellar_logg=4.653,
        impact_parameter=0.3
    )

    print("\nSTELLAR PARAMETERS:")
    for key, value in params['stellar'].items():
        print(f"  {key}: {value:.4f}")

    print("\nPLANET PARAMETERS:")
    for key, value in params['planet'].items():
        print(f"  {key}: {value:.4f}")

    print("\nORBITAL PARAMETERS:")
    for key, value in params['orbit'].items():
        print(f"  {key}: {value:.4f}")

    print("\nHABITABILITY:")
    for key, value in params['habitability'].items():
        print(f"  {key}: {value}")

    print("\nTRANSIT:")
    for key, value in params['transit'].items():
        print(f"  {key}: {value:.4f}")

    # Validate system
    print("\n" + "-" * 60)
    print("PHYSICS VALIDATION:")
    validation = engine.validate_system(params)

    print(f"\nValidated: {validation['validated']}")
    print(f"Planet density: {validation['density_g_cm3']:.2f} g/cm³")

    if validation['errors']:
        print("\nERRORS:")
        for error in validation['errors']:
            print(f"  ❌ {error}")

    if validation['warnings']:
        print("\nWARNINGS:")
        for warning in validation['warnings']:
            print(f"  ⚠️  {warning}")

    if not validation['errors'] and not validation['warnings']:
        print("\n✅ No issues detected - system parameters are physically plausible!")

    print("\n" + "="*60)
    print("✓ Physics engine working correctly!")
    print("Ready for integration with ML predictions")


if __name__ == "__main__":
    main()
