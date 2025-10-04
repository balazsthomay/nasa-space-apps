"""
Feature engineering for exoplanet detection ML models.
Extracts ~50 features from KOI catalog data for training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Extract features from KOI catalog for ML training."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"

    def extract_transit_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract transit-related features.

        Features:
        - Transit depth (ppm)
        - Transit duration (hours)
        - Orbital period (days)
        - Time of first transit
        - Impact parameter
        - Signal-to-noise ratio

        Args:
            df: KOI DataFrame

        Returns:
            DataFrame with transit features
        """
        print("\n--- Extracting Transit Features ---")

        features = pd.DataFrame()

        # Direct transit parameters
        features['transit_depth_ppm'] = df['koi_depth']
        features['transit_duration_hrs'] = df['koi_duration']
        features['orbital_period_days'] = df['koi_period']
        features['impact_parameter'] = df['koi_impact']
        features['transit_snr'] = df['koi_model_snr']

        # Log-scale versions for skewed distributions
        features['log_period'] = np.log10(df['koi_period'])
        features['log_depth'] = np.log10(df['koi_depth'])

        # Transit duration ratio (normalized by period)
        features['duration_period_ratio'] = df['koi_duration'] / (df['koi_period'] * 24)

        # Transit shape: ingress/egress timing estimate
        # Assuming circular orbit, estimate from impact parameter
        features['transit_shape_factor'] = df['koi_impact'] * df['koi_duration']

        print(f"Extracted {len([c for c in features.columns if not c.startswith('_')])} transit features")

        return features

    def extract_stellar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract stellar parameter features.

        Features from host star:
        - Effective temperature
        - Stellar radius
        - Surface gravity
        - Stellar mass (derived if available)

        Args:
            df: KOI DataFrame

        Returns:
            DataFrame with stellar features
        """
        print("\n--- Extracting Stellar Features ---")

        features = pd.DataFrame()

        # Direct stellar parameters
        features['stellar_teff_K'] = df['koi_steff']
        features['stellar_radius_rsun'] = df['koi_srad']
        features['stellar_logg'] = df['koi_slogg']

        # Normalized temperature (relative to Sun = 5778K)
        features['stellar_teff_normalized'] = df['koi_steff'] / 5778.0

        # Stellar density approximation from log g and radius
        # ρ ∝ g / R²
        features['stellar_density_proxy'] = (10 ** df['koi_slogg']) / (df['koi_srad'] ** 2)

        # Stellar luminosity estimate (L ∝ R² T⁴)
        features['stellar_luminosity_proxy'] = (df['koi_srad'] ** 2) * ((df['koi_steff'] / 5778) ** 4)

        print(f"Extracted {len([c for c in features.columns if not c.startswith('_')])} stellar features")

        return features

    def extract_planetary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract planetary parameter features.

        Features:
        - Planet radius
        - Equilibrium temperature
        - Insolation flux
        - Derived properties

        Args:
            df: KOI DataFrame

        Returns:
            DataFrame with planetary features
        """
        print("\n--- Extracting Planetary Features ---")

        features = pd.DataFrame()

        # Direct planetary parameters
        features['planet_radius_rearth'] = df['koi_prad']
        features['planet_teq_K'] = df['koi_teq']
        features['insolation_flux_earth'] = df['koi_insol']

        # Log versions
        features['log_planet_radius'] = np.log10(df['koi_prad'])
        features['log_insolation'] = np.log10(df['koi_insol'])

        # Planet size categories (feature engineering)
        # Rocky: < 1.5 R⊕, Super-Earth: 1.5-2, Mini-Neptune: 2-4, Gas Giant: > 4
        features['is_rocky_size'] = (df['koi_prad'] < 1.5).astype(int)
        features['is_super_earth_size'] = ((df['koi_prad'] >= 1.5) & (df['koi_prad'] < 2.0)).astype(int)
        features['is_mini_neptune_size'] = ((df['koi_prad'] >= 2.0) & (df['koi_prad'] < 4.0)).astype(int)
        features['is_gas_giant_size'] = (df['koi_prad'] >= 4.0).astype(int)

        # Temperature categories
        features['is_hot'] = (df['koi_teq'] > 1000).astype(int)  # Hot Jupiter territory
        features['is_warm'] = ((df['koi_teq'] >= 300) & (df['koi_teq'] <= 1000)).astype(int)
        features['is_temperate'] = ((df['koi_teq'] >= 200) & (df['koi_teq'] < 300)).astype(int)
        features['is_cold'] = (df['koi_teq'] < 200).astype(int)

        print(f"Extracted {len([c for c in features.columns if not c.startswith('_')])} planetary features")

        return features

    def derive_physical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive physics-based features from combinations.

        Features:
        - Semi-major axis (from period + stellar mass)
        - Orbital velocity
        - Planet density proxy
        - Habitability indicators

        Args:
            df: KOI DataFrame

        Returns:
            DataFrame with derived features
        """
        print("\n--- Deriving Physical Features ---")

        features = pd.DataFrame()

        # Semi-major axis using Kepler's Third Law
        # a³ = (G M_star T²) / (4π²)
        # Simplified: a (AU) ≈ (T² M_star)^(1/3) where T in years, M_star in M_sun

        # Estimate stellar mass from radius and log g
        # M ∝ R² * g
        stellar_mass_proxy = (df['koi_srad'] ** 2) * (10 ** df['koi_slogg']) / (10 ** 4.44)  # normalized to Sun

        period_years = df['koi_period'] / 365.25
        features['semi_major_axis_au'] = (period_years ** 2 * stellar_mass_proxy) ** (1/3)

        # Orbital velocity (v = 2π a / T)
        features['orbital_velocity_kms'] = (2 * np.pi * features['semi_major_axis_au'] * 1.496e8) / (df['koi_period'] * 86400)

        # Planet density proxy (assuming from radius and mass estimate)
        # For now, use radius as proxy (smaller/denser vs larger/less dense)
        features['density_proxy'] = 1 / (df['koi_prad'] ** 3)  # inversely proportional to volume

        # Habitable zone estimate (simplified)
        # HZ inner ≈ sqrt(L_star / 1.1), HZ outer ≈ sqrt(L_star / 0.53)
        stellar_luminosity = (df['koi_srad'] ** 2) * ((df['koi_steff'] / 5778) ** 4)
        hz_inner = np.sqrt(stellar_luminosity / 1.1)
        hz_outer = np.sqrt(stellar_luminosity / 0.53)

        features['in_habitable_zone'] = (
            (features['semi_major_axis_au'] >= hz_inner) &
            (features['semi_major_axis_au'] <= hz_outer)
        ).astype(int)

        # Distance from HZ center (normalized)
        hz_center = (hz_inner + hz_outer) / 2
        features['hz_distance_normalized'] = np.abs(features['semi_major_axis_au'] - hz_center) / hz_center

        # Transit probability (geometric)
        # P_transit = R_star / a
        features['transit_probability'] = (df['koi_srad'] * 0.00465) / features['semi_major_axis_au']  # R_sun to AU

        # Ratio of planet to star radius
        features['radius_ratio'] = df['koi_prad'] / (df['koi_srad'] * 109.2)  # Convert stellar radius to Earth radii

        # Expected transit depth from geometry (for validation)
        features['expected_transit_depth_ppm'] = (features['radius_ratio'] ** 2) * 1e6

        # Depth anomaly: difference between observed and expected
        features['depth_anomaly'] = np.abs(df['koi_depth'] - features['expected_transit_depth_ppm']) / features['expected_transit_depth_ppm']

        print(f"Extracted {len([c for c in features.columns if not c.startswith('_')])} derived features")

        return features

    def extract_error_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from measurement uncertainties.

        Features:
        - Uncertainty in key parameters
        - Signal quality indicators

        Args:
            df: KOI DataFrame

        Returns:
            DataFrame with error-based features
        """
        print("\n--- Extracting Error/Uncertainty Features ---")

        features = pd.DataFrame()

        # Relative uncertainties (fractional errors)
        if 'koi_period_err1' in df.columns:
            features['period_uncertainty'] = np.abs(df['koi_period_err1']) / df['koi_period']

        if 'koi_prad_err1' in df.columns:
            features['radius_uncertainty'] = np.abs(df['koi_prad_err1']) / df['koi_prad']

        if 'koi_depth_err1' in df.columns:
            features['depth_uncertainty'] = np.abs(df['koi_depth_err1']) / df['koi_depth']

        # Overall measurement quality proxy
        features['measurement_quality'] = 1 / (1 + features.get('period_uncertainty', 0) +
                                                features.get('radius_uncertainty', 0) +
                                                features.get('depth_uncertainty', 0))

        print(f"Extracted {len([c for c in features.columns if not c.startswith('_')])} error features")

        return features

    def extract_false_positive_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract false positive flag features.

        These are indicators from the catalog about potential false positives.

        Args:
            df: KOI DataFrame

        Returns:
            DataFrame with FP flag features
        """
        print("\n--- Extracting False Positive Flags ---")

        features = pd.DataFrame()

        # False positive flags from KOI catalog
        if 'koi_fpflag_nt' in df.columns:
            features['fp_flag_not_transit_like'] = df['koi_fpflag_nt']

        if 'koi_fpflag_ss' in df.columns:
            features['fp_flag_stellar_eclipse'] = df['koi_fpflag_ss']

        if 'koi_fpflag_co' in df.columns:
            features['fp_flag_centroid_offset'] = df['koi_fpflag_co']

        if 'koi_fpflag_ec' in df.columns:
            features['fp_flag_ephemeris_match'] = df['koi_fpflag_ec']

        # Total FP flags
        fp_cols = [c for c in features.columns if 'fp_flag' in c]
        if fp_cols:
            features['total_fp_flags'] = features[fp_cols].sum(axis=1)

        print(f"Extracted {len([c for c in features.columns if not c.startswith('_')])} FP flag features")

        return features

    def create_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create complete feature matrix from KOI data.

        Args:
            df: KOI DataFrame

        Returns:
            Complete feature matrix
        """
        print("\n" + "="*60)
        print("FEATURE ENGINEERING")
        print("="*60)

        # Extract all feature groups
        transit_features = self.extract_transit_features(df)
        stellar_features = self.extract_stellar_features(df)
        planetary_features = self.extract_planetary_features(df)
        derived_features = self.derive_physical_features(df)
        error_features = self.extract_error_features(df)
        fp_flags = self.extract_false_positive_flags(df)

        # Combine all features
        feature_matrix = pd.concat([
            transit_features,
            stellar_features,
            planetary_features,
            derived_features,
            error_features,
            fp_flags
        ], axis=1)

        # Add target variable
        feature_matrix['target'] = (df['koi_disposition'] == 'CONFIRMED').astype(int)
        feature_matrix['disposition'] = df['koi_disposition']

        # Add identifiers for reference
        feature_matrix['kepid'] = df['kepid']
        feature_matrix['koi_name'] = df['kepoi_name']

        print("\n" + "="*60)
        print("FEATURE MATRIX SUMMARY")
        print("="*60)
        print(f"Total samples: {len(feature_matrix)}")
        print(f"Total features: {len(feature_matrix.columns) - 4}")  # Exclude target, disposition, kepid, koi_name
        print(f"Confirmed planets: {feature_matrix['target'].sum()}")
        print(f"Non-planets: {(1 - feature_matrix['target']).sum()}")

        # Check for missing values
        missing = feature_matrix.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            print(f"\nFeatures with missing values:")
            for col, count in missing.items():
                print(f"  {col}: {count} ({100*count/len(feature_matrix):.1f}%)")

        return feature_matrix

    def save_features(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        prefix: str = "features"
    ):
        """
        Save feature matrices to CSV.

        Args:
            train_df: Training features
            val_df: Validation features
            test_df: Test features
            prefix: Filename prefix
        """
        print("\n--- Saving Feature Matrices ---")

        train_path = self.processed_dir / f"{prefix}_train.csv"
        val_path = self.processed_dir / f"{prefix}_val.csv"
        test_path = self.processed_dir / f"{prefix}_test.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"Train: {train_path}")
        print(f"Val: {val_path}")
        print(f"Test: {test_path}")


def main():
    """Run feature engineering pipeline."""

    # Load preprocessed splits
    preprocessor_dir = Path("data/processed")

    print("Loading preprocessed data splits...")
    train_df = pd.read_csv(preprocessor_dir / "koi_train.csv")
    val_df = pd.read_csv(preprocessor_dir / "koi_val.csv")
    test_df = pd.read_csv(preprocessor_dir / "koi_test.csv")

    # Create feature engineer
    engineer = FeatureEngineer()

    # Extract features for each split
    print("\n" + "="*60)
    print("PROCESSING TRAINING SET")
    print("="*60)
    train_features = engineer.create_feature_matrix(train_df)

    print("\n" + "="*60)
    print("PROCESSING VALIDATION SET")
    print("="*60)
    val_features = engineer.create_feature_matrix(val_df)

    print("\n" + "="*60)
    print("PROCESSING TEST SET")
    print("="*60)
    test_features = engineer.create_feature_matrix(test_df)

    # Save feature matrices
    engineer.save_features(train_features, val_features, test_features)

    print("\n" + "="*60)
    print("FEATURE ENGINEERING COMPLETE")
    print("="*60)
    print(f"\nFeature matrices saved with {len(train_features.columns) - 4} features")
    print("Ready for Phase 3: Model Training")


if __name__ == "__main__":
    main()
