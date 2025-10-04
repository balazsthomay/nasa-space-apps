"""
Data loader for Kepler and TESS exoplanet catalogs.
Handles loading, initial exploration, and validation of NASA Exoplanet Archive data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple


class ExoplanetDataLoader:
    """Load and explore KOI and TOI datasets from NASA Exoplanet Archive."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.koi_df = None
        self.toi_df = None

    def load_koi(self, filepath: str = None) -> pd.DataFrame:
        """
        Load Kepler Objects of Interest (KOI) catalog.

        Args:
            filepath: Path to KOI CSV file. If None, auto-detect in data_dir.

        Returns:
            DataFrame with KOI data
        """
        if filepath is None:
            # Auto-detect KOI file
            koi_files = list(self.data_dir.glob("cumulative*.csv"))
            if not koi_files:
                raise FileNotFoundError(f"No KOI file found in {self.data_dir}")
            filepath = koi_files[0]

        print(f"Loading KOI data from {filepath}...")

        # Read CSV, skipping comment lines
        self.koi_df = pd.read_csv(filepath, comment='#')

        print(f"Loaded {len(self.koi_df)} KOI entries")
        return self.koi_df

    def load_toi(self, filepath: str = None) -> pd.DataFrame:
        """
        Load TESS Objects of Interest (TOI) catalog.

        Args:
            filepath: Path to TOI CSV file. If None, auto-detect in data_dir.

        Returns:
            DataFrame with TOI data
        """
        if filepath is None:
            # Auto-detect TOI file
            toi_files = list(self.data_dir.glob("TOI*.csv"))
            if not toi_files:
                raise FileNotFoundError(f"No TOI file found in {self.data_dir}")
            filepath = toi_files[0]

        print(f"Loading TOI data from {filepath}...")

        # Read CSV, skipping comment lines
        self.toi_df = pd.read_csv(filepath, comment='#')

        print(f"Loaded {len(self.toi_df)} TOI entries")
        return self.toi_df

    def explore_koi(self) -> Dict:
        """
        Explore KOI dataset and return summary statistics.

        Returns:
            Dictionary with exploration results
        """
        if self.koi_df is None:
            raise ValueError("KOI data not loaded. Call load_koi() first.")

        print("\n" + "="*60)
        print("KEPLER OBJECTS OF INTEREST (KOI) EXPLORATION")
        print("="*60)

        # Disposition breakdown
        print("\n--- Disposition Breakdown ---")
        disposition_counts = self.koi_df['koi_disposition'].value_counts()
        print(disposition_counts)

        # Key statistics
        stats = {
            'total': len(self.koi_df),
            'confirmed': len(self.koi_df[self.koi_df['koi_disposition'] == 'CONFIRMED']),
            'candidate': len(self.koi_df[self.koi_df['koi_disposition'] == 'CANDIDATE']),
            'false_positive': len(self.koi_df[self.koi_df['koi_disposition'] == 'FALSE POSITIVE']),
        }

        print(f"\nConfirmed Planets: {stats['confirmed']}")
        print(f"Candidates: {stats['candidate']}")
        print(f"False Positives: {stats['false_positive']}")

        # Missing data analysis
        print("\n--- Missing Data Analysis (Key Features) ---")
        key_features = [
            'koi_period', 'koi_depth', 'koi_duration', 'koi_prad',
            'koi_teq', 'koi_steff', 'koi_srad', 'koi_slogg'
        ]

        missing_counts = {}
        for feature in key_features:
            if feature in self.koi_df.columns:
                missing = self.koi_df[feature].isna().sum()
                pct = (missing / len(self.koi_df)) * 100
                missing_counts[feature] = missing
                print(f"{feature}: {missing} ({pct:.1f}%)")

        stats['missing_data'] = missing_counts

        # Orbital period distribution
        print("\n--- Orbital Period Statistics ---")
        if 'koi_period' in self.koi_df.columns:
            period_stats = self.koi_df['koi_period'].describe()
            print(f"Min: {period_stats['min']:.2f} days")
            print(f"Median: {period_stats['50%']:.2f} days")
            print(f"Max: {period_stats['max']:.2f} days")
            stats['period_stats'] = period_stats.to_dict()

        # Planet radius distribution
        print("\n--- Planet Radius Statistics ---")
        if 'koi_prad' in self.koi_df.columns:
            radius_stats = self.koi_df['koi_prad'].describe()
            print(f"Min: {radius_stats['min']:.2f} Earth radii")
            print(f"Median: {radius_stats['50%']:.2f} Earth radii")
            print(f"Max: {radius_stats['max']:.2f} Earth radii")
            stats['radius_stats'] = radius_stats.to_dict()

        return stats

    def explore_toi(self) -> Dict:
        """
        Explore TOI dataset and return summary statistics.

        Returns:
            Dictionary with exploration results
        """
        if self.toi_df is None:
            raise ValueError("TOI data not loaded. Call load_toi() first.")

        print("\n" + "="*60)
        print("TESS OBJECTS OF INTEREST (TOI) EXPLORATION")
        print("="*60)

        # Disposition breakdown
        print("\n--- Disposition Breakdown ---")
        if 'tfopwg_disp' in self.toi_df.columns:
            disposition_counts = self.toi_df['tfopwg_disp'].value_counts()
            print(disposition_counts)

        stats = {
            'total': len(self.toi_df),
        }

        if 'tfopwg_disp' in self.toi_df.columns:
            stats['confirmed_pc'] = len(self.toi_df[self.toi_df['tfopwg_disp'] == 'PC'])  # Planet Candidate
            stats['confirmed_cp'] = len(self.toi_df[self.toi_df['tfopwg_disp'] == 'CP'])  # Confirmed Planet
            stats['false_positive'] = len(self.toi_df[self.toi_df['tfopwg_disp'] == 'FP'])
            stats['known_planet'] = len(self.toi_df[self.toi_df['tfopwg_disp'] == 'KP'])

            print(f"\nConfirmed Planets (CP): {stats['confirmed_cp']}")
            print(f"Planet Candidates (PC): {stats['confirmed_pc']}")
            print(f"Known Planets (KP): {stats['known_planet']}")
            print(f"False Positives (FP): {stats['false_positive']}")

        # Missing data analysis
        print("\n--- Missing Data Analysis (Key Features) ---")
        key_features = [
            'pl_orbper', 'pl_trandep', 'pl_trandurh', 'pl_rade',
            'pl_eqt', 'pl_insol'
        ]

        missing_counts = {}
        for feature in key_features:
            if feature in self.toi_df.columns:
                missing = self.toi_df[feature].isna().sum()
                pct = (missing / len(self.toi_df)) * 100
                missing_counts[feature] = missing
                print(f"{feature}: {missing} ({pct:.1f}%)")

        stats['missing_data'] = missing_counts

        return stats

    def get_confirmed_planets(self, catalog: str = 'koi') -> pd.DataFrame:
        """
        Get only confirmed planets from specified catalog.

        Args:
            catalog: 'koi' or 'toi'

        Returns:
            DataFrame with only confirmed planets
        """
        if catalog.lower() == 'koi':
            if self.koi_df is None:
                raise ValueError("KOI data not loaded")
            return self.koi_df[self.koi_df['koi_disposition'] == 'CONFIRMED'].copy()
        elif catalog.lower() == 'toi':
            if self.toi_df is None:
                raise ValueError("TOI data not loaded")
            return self.toi_df[self.toi_df['tfopwg_disp'].isin(['CP', 'PC'])].copy()
        else:
            raise ValueError("catalog must be 'koi' or 'toi'")

    def get_false_positives(self, catalog: str = 'koi') -> pd.DataFrame:
        """
        Get only false positives from specified catalog.

        Args:
            catalog: 'koi' or 'toi'

        Returns:
            DataFrame with only false positives
        """
        if catalog.lower() == 'koi':
            if self.koi_df is None:
                raise ValueError("KOI data not loaded")
            return self.koi_df[self.koi_df['koi_disposition'] == 'FALSE POSITIVE'].copy()
        elif catalog.lower() == 'toi':
            if self.toi_df is None:
                raise ValueError("TOI data not loaded")
            return self.toi_df[self.toi_df['tfopwg_disp'] == 'FP'].copy()
        else:
            raise ValueError("catalog must be 'koi' or 'toi'")

    def get_candidates(self, catalog: str = 'koi') -> pd.DataFrame:
        """
        Get only candidates (undecided) from specified catalog.

        Args:
            catalog: 'koi' or 'toi'

        Returns:
            DataFrame with only candidates
        """
        if catalog.lower() == 'koi':
            if self.koi_df is None:
                raise ValueError("KOI data not loaded")
            return self.koi_df[self.koi_df['koi_disposition'] == 'CANDIDATE'].copy()
        elif catalog.lower() == 'toi':
            if self.toi_df is None:
                raise ValueError("TOI data not loaded")
            # For TOI, candidates would be those without a disposition or with PC
            return self.toi_df[self.toi_df['tfopwg_disp'] == 'PC'].copy()
        else:
            raise ValueError("catalog must be 'koi' or 'toi'")


def main():
    """Run data exploration."""
    loader = ExoplanetDataLoader()

    # Load and explore KOI
    loader.load_koi()
    koi_stats = loader.explore_koi()

    # Load and explore TOI
    loader.load_toi()
    toi_stats = loader.explore_toi()

    print("\n" + "="*60)
    print("DATA LOADING COMPLETE")
    print("="*60)
    print(f"\nTotal KOI entries: {koi_stats['total']}")
    print(f"Total TOI entries: {toi_stats['total']}")
    print("\nReady for next phase: Gaia cross-match and light curve download")


if __name__ == "__main__":
    main()
