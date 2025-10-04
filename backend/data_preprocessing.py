"""
Data quality control, filtering, and train/val/test split creation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict
import json


class DataPreprocessor:
    """Handle data quality control and dataset splitting."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def clean_koi_data(
        self,
        koi_df: pd.DataFrame,
        require_stellar_params: bool = True,
        require_planet_params: bool = True
    ) -> pd.DataFrame:
        """
        Clean KOI data by removing entries with missing critical parameters.

        Args:
            koi_df: Raw KOI DataFrame
            require_stellar_params: Filter out rows missing stellar parameters
            require_planet_params: Filter out rows missing planet parameters

        Returns:
            Cleaned DataFrame
        """
        print("\n" + "="*60)
        print("DATA QUALITY CONTROL")
        print("="*60)

        df = koi_df.copy()
        initial_count = len(df)

        print(f"\nInitial entries: {initial_count}")

        # Critical stellar parameters
        stellar_params = ['koi_steff', 'koi_srad', 'koi_slogg']

        # Critical planet parameters
        planet_params = ['koi_period', 'koi_depth', 'koi_duration', 'koi_prad']

        # Filter stellar parameters
        if require_stellar_params:
            print("\n--- Filtering by Stellar Parameters ---")
            for param in stellar_params:
                before = len(df)
                df = df[df[param].notna()]
                removed = before - len(df)
                if removed > 0:
                    print(f"Removed {removed} entries missing {param}")

        # Filter planet parameters
        if require_planet_params:
            print("\n--- Filtering by Planet Parameters ---")
            for param in planet_params:
                before = len(df)
                df = df[df[param].notna()]
                removed = before - len(df)
                if removed > 0:
                    print(f"Removed {removed} entries missing {param}")

        # Remove unrealistic values
        print("\n--- Filtering Unrealistic Values ---")

        # Filter unrealistic periods (< 0.1 days or > 1000 days for ML purposes)
        before = len(df)
        df = df[(df['koi_period'] >= 0.1) & (df['koi_period'] <= 1000)]
        removed = before - len(df)
        if removed > 0:
            print(f"Removed {removed} entries with period outside [0.1, 1000] days")

        # Filter unrealistic planet radii (< 0.5 or > 30 Earth radii)
        before = len(df)
        df = df[(df['koi_prad'] >= 0.5) & (df['koi_prad'] <= 30)]
        removed = before - len(df)
        if removed > 0:
            print(f"Removed {removed} entries with radius outside [0.5, 30] Earth radii")

        # Filter unrealistic stellar temps (< 3000 or > 10000 K)
        before = len(df)
        df = df[(df['koi_steff'] >= 3000) & (df['koi_steff'] <= 10000)]
        removed = before - len(df)
        if removed > 0:
            print(f"Removed {removed} entries with stellar temp outside [3000, 10000] K")

        final_count = len(df)
        print(f"\nFinal entries: {final_count}")
        print(f"Removed total: {initial_count - final_count} ({100*(initial_count - final_count)/initial_count:.1f}%)")

        return df

    def create_splits(
        self,
        df: pd.DataFrame,
        train_size: float = 0.70,
        val_size: float = 0.15,
        test_size: float = 0.15,
        stratify_by: str = 'koi_disposition',
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create stratified train/validation/test splits.

        Args:
            df: Cleaned DataFrame
            train_size: Fraction for training
            val_size: Fraction for validation
            test_size: Fraction for testing
            stratify_by: Column to stratify by
            random_state: Random seed for reproducibility

        Returns:
            (train_df, val_df, test_df)
        """
        print("\n" + "="*60)
        print("CREATING TRAIN/VAL/TEST SPLITS")
        print("="*60)

        # Validate split sizes
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Split sizes must sum to 1.0"

        print(f"\nSplit ratios: {train_size:.0%} / {val_size:.0%} / {test_size:.0%}")
        print(f"Stratifying by: {stratify_by}")

        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df,
            train_size=train_size,
            stratify=df[stratify_by],
            random_state=random_state
        )

        # Second split: val vs test
        val_ratio = val_size / (val_size + test_size)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_ratio,
            stratify=temp_df[stratify_by],
            random_state=random_state
        )

        print(f"\nTrain set: {len(train_df)} samples")
        print(f"Validation set: {len(val_df)} samples")
        print(f"Test set: {len(test_df)} samples")

        # Show disposition distribution
        print("\n--- Disposition Distribution ---")
        print("\nTrain:")
        print(train_df[stratify_by].value_counts())
        print("\nValidation:")
        print(val_df[stratify_by].value_counts())
        print("\nTest:")
        print(test_df[stratify_by].value_counts())

        return train_df, val_df, test_df

    def save_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        prefix: str = "koi"
    ):
        """
        Save train/val/test splits to CSV.

        Args:
            train_df: Training data
            val_df: Validation data
            test_df: Test data
            prefix: Filename prefix
        """
        print("\n--- Saving Splits ---")

        train_path = self.processed_dir / f"{prefix}_train.csv"
        val_path = self.processed_dir / f"{prefix}_val.csv"
        test_path = self.processed_dir / f"{prefix}_test.csv"

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"Train: {train_path}")
        print(f"Val: {val_path}")
        print(f"Test: {test_path}")

    def generate_quality_report(
        self,
        original_df: pd.DataFrame,
        cleaned_df: pd.DataFrame,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Dict:
        """
        Generate data quality report.

        Args:
            original_df: Original data
            cleaned_df: After quality control
            train_df: Training split
            val_df: Validation split
            test_df: Test split

        Returns:
            Report dictionary
        """
        report = {
            'data_quality': {
                'original_count': len(original_df),
                'cleaned_count': len(cleaned_df),
                'removed_count': len(original_df) - len(cleaned_df),
                'retention_rate': len(cleaned_df) / len(original_df)
            },
            'splits': {
                'train': len(train_df),
                'val': len(val_df),
                'test': len(test_df)
            },
            'disposition_distribution': {
                'train': train_df['koi_disposition'].value_counts().to_dict(),
                'val': val_df['koi_disposition'].value_counts().to_dict(),
                'test': test_df['koi_disposition'].value_counts().to_dict()
            },
            'missing_data_summary': {
                col: int(cleaned_df[col].isna().sum())
                for col in cleaned_df.columns
                if cleaned_df[col].isna().sum() > 0
            }
        }

        report_path = self.processed_dir / "data_quality_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\n✓ Quality report saved to: {report_path}")

        return report


def main():
    """Run data preprocessing pipeline."""
    from data_loader import ExoplanetDataLoader

    # Load data
    loader = ExoplanetDataLoader()
    koi_df = loader.load_koi()

    # Preprocess
    preprocessor = DataPreprocessor()

    # Clean data
    cleaned_df = preprocessor.clean_koi_data(
        koi_df,
        require_stellar_params=True,
        require_planet_params=True
    )

    # Create splits
    train_df, val_df, test_df = preprocessor.create_splits(
        cleaned_df,
        train_size=0.70,
        val_size=0.15,
        test_size=0.15,
        stratify_by='koi_disposition',
        random_state=42
    )

    # Save splits
    preprocessor.save_splits(train_df, val_df, test_df, prefix="koi")

    # Generate report
    report = preprocessor.generate_quality_report(
        koi_df, cleaned_df, train_df, val_df, test_df
    )

    print("\n" + "="*60)
    print("DATA PREPROCESSING COMPLETE")
    print("="*60)
    print(f"\nClean data: {len(cleaned_df)} samples")
    print(f"Train/Val/Test: {len(train_df)}/{len(val_df)}/{len(test_df)}")
    print(f"\nData ready for Phase 2: Feature Engineering")


if __name__ == "__main__":
    main()
