"""
Download representative light curves from Kepler and TESS for training and demo.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import lightkurve as lk
from typing import List, Dict
import json


class LightCurveDownloader:
    """Download and save representative light curves for the project."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.lc_dir = self.data_dir / "lightcurves"
        self.lc_dir.mkdir(parents=True, exist_ok=True)

    def select_targets(self, koi_df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """
        Select diverse set of targets for download.

        Args:
            koi_df: KOI DataFrame with disposition and parameters

        Returns:
            Dictionary with categories and target info
        """
        targets = {
            'confirmed_planets': [],
            'false_positives': [],
            'candidates': [],
            'multi_planet_systems': []
        }

        # Select confirmed planets (diverse sizes)
        confirmed = koi_df[koi_df['koi_disposition'] == 'CONFIRMED'].copy()

        if len(confirmed) > 0 and 'koi_prad' in confirmed.columns:
            # Small planets (< 1.5 Earth radii)
            small = confirmed[confirmed['koi_prad'] < 1.5].nsmallest(5, 'koi_prad')
            for _, row in small.iterrows():
                targets['confirmed_planets'].append({
                    'kepid': row['kepid'],
                    'koi_name': row['kepoi_name'],
                    'kepler_name': row.get('kepler_name', ''),
                    'radius': row['koi_prad'],
                    'period': row['koi_period'],
                    'category': 'rocky'
                })

            # Super-Earths (1.5-2.5 Earth radii)
            super_earths = confirmed[
                (confirmed['koi_prad'] >= 1.5) & (confirmed['koi_prad'] < 2.5)
            ].sample(min(5, len(confirmed)), random_state=42)
            for _, row in super_earths.iterrows():
                targets['confirmed_planets'].append({
                    'kepid': row['kepid'],
                    'koi_name': row['kepoi_name'],
                    'kepler_name': row.get('kepler_name', ''),
                    'radius': row['koi_prad'],
                    'period': row['koi_period'],
                    'category': 'super_earth'
                })

            # Mini-Neptunes (2.5-4 Earth radii)
            mini_neptunes = confirmed[
                (confirmed['koi_prad'] >= 2.5) & (confirmed['koi_prad'] < 4)
            ].sample(min(5, len(confirmed)), random_state=42)
            for _, row in mini_neptunes.iterrows():
                targets['confirmed_planets'].append({
                    'kepid': row['kepid'],
                    'koi_name': row['kepoi_name'],
                    'kepler_name': row.get('kepler_name', ''),
                    'radius': row['koi_prad'],
                    'period': row['koi_period'],
                    'category': 'mini_neptune'
                })

            # Gas giants (> 4 Earth radii)
            gas_giants = confirmed[confirmed['koi_prad'] >= 4].nlargest(5, 'koi_prad')
            for _, row in gas_giants.iterrows():
                targets['confirmed_planets'].append({
                    'kepid': row['kepid'],
                    'koi_name': row['kepoi_name'],
                    'kepler_name': row.get('kepler_name', ''),
                    'radius': row['koi_prad'],
                    'period': row['koi_period'],
                    'category': 'gas_giant'
                })

        # Select false positives
        fps = koi_df[koi_df['koi_disposition'] == 'FALSE POSITIVE'].sample(
            min(20, len(koi_df[koi_df['koi_disposition'] == 'FALSE POSITIVE'])),
            random_state=42
        )
        for _, row in fps.iterrows():
            targets['false_positives'].append({
                'kepid': row['kepid'],
                'koi_name': row['kepoi_name'],
                'period': row['koi_period']
            })

        # Select candidates
        cands = koi_df[koi_df['koi_disposition'] == 'CANDIDATE'].sample(
            min(20, len(koi_df[koi_df['koi_disposition'] == 'CANDIDATE'])),
            random_state=42
        )
        for _, row in cands.iterrows():
            targets['candidates'].append({
                'kepid': row['kepid'],
                'koi_name': row['kepoi_name'],
                'period': row['koi_period']
            })

        # Add known multi-planet systems
        multi_planet_systems = [
            {'name': 'Kepler-90', 'kepid': 11442793},  # 8 planets
            {'name': 'Kepler-11', 'kepid': 6541920},   # 6 planets
            {'name': 'Kepler-20', 'kepid': 6850504},   # 6 planets
            {'name': 'Kepler-80', 'kepid': 4852528},   # 6 planets
            {'name': 'Kepler-88', 'kepid': 5446285},   # 3 planets
        ]

        for system in multi_planet_systems:
            targets['multi_planet_systems'].append(system)

        return targets

    def download_lightcurve(self, kepid: int, output_name: str) -> bool:
        """
        Download and save a single light curve.

        Args:
            kepid: Kepler ID
            output_name: Filename to save (without extension)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Search for light curves
            search_result = lk.search_lightcurve(f'KIC {kepid}', mission='Kepler')

            if len(search_result) == 0:
                print(f"  ⚠️  No light curve found for KIC {kepid}")
                return False

            # Download all quarters and stitch
            lc_collection = search_result.download_all()

            if lc_collection is None or len(lc_collection) == 0:
                print(f"  ⚠️  Failed to download for KIC {kepid}")
                return False

            # Stitch quarters together
            lc = lc_collection.stitch()

            # Save as CSV for easy loading
            output_file = self.lc_dir / f"{output_name}.csv"

            # Extract time and flux
            time = lc.time.value  # Barycentric Kepler Julian Date
            flux = lc.flux.value
            flux_err = lc.flux_err.value if hasattr(lc.flux_err, 'value') else None

            # Create DataFrame
            lc_data = pd.DataFrame({
                'time': time,
                'flux': flux,
                'flux_err': flux_err if flux_err is not None else np.nan
            })

            # Remove NaN values
            lc_data = lc_data.dropna(subset=['time', 'flux'])

            lc_data.to_csv(output_file, index=False)
            print(f"  ✓ Saved {output_name}.csv ({len(lc_data)} points)")

            return True

        except Exception as e:
            print(f"  ✗ Error downloading KIC {kepid}: {e}")
            return False

    def download_all(self, targets: Dict[str, List[Dict]], max_per_category: int = None):
        """
        Download all selected targets.

        Args:
            targets: Dictionary from select_targets()
            max_per_category: Limit downloads per category (for testing)
        """
        print("\n" + "="*60)
        print("DOWNLOADING LIGHT CURVES")
        print("="*60)

        stats = {
            'total_attempted': 0,
            'successful': 0,
            'failed': 0
        }

        # Download confirmed planets
        print("\n--- Confirmed Planets ---")
        confirmed_targets = targets['confirmed_planets']
        if max_per_category:
            confirmed_targets = confirmed_targets[:max_per_category]

        for target in confirmed_targets:
            kepid = target['kepid']
            koi_name = target['koi_name']
            kepler_name = target.get('kepler_name', '')
            category = target.get('category', 'unknown')

            name = kepler_name if kepler_name else koi_name
            print(f"\n{name} (KIC {kepid}) - {category}")

            stats['total_attempted'] += 1
            output_name = f"confirmed_{kepid}_{koi_name.replace('.', '_')}"

            if self.download_lightcurve(kepid, output_name):
                stats['successful'] += 1
            else:
                stats['failed'] += 1

        # Download false positives
        print("\n--- False Positives ---")
        fp_targets = targets['false_positives']
        if max_per_category:
            fp_targets = fp_targets[:max_per_category]

        for target in fp_targets:
            kepid = target['kepid']
            koi_name = target['koi_name']

            print(f"\n{koi_name} (KIC {kepid})")

            stats['total_attempted'] += 1
            output_name = f"fp_{kepid}_{koi_name.replace('.', '_')}"

            if self.download_lightcurve(kepid, output_name):
                stats['successful'] += 1
            else:
                stats['failed'] += 1

        # Download candidates
        print("\n--- Candidates ---")
        cand_targets = targets['candidates']
        if max_per_category:
            cand_targets = cand_targets[:max_per_category]

        for target in cand_targets:
            kepid = target['kepid']
            koi_name = target['koi_name']

            print(f"\n{koi_name} (KIC {kepid})")

            stats['total_attempted'] += 1
            output_name = f"candidate_{kepid}_{koi_name.replace('.', '_')}"

            if self.download_lightcurve(kepid, output_name):
                stats['successful'] += 1
            else:
                stats['failed'] += 1

        # Download multi-planet systems
        print("\n--- Multi-Planet Systems ---")
        multi_targets = targets['multi_planet_systems']
        if max_per_category:
            multi_targets = multi_targets[:max_per_category]

        for target in multi_targets:
            kepid = target['kepid']
            name = target['name']

            print(f"\n{name} (KIC {kepid})")

            stats['total_attempted'] += 1
            output_name = f"system_{kepid}_{name.replace('-', '_')}"

            if self.download_lightcurve(kepid, output_name):
                stats['successful'] += 1
            else:
                stats['failed'] += 1

        # Save target metadata
        metadata_file = self.lc_dir / "target_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(targets, f, indent=2)

        print("\n" + "="*60)
        print("DOWNLOAD COMPLETE")
        print("="*60)
        print(f"Total attempted: {stats['total_attempted']}")
        print(f"Successful: {stats['successful']}")
        print(f"Failed: {stats['failed']}")
        print(f"Success rate: {100*stats['successful']/stats['total_attempted']:.1f}%")
        print(f"\nLight curves saved to: {self.lc_dir}")
        print(f"Metadata saved to: {metadata_file}")


def main():
    """Download representative light curve sample."""
    from data_loader import ExoplanetDataLoader

    # Load KOI data
    loader = ExoplanetDataLoader()
    koi_df = loader.load_koi()

    # Select targets
    downloader = LightCurveDownloader()
    targets = downloader.select_targets(koi_df)

    print("\n--- Target Selection Summary ---")
    print(f"Confirmed planets: {len(targets['confirmed_planets'])}")
    print(f"False positives: {len(targets['false_positives'])}")
    print(f"Candidates: {len(targets['candidates'])}")
    print(f"Multi-planet systems: {len(targets['multi_planet_systems'])}")

    # Download (limit to 2 per category for testing)
    print("\n*** TESTING MODE: Downloading 2 per category ***")
    print("*** Remove max_per_category parameter to download all ***\n")

    downloader.download_all(targets, max_per_category=2)

    print("\n✓ Sample download complete!")
    print("To download all targets, edit this script and remove the max_per_category parameter.")


if __name__ == "__main__":
    main()
