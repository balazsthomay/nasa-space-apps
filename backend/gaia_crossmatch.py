"""
Cross-match Kepler targets with Gaia DR3 for improved stellar parameters.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
from astropy import units as u
from typing import Optional
import time


class GaiaCrossmatcher:
    """Cross-match exoplanet targets with Gaia DR3 for updated stellar parameters."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)

    def crossmatch_koi(
        self,
        koi_df: pd.DataFrame,
        output_file: str = "koi_with_gaia.csv",
        max_separation: float = 2.0,
        sample_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Cross-match KOI catalog with Gaia DR3.

        Args:
            koi_df: DataFrame with KOI data (must have 'ra' and 'dec' columns)
            output_file: Where to save enriched data
            max_separation: Maximum separation in arcseconds for a match
            sample_size: If provided, only cross-match first N entries (for testing)

        Returns:
            DataFrame with Gaia parameters added
        """
        print("\n" + "="*60)
        print("GAIA DR3 CROSS-MATCH")
        print("="*60)

        # Check required columns
        if 'ra' not in koi_df.columns or 'dec' not in koi_df.columns:
            raise ValueError("KOI DataFrame must have 'ra' and 'dec' columns")

        # Work on a copy
        df = koi_df.copy()

        # Sample if requested
        if sample_size is not None:
            print(f"\nUsing sample of {sample_size} targets for testing...")
            df = df.head(sample_size).copy()

        print(f"\nCross-matching {len(df)} KOI targets with Gaia DR3...")
        print(f"Max separation: {max_separation} arcsec")

        # Initialize Gaia columns
        gaia_columns = [
            'gaia_source_id',
            'gaia_ra',
            'gaia_dec',
            'gaia_parallax',
            'gaia_parallax_error',
            'gaia_pmra',
            'gaia_pmdec',
            'gaia_phot_g_mean_mag',
            'gaia_phot_bp_mean_mag',
            'gaia_phot_rp_mean_mag',
            'gaia_teff_gspphot',
            'gaia_logg_gspphot',
            'gaia_radius_gspphot',
            'gaia_separation_arcsec'
        ]

        for col in gaia_columns:
            df[col] = np.nan

        # Process in batches to avoid timeout
        batch_size = 100
        total_batches = (len(df) + batch_size - 1) // batch_size

        matches = 0
        for i in range(0, len(df), batch_size):
            batch_num = i // batch_size + 1
            batch = df.iloc[i:i+batch_size]

            print(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch)} targets)...")

            for idx, row in batch.iterrows():
                try:
                    # Create coordinate
                    coord = SkyCoord(
                        ra=row['ra'] * u.degree,
                        dec=row['dec'] * u.degree,
                        frame='icrs'
                    )

                    # Query Gaia within radius
                    radius = u.Quantity(max_separation, u.arcsec)

                    # Build ADQL query for cone search
                    query = f"""
                    SELECT TOP 1
                        source_id, ra, dec, parallax, parallax_error,
                        pmra, pmdec,
                        phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
                        teff_gspphot, logg_gspphot, radius_gspphot,
                        DISTANCE(
                            POINT({row['ra']}, {row['dec']}),
                            POINT(ra, dec)
                        ) AS separation
                    FROM gaiadr3.gaia_source
                    WHERE 1=CONTAINS(
                        POINT({row['ra']}, {row['dec']}),
                        CIRCLE(ra, dec, {max_separation/3600.0})
                    )
                    ORDER BY separation ASC
                    """

                    job = Gaia.launch_job_async(query)
                    result = job.get_results()

                    if len(result) > 0:
                        gaia_match = result[0]

                        # Store Gaia data
                        df.at[idx, 'gaia_source_id'] = int(gaia_match['source_id'])
                        df.at[idx, 'gaia_ra'] = float(gaia_match['ra'])
                        df.at[idx, 'gaia_dec'] = float(gaia_match['dec'])
                        df.at[idx, 'gaia_parallax'] = float(gaia_match['parallax']) if not np.ma.is_masked(gaia_match['parallax']) else np.nan
                        df.at[idx, 'gaia_parallax_error'] = float(gaia_match['parallax_error']) if not np.ma.is_masked(gaia_match['parallax_error']) else np.nan
                        df.at[idx, 'gaia_pmra'] = float(gaia_match['pmra']) if not np.ma.is_masked(gaia_match['pmra']) else np.nan
                        df.at[idx, 'gaia_pmdec'] = float(gaia_match['pmdec']) if not np.ma.is_masked(gaia_match['pmdec']) else np.nan
                        df.at[idx, 'gaia_phot_g_mean_mag'] = float(gaia_match['phot_g_mean_mag']) if not np.ma.is_masked(gaia_match['phot_g_mean_mag']) else np.nan
                        df.at[idx, 'gaia_phot_bp_mean_mag'] = float(gaia_match['phot_bp_mean_mag']) if not np.ma.is_masked(gaia_match['phot_bp_mean_mag']) else np.nan
                        df.at[idx, 'gaia_phot_rp_mean_mag'] = float(gaia_match['phot_rp_mean_mag']) if not np.ma.is_masked(gaia_match['phot_rp_mean_mag']) else np.nan
                        df.at[idx, 'gaia_teff_gspphot'] = float(gaia_match['teff_gspphot']) if not np.ma.is_masked(gaia_match['teff_gspphot']) else np.nan
                        df.at[idx, 'gaia_logg_gspphot'] = float(gaia_match['logg_gspphot']) if not np.ma.is_masked(gaia_match['logg_gspphot']) else np.nan
                        df.at[idx, 'gaia_radius_gspphot'] = float(gaia_match['radius_gspphot']) if not np.ma.is_masked(gaia_match['radius_gspphot']) else np.nan
                        df.at[idx, 'gaia_separation_arcsec'] = float(gaia_match['separation']) * 3600  # Convert degrees to arcsec

                        matches += 1

                    # Small delay to avoid rate limiting
                    time.sleep(0.1)

                except Exception as e:
                    print(f"Warning: Failed to match KOI at index {idx}: {e}")
                    continue

            print(f"Matched {matches}/{i+len(batch)} so far ({100*matches/(i+len(batch)):.1f}%)")

        print(f"\n" + "="*60)
        print(f"CROSS-MATCH COMPLETE")
        print(f"="*60)
        print(f"Total targets: {len(df)}")
        print(f"Successful matches: {matches}")
        print(f"Match rate: {100*matches/len(df):.1f}%")

        # Save enriched data
        output_path = self.data_dir / "processed" / output_file
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)
        print(f"\nEnriched data saved to: {output_path}")

        # Show improvement statistics
        self._show_gaia_improvement(df)

        return df

    def _show_gaia_improvement(self, df: pd.DataFrame):
        """Show how Gaia data improves coverage."""
        print("\n--- Gaia Data Quality ---")

        # Temperature coverage
        kic_teff = df['koi_steff'].notna().sum()
        gaia_teff = df['gaia_teff_gspphot'].notna().sum()
        print(f"Effective Temperature (Teff):")
        print(f"  Original KIC: {kic_teff}/{len(df)} ({100*kic_teff/len(df):.1f}%)")
        print(f"  Gaia DR3: {gaia_teff}/{len(df)} ({100*gaia_teff/len(df):.1f}%)")

        # Radius coverage
        kic_rad = df['koi_srad'].notna().sum()
        gaia_rad = df['gaia_radius_gspphot'].notna().sum()
        print(f"\nStellar Radius:")
        print(f"  Original KIC: {kic_rad}/{len(df)} ({100*kic_rad/len(df):.1f}%)")
        print(f"  Gaia DR3: {gaia_rad}/{len(df)} ({100*gaia_rad/len(df):.1f}%)")

        # Distance (from parallax)
        gaia_dist = (df['gaia_parallax'].notna() & (df['gaia_parallax'] > 0)).sum()
        print(f"\nDistance (from parallax): {gaia_dist}/{len(df)} ({100*gaia_dist/len(df):.1f}%)")


def main():
    """Run Gaia cross-match on a sample."""
    from data_loader import ExoplanetDataLoader

    # Load KOI data
    loader = ExoplanetDataLoader()
    koi_df = loader.load_koi()

    # Cross-match with Gaia (start with small sample for testing)
    crossmatcher = GaiaCrossmatcher()

    print("\n*** TESTING MODE: Cross-matching first 50 targets ***")
    print("*** Remove sample_size parameter to process all targets ***\n")

    enriched_df = crossmatcher.crossmatch_koi(
        koi_df,
        output_file="koi_with_gaia_sample.csv",
        sample_size=50  # Remove this to process all
    )

    print("\n✓ Sample cross-match complete!")
    print("To process all targets, edit this script and remove the sample_size parameter.")


if __name__ == "__main__":
    main()
