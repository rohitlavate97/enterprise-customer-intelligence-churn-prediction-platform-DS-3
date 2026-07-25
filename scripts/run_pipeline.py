"""CLI script to run data cleaning, quality reporting, and versioning pipeline."""

import argparse
from pathlib import Path
import pandas as pd
from config.settings import settings
from data.cleaner import DataCleaner
from data.quality_report import DataQualityReporter
from data.versioning import DatasetVersionManager
from features.builder import FeatureBuilder
from utils.logger import get_logger

logger = get_logger("scripts.run_pipeline")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Data Pipeline (Clean, Version, Quality Report).")
    parser.add_argument(
        "--raw-path",
        type=str,
        default=str(settings.raw_data_dir / "customer_churn_dataset.csv"),
        help="Path to raw customer churn CSV.",
    )
    args = parser.parse_args()

    raw_path = Path(args.raw_path)
    if not raw_path.exists():
        logger.error(f"Raw data file not found at {raw_path}. Run generate_data script first!")
        return

    logger.info(f"Loading raw dataset from {raw_path}...")
    df_raw = pd.read_csv(raw_path)

    # 1. Clean
    cleaner = DataCleaner()
    df_clean = cleaner.clean(df_raw)

    # 2. Quality Report
    report_path = settings.processed_data_dir / "data_quality_report.json"
    DataQualityReporter.save_report(df_clean, report_path)

    # 3. Feature Builder (Leakage Safe)
    builder = FeatureBuilder()
    df_featured = builder.transform(df_clean)

    # 4. Version & Register
    filepath, checksum = DatasetVersionManager.register_version(
        df_featured,
        dataset_name="processed_customer_churn",
        output_dir=settings.processed_data_dir,
        metadata={"raw_source": str(raw_path), "stage": "processed_featured"},
    )

    logger.info(f"Pipeline complete! Processed & versioned dataset at {filepath} (Checksum: {checksum[:12]}...)")


if __name__ == "__main__":
    main()
