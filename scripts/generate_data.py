"""CLI script for generating raw synthetic customer churn dataset."""

import argparse
from data.generator import generate_and_save_dataset
from utils.logger import get_logger

logger = get_logger("scripts.generate_data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic customer dataset.")
    parser.add_argument("--samples", type=int, default=100000, help="Number of customer records to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    logger.info(f"Triggering data generation script: samples={args.samples}, seed={args.seed}")
    generate_and_save_dataset(n_samples=args.samples, seed=args.seed)
    logger.info("Synthetic customer dataset generation complete.")


if __name__ == "__main__":
    main()
