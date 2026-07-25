"""CLI script to train and benchmark baseline machine learning models."""

import argparse
from pathlib import Path
import pandas as pd
from config.settings import settings
from data.cleaner import DataCleaner
from features.builder import FeatureBuilder
from training.trainer import ModelTrainer
from utils.logger import get_logger

logger = get_logger("scripts.train_baselines")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and benchmark baseline churn prediction models.")
    parser.add_argument("--folds", type=int, default=5, help="Number of StratifiedKFold splits.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    raw_path = settings.raw_data_dir / "customer_churn_dataset.csv"
    if not raw_path.exists():
        logger.error(f"Raw dataset not found at {raw_path}. Run generate_data script first!")
        return

    logger.info(f"Loading raw dataset from {raw_path}...")
    df_raw = pd.read_csv(raw_path)

    cleaner = DataCleaner()
    df_clean = cleaner.clean(df_raw)

    builder = FeatureBuilder(enforce_leakage_guard=True)
    df_featured = builder.transform(df_clean)

    logger.info("Starting baseline model training suite with StratifiedKFold cross-validation...")
    trainer = ModelTrainer(n_splits=args.folds, seed=args.seed)

    # For fast CLI demonstration benchmark, train core fast baselines
    fast_baselines = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Extra Trees",
        "Gradient Boosting",
        "Naive Bayes",
    ]
    results_df = trainer.train_and_evaluate_baselines(df_featured, selected_models=fast_baselines)

    out_csv = settings.artifacts_dir / "baseline_model_benchmark.csv"
    results_df.to_csv(out_csv, index=False)

    logger.info(f"\n==================== BASELINE BENCHMARK RESULTS ====================\n{results_df.to_string(index=False)}")
    logger.info(f"Saved baseline benchmark results to {out_csv}")


if __name__ == "__main__":
    main()
