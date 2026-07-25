"""CLI script to run automated retraining pipeline and Champion vs Challenger validation gate."""

import json
from config.settings import settings
from training.retraining_pipeline import AutomatedRetrainingPipeline
from utils.logger import get_logger

logger = get_logger("scripts.run_retraining_pipeline")


def main() -> None:
    logger.info("Executing Automated Retraining Pipeline & Champion vs Challenger Gate...")

    pipeline = AutomatedRetrainingPipeline(relative_margin_threshold=0.01)
    result = pipeline.run_retraining_cycle(n_samples=1500, seed=42)

    summary_path = settings.artifacts_dir / "retraining_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    logger.info(f"\n==================== RETRAINING PIPELINE SUMMARY ====================")
    logger.info(f"Candidate Version: {result['version_tag']} | Gate Passed: {result['gate_passed']}")
    logger.info(f"Challenger PR AUC: {result['challenger_pr_auc']:.4f} (Relative Gain: {result['relative_gain_pct']:.2f}%)")
    logger.info(f"Gate Reason: {result['gate_reason']}")
    logger.info(f"Active Champion: {result['active_champion_version']}")
    logger.info(f"Saved Retraining Summary JSON to {summary_path}")


if __name__ == "__main__":
    main()
