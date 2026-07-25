"""Seeded, reproducible synthetic customer dataset generator with documented ground-truth signal."""

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from config.settings import settings
from data.schema import (
    ALL_PREDICTOR_FEATURES,
    CATEGORICAL_FEATURES,
    ID_COLS,
    LEAKAGE_FIELDS,
    NUMERICAL_FEATURES,
    TARGET_COL,
)
from utils.logger import get_logger

logger = get_logger("data.generator")


class CustomerDataGenerator:
    """Enterprise Customer Dataset Generator with deliberate ground-truth signal and audit leakage fields."""

    def __init__(
        self,
        n_samples: int = 100000,
        churn_rate_target: float = 0.18,
        seed: int = 42,
    ) -> None:
        self.n_samples = n_samples
        self.churn_rate_target = churn_rate_target
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate(self) -> pd.DataFrame:
        """Generate synthetic customer dataset with realistic distributions and engineered ground-truth signal."""
        logger.info(f"Generating {self.n_samples} customer records with seed={self.seed}...")

        # 1. Customer IDs
        customer_ids = [f"CUST-{100000 + i}" for i in range(self.n_samples)]

        # 2. Demographics
        age = self.rng.integers(18, 75, size=self.n_samples)
        gender = self.rng.choice(["Female", "Male"], size=self.n_samples, p=[0.51, 0.49])
        partner = self.rng.choice(["Yes", "No"], size=self.n_samples, p=[0.48, 0.52])
        dependents = self.rng.choice(["Yes", "No"], size=self.n_samples, p=[0.30, 0.70])
        geography = self.rng.choice(
            ["North America", "Europe", "Asia-Pacific", "Latin America"],
            size=self.n_samples,
            p=[0.40, 0.30, 0.20, 0.10],
        )

        # 3. Account / Subscription History
        tenure_months = self.rng.integers(1, 72, size=self.n_samples)
        contract_type = self.rng.choice(
            ["Month-to-month", "One year", "Two year"],
            size=self.n_samples,
            p=[0.55, 0.25, 0.20],
        )
        payment_method = self.rng.choice(
            ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
            size=self.n_samples,
            p=[0.35, 0.15, 0.25, 0.25],
        )
        auto_pay_enabled = np.where(
            np.isin(payment_method, ["Bank transfer", "Credit card"]),
            self.rng.choice(["Yes", "No"], size=self.n_samples, p=[0.80, 0.20]),
            self.rng.choice(["Yes", "No"], size=self.n_samples, p=[0.10, 0.90]),
        )
        plan_tier = self.rng.choice(
            ["Basic", "Standard", "Premium", "Enterprise"],
            size=self.n_samples,
            p=[0.40, 0.35, 0.20, 0.05],
        )

        # Base monthly charge depends on plan tier
        tier_base_charge = {
            "Basic": 29.99,
            "Standard": 59.99,
            "Premium": 99.99,
            "Enterprise": 199.99,
        }
        base_charges = np.vectorize(tier_base_charge.get)(plan_tier)
        monthly_charges = np.round(
            base_charges + self.rng.normal(0, 5, size=self.n_samples), 2
        )
        monthly_charges = np.maximum(19.99, monthly_charges)
        total_charges = np.round(
            monthly_charges * tenure_months + self.rng.normal(0, 20, size=self.n_samples), 2
        )
        total_charges = np.maximum(monthly_charges, total_charges)

        # 4. Usage Behavior & Engagement
        avg_monthly_gb_download = np.round(
            self.rng.gamma(shape=3.0, scale=35.0, size=self.n_samples), 1
        )
        active_days_per_month = self.rng.integers(1, 31, size=self.n_samples)
        login_frequency_30d = self.rng.integers(0, 60, size=self.n_samples)
        app_usage_hours = np.round(
            self.rng.exponential(scale=15.0, size=self.n_samples), 1
        )
        feature_usage_score = np.round(
            self.rng.beta(a=5, b=2, size=self.n_samples) * 100, 1
        )
        change_in_usage_pct = np.round(
            self.rng.normal(loc=-0.05, scale=0.30, size=self.n_samples), 3
        )

        # 5. Support & Interactions
        support_tickets_30d = self.rng.poisson(lam=0.8, size=self.n_samples)
        support_tickets_90d = support_tickets_30d + self.rng.poisson(lam=1.5, size=self.n_samples)
        complaints_opened = self.rng.poisson(lam=0.3, size=self.n_samples)
        resolution_satisfaction_score = self.rng.choice(
            [1, 2, 3, 4, 5], size=self.n_samples, p=[0.10, 0.15, 0.25, 0.35, 0.15]
        )
        chat_interactions = self.rng.poisson(lam=1.2, size=self.n_samples)

        # 6. Payment & Financial Risk Indicators
        payment_failures_90d = self.rng.poisson(lam=0.4, size=self.n_samples)
        late_payments_12m = payment_failures_90d + self.rng.poisson(lam=0.6, size=self.n_samples)
        price_increase_applied_30d = self.rng.choice(
            ["Yes", "No"], size=self.n_samples, p=[0.20, 0.80]
        )
        credit_card_expiring_soon = self.rng.choice(
            ["Yes", "No"], size=self.n_samples, p=[0.08, 0.92]
        )

        # 7. Risk & Marketing Indicators
        nps_score = self.rng.choice(
            np.arange(0, 11), size=self.n_samples, p=[0.05, 0.05, 0.05, 0.05, 0.05, 0.10, 0.10, 0.15, 0.20, 0.10, 0.10]
        )
        competitor_offer_viewed = self.rng.choice(
            ["Yes", "No"], size=self.n_samples, p=[0.25, 0.75]
        )
        marketing_emails_opened_30d = self.rng.integers(0, 15, size=self.n_samples)
        retention_offer_accepted = self.rng.choice(
            ["Yes", "No"], size=self.n_samples, p=[0.12, 0.88]
        )

        # 8. Deliberately Engineered Ground-Truth Signal (Log Odds)
        # Logit calculation expressing true causal/predictive relationships
        logit = (
            -2.20
            + 1.35 * (contract_type == "Month-to-month")
            + 0.55 * support_tickets_30d
            + 0.75 * payment_failures_90d
            + 0.60 * (resolution_satisfaction_score <= 2)
            + 0.80 * (change_in_usage_pct < -0.25)
            + 0.50 * (price_increase_applied_30d == "Yes")
            + 0.65 * (competitor_offer_viewed == "Yes")
            + 0.45 * (nps_score <= 4)
            - 0.035 * tenure_months
            - 0.40 * (auto_pay_enabled == "Yes")
            - 0.35 * (retention_offer_accepted == "Yes")
            + self.rng.normal(0, 0.4, size=self.n_samples)
        )

        probabilities = 1.0 / (1.0 + np.exp(-logit))

        # Adjust intercept scaling dynamically to hit exact churn_rate_target
        threshold = np.quantile(probabilities, 1.0 - self.churn_rate_target)
        churn_label = (probabilities >= threshold).astype(int)

        # 9. Generate Target Leakage Fields (Strictly conditional on churn_label == 1)
        cancellation_processed_date = [
            f"2026-06-{(i % 28) + 1:02d}" if label == 1 else None for i, label in enumerate(churn_label)
        ]
        final_invoice_flag = [1 if label == 1 else 0 for label in churn_label]
        account_status_deactivated = [1 if label == 1 else 0 for label in churn_label]
        churn_reasons = [
            "Competitor pricing",
            "Service outage",
            "Customer service dissatisfaction",
            "Price increase",
            "Relocation",
        ]
        churn_reason_recorded = [
            self.rng.choice(churn_reasons) if label == 1 else None for label in churn_label
        ]

        df = pd.DataFrame(
            {
                "customer_id": customer_ids,
                "age": age,
                "gender": gender,
                "partner": partner,
                "dependents": dependents,
                "geography": geography,
                "tenure_months": tenure_months,
                "contract_type": contract_type,
                "payment_method": payment_method,
                "auto_pay_enabled": auto_pay_enabled,
                "plan_tier": plan_tier,
                "monthly_charges": monthly_charges,
                "total_charges": total_charges,
                "avg_monthly_gb_download": avg_monthly_gb_download,
                "active_days_per_month": active_days_per_month,
                "login_frequency_30d": login_frequency_30d,
                "app_usage_hours": app_usage_hours,
                "feature_usage_score": feature_usage_score,
                "change_in_usage_pct": change_in_usage_pct,
                "support_tickets_30d": support_tickets_30d,
                "support_tickets_90d": support_tickets_90d,
                "complaints_opened": complaints_opened,
                "resolution_satisfaction_score": resolution_satisfaction_score,
                "chat_interactions": chat_interactions,
                "payment_failures_90d": payment_failures_90d,
                "late_payments_12m": late_payments_12m,
                "price_increase_applied_30d": price_increase_applied_30d,
                "credit_card_expiring_soon": credit_card_expiring_soon,
                "nps_score": nps_score,
                "competitor_offer_viewed": competitor_offer_viewed,
                "marketing_emails_opened_30d": marketing_emails_opened_30d,
                "retention_offer_accepted": retention_offer_accepted,
                # Leakage fields
                "cancellation_processed_date": cancellation_processed_date,
                "final_invoice_flag": final_invoice_flag,
                "account_status_deactivated": account_status_deactivated,
                "churn_reason_recorded": churn_reason_recorded,
                # Target Label
                TARGET_COL: churn_label,
            }
        )

        actual_rate = df[TARGET_COL].mean()
        logger.info(f"Generated dataset shape {df.shape}. Empirical churn rate: {actual_rate:.4f}")
        return df

    def save(self, df: pd.DataFrame, output_dir: Path | None = None) -> Path:
        """Save dataset to raw directory with accompanying metadata manifest."""
        out_dir = output_dir or settings.raw_data_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / "customer_churn_dataset.csv"
        manifest_path = out_dir / "data_manifest.json"

        df.to_csv(csv_path, index=False)
        logger.info(f"Saved dataset CSV to {csv_path}")

        manifest = {
            "n_samples": len(df),
            "n_features": len(df.columns) - 1,
            "churn_rate": float(df[TARGET_COL].mean()),
            "seed": self.seed,
            "target_column": TARGET_COL,
            "id_columns": ID_COLS,
            "leakage_fields": LEAKAGE_FIELDS,
            "categorical_features": CATEGORICAL_FEATURES,
            "numerical_features": NUMERICAL_FEATURES,
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Saved data manifest JSON to {manifest_path}")
        return csv_path


def generate_and_save_dataset(
    n_samples: int = 100000,
    seed: int = 42,
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Utility function to generate and save synthetic dataset."""
    gen = CustomerDataGenerator(n_samples=n_samples, seed=seed)
    df = gen.generate()
    gen.save(df, output_dir=output_dir)
    return df
