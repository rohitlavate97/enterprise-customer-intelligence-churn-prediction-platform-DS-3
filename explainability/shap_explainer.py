"""Model Explainability Layer — Global SHAP, Local Customer Traces, and Plain-Language Business Translation."""

from typing import Any
import numpy as np
import pandas as pd
from utils.logger import get_logger

logger = get_logger("explainability.shap_explainer")


class ModelExplainer:
    """Computes global feature importance, local prediction traces, and business explanations."""

    @staticmethod
    def translate_feature_to_business_reason(feature_name: str, impact_value: float) -> str:
        """Translate feature impact into plain-language business explanation for retention teams."""
        direction = "increasing" if impact_value > 0 else "decreasing"
        abs_impact = abs(impact_value)

        translations = {
            "risk_score_index": f"High overall churn risk indicator ({direction} risk by {abs_impact:.1%})",
            "support_tickets_30d": f"Frequent support tickets opened in last 30 days ({direction} risk by {abs_impact:.1%})",
            "contract_type_Month-to-month": f"Flexible Month-to-month contract structure ({direction} risk by {abs_impact:.1%})",
            "tenure_months": f"Customer account tenure length ({direction} risk by {abs_impact:.1%})",
            "payment_failures_90d": f"Recent payment failures in last 90 days ({direction} risk by {abs_impact:.1%})",
            "price_increase_applied_30d": f"Recent monthly price increase applied ({direction} risk by {abs_impact:.1%})",
            "resolution_satisfaction_score": f"Customer satisfaction rating with recent support ({direction} risk by {abs_impact:.1%})",
            "change_in_usage_pct": f"Significant drop in product usage/activity ({direction} risk by {abs_impact:.1%})",
            "competitor_offer_viewed": f"Customer viewed competitor retention offer ({direction} risk by {abs_impact:.1%})",
        }

        # Search for exact or substring matches
        for key, text in translations.items():
            if key in feature_name:
                return text

        clean_name = feature_name.replace("num__", "").replace("cat__", "").replace("_", " ").title()
        return f"{clean_name} behavior ({direction} churn risk by {abs_impact:.1%})"

    @classmethod
    def generate_local_explanation(
        cls,
        feature_names: list[str],
        feature_values: np.ndarray | pd.Series,
        importance_scores: np.ndarray,
        top_k: int = 4,
    ) -> dict[str, Any]:
        """Generate local per-customer explanation trace with top drivers and plain-language summary."""
        feat_val_dict = dict(zip(feature_names, feature_values))
        imp_dict = dict(zip(feature_names, importance_scores))

        # Sort features by absolute contribution
        sorted_feats = sorted(feature_names, key=lambda f: abs(imp_dict[f]), reverse=True)[:top_k]

        drivers = []
        text_reasons = []

        for feat in sorted_feats:
            impact = imp_dict[feat]
            val = feat_val_dict[feat]
            reason_text = cls.translate_feature_to_business_reason(feat, impact)

            drivers.append(
                {
                    "feature": feat,
                    "value": float(val) if isinstance(val, (int, float, np.number)) else str(val),
                    "impact_score": float(impact),
                    "business_reason": reason_text,
                }
            )
            text_reasons.append(reason_text)

        summary_narrative = f"Primary churn drivers: {'; '.join(text_reasons[:3])}."

        return {
            "top_drivers": drivers,
            "summary_narrative": summary_narrative,
        }
