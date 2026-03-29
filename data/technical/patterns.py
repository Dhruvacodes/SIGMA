"""
SIGMA Pattern Classifiers.
Bulk deal intent classifier using feature-based scoring.
"""

from typing import Any


class BulkDealClassifier:
    """
    Classifier for bulk deal intent using heuristic rules.

    The classifier uses a simple feature-based scoring model
    (no training data required — uses heuristic rules that mimic
    what an ML classifier would learn).
    """

    def classify_intent(
        self, deal: dict, earnings_data: dict | None = None, pledge_data: dict | None = None
    ) -> dict[str, Any]:
        """
        Classify whether a bulk deal is "distress_selling" or "routine_block".

        Args:
            deal: Dict containing deal details:
                - price_discount_to_market: float (negative = discount)
                - stake_sold_pct: float
                - management_commentary_sentiment: str (optional)
            earnings_data: Dict with {q1_margin, q2_margin, q3_margin, q4_margin}
            pledge_data: Dict with {pledged_pct: float}

        Returns:
            Dict with classification results.
        """
        earnings_data = earnings_data or {}
        pledge_data = pledge_data or {}

        distress_score = 0.0
        routine_score = 0.0
        feature_breakdown = {}

        # Feature 1: Price discount to market
        discount = deal.get("price_discount_to_market", deal.get("price_discount_to_prev_close_pct", 0))
        if discount > 5:
            distress_score += 0.3
            feature_breakdown["price_discount"] = {"contribution": "distress +0.3", "value": discount}
        elif 2 <= discount <= 5:
            distress_score += 0.1
            feature_breakdown["price_discount"] = {"contribution": "distress +0.1", "value": discount}
        else:
            routine_score += 0.2
            feature_breakdown["price_discount"] = {"contribution": "routine +0.2", "value": discount}

        # Feature 2: Stake sold percentage
        stake_pct = deal.get("stake_sold_pct", deal.get("quantity_pct_equity", 0))
        if stake_pct > 3:
            distress_score += 0.2
            feature_breakdown["stake_sold"] = {"contribution": "distress +0.2", "value": stake_pct}
        elif stake_pct < 1:
            routine_score += 0.1
            feature_breakdown["stake_sold"] = {"contribution": "routine +0.1", "value": stake_pct}
        else:
            feature_breakdown["stake_sold"] = {"contribution": "neutral", "value": stake_pct}

        # Feature 3: Earnings trajectory
        margins = [
            earnings_data.get("q1_margin"),
            earnings_data.get("q2_margin"),
            earnings_data.get("q3_margin"),
            earnings_data.get("q4_margin"),
        ]
        valid_margins = [m for m in margins if m is not None]

        if len(valid_margins) >= 3:
            # Check for consecutive contraction
            contractions = sum(
                1 for i in range(1, len(valid_margins)) if valid_margins[i] < valid_margins[i - 1]
            )
            if contractions >= 3:
                distress_score += 0.25
                feature_breakdown["earnings_trajectory"] = {
                    "contribution": "distress +0.25",
                    "pattern": "consecutive_contraction",
                }
            elif contractions == 0 and len(valid_margins) >= 2:
                routine_score += 0.15
                feature_breakdown["earnings_trajectory"] = {
                    "contribution": "routine +0.15",
                    "pattern": "expansion",
                }
            else:
                feature_breakdown["earnings_trajectory"] = {"contribution": "neutral", "pattern": "mixed"}
        else:
            feature_breakdown["earnings_trajectory"] = {"contribution": "neutral", "data": "insufficient"}

        # Feature 4: Pledge percentage
        pledged_pct = pledge_data.get("pledged_pct", 0)
        if pledged_pct > 10:
            distress_score += 0.2
            feature_breakdown["pledge_pct"] = {"contribution": "distress +0.2", "value": pledged_pct}
        elif 5 <= pledged_pct <= 10:
            distress_score += 0.1
            feature_breakdown["pledge_pct"] = {"contribution": "distress +0.1", "value": pledged_pct}
        else:
            routine_score += 0.05
            feature_breakdown["pledge_pct"] = {"contribution": "routine +0.05", "value": pledged_pct}

        # Feature 5: Management commentary sentiment
        sentiment = deal.get("management_commentary_sentiment")
        if sentiment == "negative":
            distress_score += 0.1
            feature_breakdown["management_sentiment"] = {
                "contribution": "distress +0.1",
                "value": sentiment,
            }
        elif sentiment == "positive":
            routine_score += 0.1
            feature_breakdown["management_sentiment"] = {
                "contribution": "routine +0.1",
                "value": sentiment,
            }
        else:
            feature_breakdown["management_sentiment"] = {"contribution": "neutral", "value": sentiment}

        # Normalize to probabilities
        total = distress_score + routine_score
        if total == 0:
            distress_prob = 0.5
            routine_prob = 0.5
        else:
            distress_prob = distress_score / total
            routine_prob = routine_score / total

        # Classification thresholds
        if distress_prob > 0.65:
            classification = "likely_distress"
            threshold_explanation = f"Distress probability {distress_prob:.2f} > 0.65 threshold"
        elif distress_prob < 0.40:
            classification = "likely_routine"
            threshold_explanation = f"Distress probability {distress_prob:.2f} < 0.40 threshold"
        else:
            classification = "inconclusive"
            threshold_explanation = f"Distress probability {distress_prob:.2f} between 0.40-0.65"

        return {
            "distress_probability": distress_prob,
            "routine_probability": routine_prob,
            "classification": classification,
            "feature_breakdown": feature_breakdown,
            "threshold_explanation": threshold_explanation,
        }
