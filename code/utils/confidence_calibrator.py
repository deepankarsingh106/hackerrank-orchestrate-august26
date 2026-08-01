"""Module for calibrating and refining decision confidence scores."""

from __future__ import annotations

import logging
from config import Action, MessageType
from features.feature_engine import FeatureVector
from rules.rule_engine import RuleDecision

logger = logging.getLogger(__name__)


class ConfidenceCalibrator:
    """Calibrates and adjusts raw classification confidence based on contextual features."""

    def __init__(self) -> None:
        pass

    def calibrate(self, decision: RuleDecision, features: FeatureVector) -> RuleDecision:
        """Adjust confidence based on security risk, historical engagement, and policy overrides."""
        raw_conf = decision.confidence
        calibrated_conf = raw_conf

        # 1. Boost confidence for high-risk security matches (scams and phishing)
        if decision.message_type == MessageType.SCAM and decision.action == Action.MUTE:
            if features.scam_score > 0.7:
                calibrated_conf = max(calibrated_conf, 0.98)
            else:
                calibrated_conf = max(calibrated_conf, 0.92)

        # 2. Boost confidence for clear spam matches
        elif decision.message_type == MessageType.SPAM and decision.action == Action.MUTE:
            if features.spam_score > 0.7:
                calibrated_conf = max(calibrated_conf, 0.95)
            else:
                calibrated_conf = max(calibrated_conf, 0.88)

        # 3. Adjust based on historical engagement alignment
        # If user always dismisses or has very low engagement, and we decide to mute or digest
        elif decision.action in (Action.MUTE, Action.DIGEST):
            if features.historical_engagement_score < 0.25:
                calibrated_conf = min(calibrated_conf + 0.05, 1.0)
            elif features.historical_engagement_score > 0.75:
                # Disagreement with user's historical engagement slightly dampens confidence
                calibrated_conf = max(calibrated_conf - 0.08, 0.40)

        # If user has very high engagement, and we decide to notify
        elif decision.action == Action.NOTIFY:
            if features.historical_engagement_score > 0.75:
                calibrated_conf = min(calibrated_conf + 0.05, 1.0)
            elif features.historical_engagement_score < 0.25:
                # User rarely opens/replies, notify might be false positive
                calibrated_conf = max(calibrated_conf - 0.12, 0.40)

        # 4. Handle quiet hours policy confidence (deterministic, so keep high)
        if features.dnd_active == 1 and "quiet hours" in decision.reason.lower():
            calibrated_conf = max(calibrated_conf, 0.85)

        # Ensure bounds [0.0, 1.0]
        final_conf = max(0.0, min(calibrated_conf, 1.0))

        return RuleDecision(
            action=decision.action,
            message_type=decision.message_type,
            reason=decision.reason,
            confidence=round(final_conf, 4),
        )
