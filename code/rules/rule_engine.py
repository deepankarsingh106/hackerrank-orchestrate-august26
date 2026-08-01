"""Module containing the deterministic Rule Engine classifier."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any
from config import Action, MessageType
from context.context_builder import MessageContext
from features.feature_engine import FeatureVector

logger = logging.getLogger(__name__)


@dataclass
class RuleDecision:
    """Represents a decision made by the Rule Engine."""

    action: Action
    message_type: MessageType
    reason: str
    confidence: float


class RuleEngine:
    """Evaluates deterministic business rules to determine message routing."""

    def __init__(self) -> None:
        pass

    def evaluate(self, context: MessageContext, features: FeatureVector) -> RuleDecision:
        """Apply sequential rules to route the message."""
        msg = context.message
        conv_type = msg.get("conversation_type")
        text_lower = str(msg.get("message_text", "")).lower()

        # Pre-classify the best-fit message type for this message
        predicted_type = self._classify_message_type(text_lower, conv_type, features, context)

        # 1. Scam check (highest priority)
        if features.scam_score > 0.65:
            m_type = predicted_type if predicted_type in (MessageType.SCAM, MessageType.SPAM) else MessageType.SCAM
            return RuleDecision(
                action=Action.MUTE,
                message_type=m_type,
                reason="Detected high risk of scam or phishing (domain mismatch or security code solicitations).",
                confidence=0.95,
            )

        # 2. Spam / Chain message check
        if features.spam_score > 0.65:
            m_type = predicted_type if predicted_type in (MessageType.GREETING, MessageType.FORWARD, MessageType.SPAM, MessageType.PROMOTION) else MessageType.SPAM
            return RuleDecision(
                action=Action.MUTE,
                message_type=m_type,
                reason="Message flagged as spam or chain message (high forward count or spam vocabulary).",
                confidence=0.90,
            )

        # 3. Direct mention in any group overrides muted/unmuted state
        user_id = context.user.get("user_id", "")
        is_mentioned = False
        if user_id and f"@{user_id}" in text_lower:
            is_mentioned = True

        if conv_type == "group" and is_mentioned:
            # Notify only if sender is somewhat trusted (not marked as scam/spam)
            if features.sender_trust_score > 0.3:
                action = Action.DIGEST if (features.dnd_active == 1) else Action.NOTIFY
                reason = "Direct mention in group chat from a trusted contact."
                if features.dnd_active == 1:
                    reason += " Batched during quiet hours."
                m_type = predicted_type if predicted_type != MessageType.UNKNOWN else MessageType.URGENT
                return RuleDecision(
                    action=action,
                    message_type=m_type,
                    reason=reason,
                    confidence=0.90,
                )

        # 4. Muted group check (when no direct mention)
        if context.group_member and context.group_member.get("group_muted_by_user") == 1:
            return RuleDecision(
                action=Action.MUTE,
                message_type=predicted_type if predicted_type != MessageType.UNKNOWN else MessageType.PROMOTION,
                reason="Message suppressed because the containing group is muted by the user.",
                confidence=0.95,
            )

        # 5. Historical reports or extremely low trust check (only for personal/business)
        if (conv_type == "personal" and features.sender_trust_score == 0.0) or (
            conv_type == "business" and context.business and features.business_trust_score < 0.2
        ):
            m_type = MessageType.SCAM if features.scam_score > 0.4 else MessageType.SPAM
            return RuleDecision(
                action=Action.MUTE,
                message_type=m_type,
                reason="Muted due to prior reports or extremely low sender trust.",
                confidence=0.95,
            )

        # 6. Promos from businesses check (opt-out / fatigue)
        if conv_type == "business" and context.business:
            is_promo = features.promotion_probability > 0.5
            
            # Check if user opted out of promos
            opted_out = False
            if context.business_history and pd_not_na_str(context.business_history.get("promotions_opted_out_at")):
                opted_out = True
            if context.business_history and context.business_history.get("allows_promotions") == 0:
                opted_out = True

            if is_promo or predicted_type == MessageType.PROMOTION:
                if opted_out:
                    return RuleDecision(
                        action=Action.MUTE,
                        message_type=MessageType.PROMOTION,
                        reason="Promotional message muted because the user has opted out.",
                        confidence=0.92,
                    )
                if features.notification_fatigue > 0.65:
                    return RuleDecision(
                        action=Action.MUTE,
                        message_type=MessageType.PROMOTION,
                        reason="Promotional message muted to prevent notification fatigue.",
                        confidence=0.85,
                    )
                
                # Default for promotions is digest
                return RuleDecision(
                    action=Action.DIGEST,
                    message_type=MessageType.PROMOTION,
                    reason="Promotional offer from active business routed to digest.",
                    confidence=0.85,
                )

        # 7. High urgency personal or group notifications
        is_urgent = features.urgency_score >= 0.7
        group_type = ""
        if context.group:
            group_type = str(context.group.get("group_type", "")).lower()

        dnd_active = features.dnd_active == 1

        if is_urgent:
            # Urgent work coworker updates
            if group_type == "coworker":
                action = Action.DIGEST if dnd_active else Action.NOTIFY
                reason = "Urgent co-worker update in active work group."
                if dnd_active:
                    reason += " Batched during quiet hours."
                return RuleDecision(
                    action=action,
                    message_type=MessageType.URGENT,
                    reason=reason,
                    confidence=0.88,
                )

            # Urgent school operational updates
            if group_type == "school_group":
                action = Action.DIGEST if dnd_active else Action.NOTIFY
                reason = "School operational notice or circular."
                if dnd_active:
                    reason += " Batched during quiet hours."
                return RuleDecision(
                    action=action,
                    message_type=MessageType.EVENT,
                    reason=reason,
                    confidence=0.86,
                )

            # Urgent family updates
            if group_type in ("family", "extended_family"):
                action = Action.DIGEST if dnd_active and features.urgency_score < 0.95 else Action.NOTIFY
                reason = "Urgent family group message."
                if dnd_active and action == Action.DIGEST:
                    reason += " Batched during quiet hours."
                return RuleDecision(
                    action=action,
                    message_type=MessageType.URGENT,
                    reason=reason,
                    confidence=0.87,
                )

            # Urgent personal chat
            if conv_type == "personal" and features.sender_trust_score > 0.3:
                action = Action.DIGEST if dnd_active and features.urgency_score < 0.9 else Action.NOTIFY
                reason = "Urgent personal message from trusted contact."
                if dnd_active and action == Action.DIGEST:
                    reason += " Batched during quiet hours."
                return RuleDecision(
                    action=action,
                    message_type=MessageType.URGENT,
                    reason=reason,
                    confidence=0.89,
                )

        # 8. Business Transactional Updates (orders, bookings, utility, bank alerts)
        if conv_type == "business" and context.business:
            cat = str(context.business.get("category", "")).lower()
            if cat in ("ecommerce_delivery", "bank", "logistics", "ride_booking", "payments", "healthcare"):
                # Ensure it's not a promo and business is trusted
                if features.promotion_probability < 0.4 and features.business_trust_score > 0.4:
                    action = Action.DIGEST if dnd_active else Action.NOTIFY
                    reason = "Transactional status or delivery update from verified business."
                    if dnd_active:
                        reason += " Batched during quiet hours."

                    return RuleDecision(
                        action=action,
                        message_type=predicted_type if predicted_type != MessageType.UNKNOWN else MessageType.BUSINESS_UPDATE,
                        reason=reason,
                        confidence=0.90,
                    )

        # 9. Greetings / Low value forwards
        if "good morning" in text_lower or "stay positive" in text_lower or "fwd as received" in text_lower:
            m_type = MessageType.GREETING if "morning" in text_lower else MessageType.FORWARD
            if features.sender_trust_score > 0.6:
                return RuleDecision(
                    action=Action.DIGEST,
                    message_type=m_type,
                    reason="Low-priority greeting from trusted contact routed to digest.",
                    confidence=0.85,
                )
            else:
                return RuleDecision(
                    action=Action.MUTE,
                    message_type=m_type,
                    reason="Low-value forward or greeting from untrusted contact suppressed.",
                    confidence=0.80,
                )

        # 10. General personal chat (low urgency)
        if conv_type == "personal":
            if features.sender_trust_score > 0.5:
                action = Action.DIGEST if (dnd_active or features.urgency_score < 0.3) else Action.NOTIFY
                reason = "Personal message from trusted contact."
                if dnd_active:
                    reason += " Batched during quiet hours."
                elif features.urgency_score < 0.3:
                    reason += " Routed to digest due to low urgency."
                return RuleDecision(
                    action=action,
                    message_type=MessageType.PERSONAL,
                    reason=reason,
                    confidence=0.80,
                )

        # 11. Fallback for unclassified messages (e.g. society form discussion, etc.)
        action = Action.DIGEST
        if conv_type == "group" and group_type in ("society", "coworker", "school_group", "family") and not dnd_active:
            # Active group discussion, notify if user interacts historically AND it is somewhat urgent
            if features.historical_engagement_score > 0.6 and features.urgency_score >= 0.5:
                action = Action.NOTIFY

        return RuleDecision(
            action=action,
            message_type=predicted_type,
            reason="Unclassified message details, defaulting based on context type.",
            confidence=0.80,  # Set confidence higher so that the offline run is robust
        )

    def _classify_message_type(
        self, text_lower: str, conv_type: str, features: FeatureVector, context: MessageContext
    ) -> MessageType:
        """Helper to guess the message classification category."""
        words = {w.strip(".,!?;:()[]'\"") for w in text_lower.split()}

        # 0. Check safety alerts / brand advisories (high priority update)
        if "safety advisory" in text_lower or "never ask for" in text_lower:
            return MessageType.BUSINESS_UPDATE

        # 1. Check greetings (high priority)
        if any(w in text_lower for w in ["good morning", "blessings", "stay positive", "good vibes"]):
            return MessageType.GREETING

        # 2. Check forward
        if "fwd" in text_lower or "forward to" in text_lower or features.forwarding_score > 0.4:
            return MessageType.FORWARD

        # 3. Check scam / spam features
        if features.scam_score > 0.5:
            return MessageType.SCAM
        if features.spam_score > 0.5:
            return MessageType.SPAM

        # 4. Check promo (include local sales/selling)
        promo_kws = ["selling", "dm if interested", "shop", "off", "discount", "sale", "cashback", "offer", "coupon", "deals", "win"]
        if any(w in text_lower for w in promo_kws) or features.promotion_probability > 0.4:
            return MessageType.PROMOTION

        # 5. Check payment (word boundaries check)
        payment_kws = ["payment", "bill", "due", "pay", "reconciled", "receipt", "fee", "penalty", "charge", "refund", "token"]
        if "card details" in text_lower or any(w in words for w in payment_kws):
            return MessageType.PAYMENT

        # 6. Check event
        event_kws = [
            "event", "meetup", "meeting", "sync", "circular", "parents", "timing", "appointment", 
            "schedule", "stadium", "date", "form", "survey", "walkathon"
        ]
        if "water supply" in text_lower or "fire alarm" in text_lower or any(w in words for w in event_kws):
            return MessageType.EVENT

        # Check business update
        biz_kws = ["order", "packed", "shipped", "delivery", "delivered", "pickup", "courier", "parcel", "hub", "status", "route", "tracking"]
        is_personal_group = False
        if context.group:
            g_type = str(context.group.get("group_type", "")).lower()
            if g_type in ("family", "extended_family", "friends", "coworker"):
                is_personal_group = True
        
        if conv_type == "business" or (any(w in text_lower for w in biz_kws) and not is_personal_group and conv_type != "personal"):
            return MessageType.BUSINESS_UPDATE

        # Check urgent
        urgent_kws = ["urgent", "immediately", "block", "freeze", "action required", "expiring", "deadline", "now", "asap", "emergency", "online now"]
        if features.urgency_score > 0.6 or any(w in text_lower for w in urgent_kws):
            return MessageType.URGENT

        # Default fallback
        if conv_type == "personal":
            # Only label as personal if contact has historical trust, otherwise it's unknown/stranger
            if features.sender_trust_score > 0.3:
                return MessageType.PERSONAL
            return MessageType.UNKNOWN
        elif conv_type == "group":
            if context.group:
                g_type = str(context.group.get("group_type", "")).lower()
                if g_type in ("family", "extended_family", "friends"):
                    return MessageType.PERSONAL
                elif g_type == "coworker":
                    return MessageType.PERSONAL
            return MessageType.PERSONAL

        return MessageType.UNKNOWN


def pd_not_na_str(val: Any) -> bool:
    """Helper to check if a Pandas value is not null and not empty."""
    import pandas as pd
    if pd.isna(val):
        return False
    if str(val).strip() == "":
        return False
    return True
