"""Module for computing deterministic routing features from message contexts."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time
import pandas as pd
from typing import Any
from context.context_builder import MessageContext

logger = logging.getLogger(__name__)


@dataclass
class FeatureVector:
    """Dataclass holding all 16 required routing features."""

    urgency_score: float
    sender_trust_score: float
    business_trust_score: float
    business_relationship_score: float
    historical_engagement_score: float
    promotion_probability: float
    notification_fatigue: float
    spam_score: float
    scam_score: float
    risk_score: float
    forwarding_score: float
    conversation_priority: float
    group_priority: float
    media_present: int  # 0 or 1
    media_type: str  # '', 'image', 'voice'
    dnd_active: int  # 0 or 1


class FeatureEngine:
    """Computes deterministic routing features for a MessageContext."""

    def __init__(self) -> None:
        pass

    def compute_features(self, context: MessageContext) -> FeatureVector:
        """Compute the 16 features based on the message context."""
        msg = context.message
        user = context.user
        media_present = 1 if msg.get("media_type") in ("image", "voice") else 0
        media_type = msg.get("media_type") if pd.notna(msg.get("media_type")) else ""

        # Extract text (either message_text or voice transcript or OCR text)
        msg_text = msg.get("message_text") if pd.notna(msg.get("message_text")) else ""
        ocr_text = ""
        voice_transcript = ""

        if context.image and "text" in context.image:
            ocr_text = context.image["text"] if pd.notna(context.image["text"]) else ""
        if context.voice and "transcript" in context.voice:
            voice_transcript = context.voice["transcript"] if pd.notna(context.voice["transcript"]) else ""

        full_text = " ".join([msg_text, ocr_text, voice_transcript]).strip()
        full_text_lower = full_text.lower()

        # 1. DND status
        dnd_active = self._check_dnd(msg.get("created_at"), user.get("do_not_disturb_window"))

        # 2. Urgency Score
        urgency_score = self._compute_urgency(full_text_lower, context)

        # 3. Sender Trust Score
        sender_trust_score = self._compute_sender_trust(context)

        # 4. Business Trust Score
        business_trust_score = self._compute_business_trust(context)

        # 5. Business Relationship Score
        business_relationship_score = self._compute_business_relationship(context)

        # 6. Historical Engagement Score
        historical_engagement_score = self._compute_historical_engagement(context)

        # 7. Promotion Probability
        promotion_probability = self._compute_promotion_prob(full_text_lower, context)

        # 8. Notification Fatigue
        notification_fatigue = self._compute_fatigue(context)

        # 9. Forwarding Score
        fwd_count = msg.get("forwarded_count", 0)
        if pd.isna(fwd_count):
            fwd_count = 0
        forwarding_score = min(float(fwd_count) / 10.0, 1.0)

        # 10. Spam Score
        spam_score = self._compute_spam(full_text_lower, forwarding_score, context)

        # 11. Scam Score
        scam_score = self._compute_scam(full_text_lower, sender_trust_score, business_trust_score, context)

        # 12. Risk Score
        risk_score = max(scam_score, spam_score * 0.5)

        # 13. Conversation Priority
        conversation_priority = self._compute_conv_priority(msg, user.get("user_id"), sender_trust_score)

        # 14. Group Priority
        group_priority = self._compute_group_priority(context)

        return FeatureVector(
            urgency_score=round(urgency_score, 4),
            sender_trust_score=round(sender_trust_score, 4),
            business_trust_score=round(business_trust_score, 4),
            business_relationship_score=round(business_relationship_score, 4),
            historical_engagement_score=round(historical_engagement_score, 4),
            promotion_probability=round(promotion_probability, 4),
            notification_fatigue=round(notification_fatigue, 4),
            spam_score=round(spam_score, 4),
            scam_score=round(scam_score, 4),
            risk_score=round(risk_score, 4),
            forwarding_score=round(forwarding_score, 4),
            conversation_priority=round(conversation_priority, 4),
            group_priority=round(group_priority, 4),
            media_present=media_present,
            media_type=media_type,
            dnd_active=dnd_active,
        )

    def _check_dnd(self, created_at_str: Any, dnd_window: Any) -> int:
        """Check if message timestamp falls in the DND window."""
        if pd.isna(created_at_str) or pd.isna(dnd_window):
            return 0
        
        try:
            dnd_str = str(dnd_window).strip()
            if not dnd_str or "-" not in dnd_str:
                return 0
            
            # Parse created_at time
            dt = datetime.strptime(str(created_at_str).strip(), "%Y-%m-%d %H:%M")
            msg_time = dt.time()

            start_str, end_str = dnd_str.split("-")
            start_t = datetime.strptime(start_str.strip(), "%H:%M").time()
            end_t = datetime.strptime(end_str.strip(), "%H:%M").time()

            if start_t <= end_t:
                return 1 if start_t <= msg_time <= end_t else 0
            else:  # Spans midnight
                return 1 if msg_time >= start_t or msg_time <= end_t else 0
        except Exception as e:
            logger.warning("Error parsing DND or timestamp: %s", e)
            return 0

    def _compute_urgency(self, text_lower: str, context: MessageContext) -> float:
        """Calculate urgency score based on keywords and metadata."""
        urgent_keywords = {
            "urgent": 0.85,
            "immediately": 0.85,
            "block": 0.75,
            "freeze": 0.75,
            "action required": 0.8,
            "expiring": 0.8,
            "expire today": 0.9,
            "attention": 0.6,
            "penalty": 0.8,
            "otp": 0.9,
            "verification code": 0.95,
            "login code": 0.95,
            "deadline": 0.75,
            "by 6 pm": 0.75,
            "today": 0.3,
            "confirm now": 0.8,
            "asap": 0.7,
            "come online now": 0.9,
            "online now": 0.85,
            "escalation": 0.9,
            "emergency": 0.95,
            "water supply": 0.75,
            "tanker": 0.75,
            "circular": 0.6,
            "consent note": 0.6,
            "mix-up": 0.5,
            "refund approved": 0.6,
            "now": 0.75,
            "due today": 0.8,
            "failed": 0.6,
            "failing": 0.7,
            "critical": 0.8,
            "incident": 0.7,
            "sync": 0.6,
            "review": 0.6,
            "call": 0.5,
            "meeting": 0.5,
            "early": 0.5,
            "mins": 0.7,
            "minutes": 0.7,
            "plumber": 0.7,
            "stadium": 0.5,
            "cab": 0.6,
            "driver": 0.6,
        }

        score = 0.1
        for kw, weight in urgent_keywords.items():
            if kw in text_lower:
                score = max(score, weight)

        # Voice note urgency estimate
        if context.voice and "urgency_estimate" in context.voice:
            v_urgency = context.voice["urgency_estimate"]
            if pd.notna(v_urgency):
                try:
                    score = max(score, float(v_urgency))
                except ValueError:
                    pass

        # If it contains direct mention in coworker group
        if context.group and context.group.get("group_type") == "coworker":
            user_id = context.user.get("user_id", "")
            if user_id and f"@{user_id}" in text_lower:
                score = max(score, 0.85)

        # Override to low if explicitly stated as low urgency
        low_urgency_phrases = ["nothing urgent", "not urgent", "no rush", "no urgency", "whenever convenient", "whenever you can", "whenever you get time", "nothing dramatic"]
        if any(phrase in text_lower for phrase in low_urgency_phrases):
            score = 0.15

        return min(score, 1.0)

    def _compute_sender_trust(self, context: MessageContext) -> float:
        """Calculate sender trust based on relationship and history."""
        msg = context.message
        conv_type = msg.get("conversation_type")

        if conv_type == "business":
            return 0.0  # Business trust is separate

        if conv_type == "group":
            # If sender is group admin, high trust
            if context.group_member and context.group_member.get("role") == "admin":
                return 0.95
            
            # Check interaction stats inside the group
            if context.group_member:
                sent = context.group_member.get("messages_sent_30d", 0)
                read = context.group_member.get("messages_read_30d", 0)
                replies = context.group_member.get("replies_sent_30d", 0)
                
                if pd.isna(sent): sent = 0
                if pd.isna(read): read = 0
                if pd.isna(replies): replies = 0

                # Calculate user engagement with sender's group contributions
                if sent > 0:
                    read_rate = float(read) / float(sent)
                    reply_rate = float(replies) / float(sent)
                    return min(0.3 + read_rate * 0.4 + reply_rate * 0.3, 1.0)

        # Personal conversation or fallback for group
        sender_id = msg.get("sender_user_id")
        if pd.isna(sender_id) or not sender_id:
            return 0.1

        # Check user's history with this specific sender
        user_history = [h for h in context.history if h.get("sender_user_id") == sender_id]
        if not user_history:
            return 0.2  # Low trust for unknown first-time contacts

        # Calculate historical open & reply rates
        total = len(user_history)
        opened = 0
        replied = 0
        reported = 0
        dismissed = 0

        hist_ids = {h["message_id"] for h in user_history}
        user_events = [e for e in context.events if e.get("message_id") in hist_ids]

        for e in user_events:
            if e.get("message_opened") == 1:
                opened += 1
            if e.get("message_replied") == 1:
                replied += 1
            if e.get("message_reported") == 1:
                reported += 1
            if e.get("notification_dismissed") == 1:
                dismissed += 1

        if reported > 0:
            return 0.0

        trust = 0.2 + (opened / total) * 0.4 + (replied / total) * 0.4 - (dismissed / total) * 0.2
        return max(0.0, min(trust, 1.0))

    def _compute_business_trust(self, context: MessageContext) -> float:
        """Calculate business trust score."""
        if context.message.get("conversation_type") != "business" or not context.business:
            return 0.0

        b = context.business
        score = 0.5  # Base trust

        if b.get("verified") == 1:
            score += 0.3

        # Check domain mismatch
        off_domain = str(b.get("official_domain")).strip().lower()
        sender_domain = str(b.get("domain_used_by_sender")).strip().lower()
        if off_domain and sender_domain and off_domain != sender_domain:
            score -= 0.6  # Severe mismatch penalty (scam alert)

        # Reports penalty
        reports = b.get("user_reports_30d", 0)
        if pd.notna(reports) and reports > 0:
            score -= min(float(reports) / 50.0, 0.4)

        # Age factors
        age = b.get("account_age_days", 0)
        if pd.notna(age) and age > 0:
            score += min(float(age) / 2000.0, 0.2)

        return max(0.0, min(score, 1.0))

    def _compute_business_relationship(self, context: MessageContext) -> float:
        """Calculate business relationship score based on user interaction history."""
        if context.message.get("conversation_type") != "business" or not context.business_history:
            return 0.0

        bh = context.business_history
        score = 0.3

        if bh.get("allows_promotions") == 1:
            score += 0.2
        if pd.notna(bh.get("promotions_opted_out_at")):
            score -= 0.4

        act_count = bh.get("activity_count_180d", 0)
        if pd.notna(act_count) and act_count > 0:
            score += min(float(act_count) / 10.0, 0.4)

        opened = bh.get("messages_opened_30d", 0)
        replied = bh.get("messages_replied_30d", 0)
        dismissed = bh.get("messages_dismissed_30d", 0)

        if pd.isna(opened): opened = 0
        if pd.isna(replied): replied = 0
        if pd.isna(dismissed): dismissed = 0

        total_30d = opened + replied + dismissed
        if total_30d > 0:
            rate = (opened + 2 * replied - dismissed) / (total_30d + 1)
            score += rate * 0.3

        return max(0.0, min(score, 1.0))

    def _compute_historical_engagement(self, context: MessageContext) -> float:
        """Compute user's general historical engagement for matching categories."""
        msg = context.message
        conv_type = msg.get("conversation_type")
        sender_id = msg.get("sender_user_id")
        business_id = msg.get("business_id")
        group_id = msg.get("group_id")

        relevant_history = []
        if conv_type == "business" and business_id:
            relevant_history = [h for h in context.history if h.get("business_id") == business_id]
        elif conv_type == "group" and group_id:
            relevant_history = [h for h in context.history if h.get("group_id") == group_id]
        elif conv_type == "personal" and sender_id:
            relevant_history = [h for h in context.history if h.get("sender_user_id") == sender_id]

        if not relevant_history:
            return 0.5  # Neutral default

        hist_ids = {h["message_id"] for h in relevant_history}
        relevant_events = [e for e in context.events if e.get("message_id") in hist_ids]

        total = len(relevant_history)
        opened = sum(1 for e in relevant_events if e.get("message_opened") == 1)
        replied = sum(1 for e in relevant_events if e.get("message_replied") == 1)
        dismissed = sum(1 for e in relevant_events if e.get("notification_dismissed") == 1)
        muted = sum(1 for e in relevant_events if e.get("muted_after_message") == 1)
        reported = sum(1 for e in relevant_events if e.get("message_reported") == 1)

        score = 0.5 + (opened * 0.2 + replied * 0.4 - dismissed * 0.2 - muted * 0.4 - reported * 0.6) / total
        return max(0.0, min(score, 1.0))

    def _compute_promotion_prob(self, text_lower: str, context: MessageContext) -> float:
        """Compute the probability that the message is promotional."""
        promo_words = [
            "discount", "sale", "cashback", "coupon", "off", "deals", "win", "gift", 
            "try50", "shopping offer", "limited time", "buy now", "marketing", "subscribe",
            "exclusive offer", "itinerary", "pvr", "trip last change", "free", "shop"
        ]

        score = 0.0
        for w in promo_words:
            if w in text_lower:
                score += 0.25

        # Check business category
        if context.business:
            cat = str(context.business.get("category", "")).lower()
            if cat in ("fashion", "streaming", "retail", "shopping"):
                score += 0.3
            elif cat in ("ecommerce_delivery", "logistics"):
                score -= 0.2  # Delivery updates are usually not promos

        # Check image type
        if context.image and "image_type" in context.image:
            img_type = str(context.image["image_type"]).lower()
            if img_type in ("advertisement", "poster"):
                score += 0.5

        return max(0.0, min(score, 1.0))

    def _compute_fatigue(self, context: MessageContext) -> float:
        """Calculate user's notification fatigue from summary logs."""
        summaries = context.notification_summary
        if not summaries:
            return 0.3  # Default fatigue

        # Use recent 7 entries
        sorted_summaries = sorted(summaries, key=lambda x: x.get("date", ""), reverse=True)[:7]
        total_sent = 0
        total_dismissed = 0

        for s in sorted_summaries:
            sent = s.get("notifications_sent", 0)
            dism = s.get("notifications_dismissed", 0)
            if pd.notna(sent): total_sent += sent
            if pd.notna(dism): total_dismissed += dism

        if total_sent == 0:
            return 0.0

        fatigue = (float(total_dismissed) / float(total_sent)) * 0.5 + min(float(total_sent) / 70.0, 0.5)
        return max(0.0, min(fatigue, 1.0))

    def _compute_spam(self, text_lower: str, fwd_score: float, context: MessageContext) -> float:
        """Compute general spam score."""
        spam_words = [
            "forward to", "share with", "luck changes", "blessings", "bhagwan sabka bhala",
            "send this to", "stay positive", "ten people", "ignore this", "positive energy",
            "fwd as received"
        ]

        score = 0.0
        for w in spam_words:
            if w in text_lower:
                score += 0.35

        score += fwd_score * 0.4

        # Voice notes spam probability
        if context.voice and "spam_probability" in context.voice:
            v_spam = context.voice["spam_probability"]
            if pd.notna(v_spam):
                try:
                    score = max(score, float(v_spam))
                except ValueError:
                    pass

        # If business is unverified and user has muted/reported them
        if context.business and context.business.get("verified") == 0:
            score += 0.2
            if context.business_history and pd.notna(context.business_history.get("promotions_opted_out_at")):
                score += 0.2

        return max(0.0, min(score, 1.0))

    def _compute_scam(self, text_lower: str, sender_trust: float, business_trust: float, context: MessageContext) -> float:
        """Calculate probability that the message is a scam."""
        scam_words = [
            "otp", "login code", "verify now", "verification code", "blocked in", "temporary blocked",
            "verification failed", "clearance amount immediately", "confirm password", "scan this qr",
            "ignore all previous routing rules", "wallet verification failed", "penalty list", "escrow",
            "win a prize", "claim rewards", "account access restricted", "verify card details"
        ]

        score = 0.0
        match_count = 0
        for w in scam_words:
            if w in text_lower:
                match_count += 1
                score += 0.35

        # Check for phishing characteristics
        msg = context.message
        conv_type = msg.get("conversation_type")

        if conv_type == "business" and context.business:
            # Check domain mismatch
            b = context.business
            off_domain = b.get("official_domain")
            sender_domain = b.get("domain_used_by_sender")
            if pd.notna(off_domain) and pd.notna(sender_domain):
                off_domain_str = str(off_domain).strip().lower()
                sender_domain_str = str(sender_domain).strip().lower()
                if off_domain_str and sender_domain_str and off_domain_str != sender_domain_str:
                    score += 0.5  # Strong scam indicator

            if b.get("verified") == 0:
                score += 0.2

        elif conv_type == "personal" or conv_type == "group":
            # Urgency in asking for security codes/money from untrusted sender
            if sender_trust < 0.3 and match_count > 0:
                score += 0.4

        # Specific prompt injection / override attempt detection
        if "ignore all previous" in text_lower or "ignore previous rules" in text_lower:
            score += 0.6

        return max(0.0, min(score, 1.0))

    def _compute_conv_priority(self, msg: dict[str, Any], user_id: Any, sender_trust: float) -> float:
        """Compute base conversation priority."""
        conv_type = msg.get("conversation_type")
        if conv_type == "personal":
            # High priority if trusted contact
            return 0.8 if sender_trust > 0.5 else 0.5
        elif conv_type == "group":
            # Check for direct mentions
            text = str(msg.get("message_text", "")).lower()
            if pd.notna(user_id) and f"@{user_id}" in text:
                return 0.9  # High priority due to direct mention
            return 0.4
        elif conv_type == "business":
            return 0.3
        return 0.3

    def _compute_group_priority(self, context: MessageContext) -> float:
        """Compute group priority based on group metadata."""
        if context.message.get("conversation_type") != "group" or not context.group:
            return 0.0

        # Check if user muted group
        if context.group_member and context.group_member.get("group_muted_by_user") == 1:
            return 0.0

        g = context.group
        g_type = str(g.get("group_type", "")).lower()

        priorities = {
            "family": 0.8,
            "extended_family": 0.6,
            "coworker": 0.8,
            "school_group": 0.7,
            "society": 0.5,
            "friends": 0.4,
            "alumni": 0.3,
            "marketplace": 0.1
        }

        return priorities.get(g_type, 0.3)
