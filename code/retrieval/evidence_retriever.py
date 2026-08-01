"""Module for retrieving historical message IDs as evidence for routing decisions."""

from __future__ import annotations

import logging
from typing import Any
from context.context_builder import MessageContext

logger = logging.getLogger(__name__)


class EvidenceRetriever:
    """Selects relevant historical message IDs that justify a routing action."""

    def __init__(self) -> None:
        pass

    def retrieve_evidence(self, context: MessageContext) -> str:
        """Analyze message history and return semicolon-separated evidence IDs or 'none'."""
        msg = context.message
        history = context.history
        events = context.events

        if not history:
            return "none"

        conv_type = msg.get("conversation_type")
        sender_id = msg.get("sender_user_id")
        business_id = msg.get("business_id")
        group_id = msg.get("group_id")

        # Precompute current message keywords for similarity check
        msg_text = str(msg.get("message_text", "")).lower()
        curr_words = set(msg_text.split())

        # Keywords for categorization
        promo_kws = {"sale", "discount", "offer", "coupon", "off", "deals", "win", "gift", "try50", "pvr", "trip", "itinerary"}
        scam_kws = {"otp", "login code", "verify now", "verification", "blocked", "password", "qr", "failed", "penalty"}
        greet_kws = {"morning", "blessed", "blessings", "smile", "positive", "vibes", "hello", "hi"}

        is_promo = any(w in curr_words for w in promo_kws)
        is_scam = any(w in curr_words for w in scam_kws)
        is_greet = any(w in curr_words for w in greet_kws)

        # Map events for fast lookup
        event_map = {e["message_id"]: e for e in events if "message_id" in e}

        scored_history: list[tuple[str, float]] = []

        for h in history:
            h_id = h.get("message_id")
            if not h_id:
                continue

            score = 0.0

            # 1. Conversation type match
            if h.get("conversation_type") == conv_type:
                score += 1.0

            # 2. Sender/business alignment
            if conv_type == "business" and business_id and h.get("business_id") == business_id:
                score += 5.0
            elif conv_type == "group" and group_id and h.get("group_id") == group_id:
                score += 3.0
                if sender_id and h.get("sender_user_id") == sender_id:
                    score += 2.0
            elif conv_type == "personal" and sender_id and h.get("sender_user_id") == sender_id:
                score += 5.0

            # 3. Text content similarity
            h_text = str(h.get("message_text", "")).lower()
            h_words = set(h_text.split())
            
            # Jaccard word overlap
            overlap = len(curr_words.intersection(h_words))
            if overlap > 0:
                score += min(overlap * 1.5, 6.0)

            # Topic matching
            h_promo = any(w in h_words for w in promo_kws)
            h_scam = any(w in h_words for w in scam_kws)
            h_greet = any(w in h_words for w in greet_kws)

            if is_promo and h_promo:
                score += 3.0
            if is_scam and h_scam:
                score += 4.0
            if is_greet and h_greet:
                score += 3.0

            # 4. User engagement signal (events)
            ev = event_map.get(h_id)
            if ev:
                score += 1.0  # Present in events is a plus
                if ev.get("message_replied") == 1:
                    score += 2.0
                if ev.get("muted_after_message") == 1:
                    score += 3.0
                if ev.get("message_reported") == 1:
                    score += 4.0
                if ev.get("notification_dismissed") == 1:
                    score += 1.0

            if score > 3.0:  # Threshold to be considered relevant evidence
                scored_history.append((h_id, score))

        if not scored_history:
            return "none"

        # Sort by score descending, then by message_id descending (newer/higher IDs first)
        scored_history.sort(key=lambda x: (x[1], x[0]), reverse=True)

        # Select top up to 2 evidence IDs
        top_ids = [item[0] for item in scored_history[:2]]

        # Keep output order deterministic (sort alphabetically/numerically)
        top_ids.sort()

        return ";".join(top_ids)
