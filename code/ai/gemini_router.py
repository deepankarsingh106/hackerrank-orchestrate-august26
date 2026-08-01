"""Module for AI routing using the Gemini API."""

from __future__ import annotations

import json
import logging
from typing import Any
from config import Action, MessageType
from context.context_builder import MessageContext
from features.feature_engine import FeatureVector
from rules.rule_engine import RuleDecision

logger = logging.getLogger(__name__)


class GeminiRouter:
    """Invokes Gemini LLM for complex, low-confidence routing scenarios."""

    def __init__(self, api_key: str = "", model_name: str = "gemini-2.0-flash") -> None:
        self.api_key = api_key
        self.model_name = model_name

    def route(self, context: MessageContext, features: FeatureVector) -> RuleDecision | None:
        """Call Gemini to make a routing decision. Returns None if API key is missing or call fails."""
        if not self.api_key.strip():
            logger.debug("Gemini API key is not configured. Falling back to Rule Engine.")
            return None

        try:
            return self._route_with_gemini(context, features)
        except Exception as e:
            logger.exception("Error during Gemini routing: %s", e)
            return None

    def _route_with_gemini(self, context: MessageContext, features: FeatureVector) -> RuleDecision | None:
        """Construct prompt and query Gemini API."""
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model_name)

        # Build context metadata details for the prompt
        msg = context.message
        user = context.user
        group = context.group or {}
        business = context.business or {}
        business_history = context.business_history or {}

        # Precompute media text if available
        media_text = ""
        if context.image and context.image.get("text"):
            media_text += f"\nImage OCR Text: {context.image['text']}"
            media_text += f"\nImage Category: {context.image.get('image_type', 'unknown')}"
        if context.voice and context.voice.get("transcript"):
            media_text += f"\nVoice Note Transcript: {context.voice['transcript']}"

        prompt = f"""You are a WhatsApp Message Notification Router.
Decide the appropriate routing action for the following incoming message.

=== INCOMING MESSAGE ===
Message ID: {msg.get("message_id")}
Sender ID: {msg.get("sender_user_id", "N/A")} (or Business ID: {msg.get("business_id", "N/A")} if business)
Conversation Type: {msg.get("conversation_type")}
Message Text: {msg.get("message_text", "N/A")}{media_text}
Forward Count: {msg.get("forwarded_count", 0)}

=== RECEIVING USER PROFILE ===
User ID: {user.get("user_id")}
DND Active Now: {features.dnd_active} (1 = yes, 0 = no)
Fatigue Score: {features.notification_fatigue}

=== CONTEXT & BEHAVIOR ===
Group Details (if group): Type={group.get("group_type")}, Name={group.get("group_name")}
Group Member Muted: {context.group_member.get("group_muted_by_user", 0) if context.group_member else 0}
Business Category: {business.get("category", "N/A")}
Business Brand: {business.get("brand_name", "N/A")}
Business Verified: {business.get("verified", "N/A")}
Domain Mismatch: {1 if business.get("official_domain") != business.get("domain_used_by_sender") else 0}
User Allows Promotions: {business_history.get("allows_promotions", "N/A")}
Opted Out of Business Promotions: {1 if business_history.get("promotions_opted_out_at") else 0}

=== COMPUTED FEATURES ===
Urgency Score: {features.urgency_score}
Sender Trust Score: {features.sender_trust_score}
Business Trust Score: {features.business_trust_score}
Business Relationship Score: {features.business_relationship_score}
Historical Engagement Score: {features.historical_engagement_score}
Promotion Probability: {features.promotion_probability}
Spam Score: {features.spam_score}
Scam/Phishing Score: {features.scam_score}
Conversation Priority: {features.conversation_priority}
Group Priority: {features.group_priority}

=== INSTRUCTIONS ===
Determine:
1. 'action':
   - 'notify': Important/urgent messages that should interrupt the user now.
   - 'digest': Safe, low-priority messages that can wait (promotions, general personal chat, events).
   - 'mute': Repetitive, unwanted, low-value, muted groups (no direct mention), spam, scam, or unverified promos.
2. 'message_type': Choose EXACTLY one: 'personal', 'urgent', 'event', 'payment', 'business_update', 'promotion', 'greeting', 'forward', 'spam', 'scam', 'unknown'.
3. 'reason': Short, human-readable sentence justifying the decision.
4. 'confidence': Float between 0.0 and 1.0.

Return ONLY a valid JSON object matching this schema:
{{
  "action": "notify" | "digest" | "mute",
  "message_type": "...",
  "reason": "...",
  "confidence": 0.0
}}
"""

        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                response_mime_type="application/json",
            )
        )
        response_text = response.text.strip()

        # Clean potential markdown formatting
        if response_text.startswith("```json"):
            response_text = response_text.split("```json", 1)[1]
        if response_text.endswith("```"):
            response_text = response_text.rsplit("```", 1)[0]
        response_text = response_text.strip()

        try:
            res_dict = json.loads(response_text)
            action_str = str(res_dict["action"]).strip().lower()
            type_str = str(res_dict["message_type"]).strip().lower()
            
            # Map strings to enums safely
            action = Action(action_str)
            message_type = MessageType(type_str)

            return RuleDecision(
                action=action,
                message_type=message_type,
                reason=str(res_dict["reason"]),
                confidence=float(res_dict["confidence"]),
            )
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error("Failed to parse Gemini decision response: %s (Error: %s)", response.text, e)

        return None
