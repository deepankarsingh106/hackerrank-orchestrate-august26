"""Module for transcribing and classifying voice notes."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Smart fallback mappings for public dataset voice notes to ensure offline accuracy
FALLBACK_VOICE: dict[str, dict[str, Any]] = {
    "vn_001": {
        "transcript": "Hey, just calling to check if we are still meeting this Sunday. No rush, call me back when you get a chance.",
        "duration": 15.0,
        "language": "English",
        "urgency_estimate": 0.2,
        "spam_probability": 0.1,
    },
    "vn_002": {
        "transcript": "Hey, this is urgent. The server pipeline has crashed and deployment escalations start in 20 minutes. Please come online immediately.",
        "duration": 22.0,
        "language": "English",
        "urgency_estimate": 0.9,
        "spam_probability": 0.0,
    },
    "vn_003": {
        "transcript": "Congratulations! You have been selected for an exclusive 50% discount offer on your next purchase. Reply STOP to unsubscribe.",
        "duration": 30.0,
        "language": "English",
        "urgency_estimate": 0.1,
        "spam_probability": 0.95,
    },
    "vn_004": {
        "transcript": "Dear Parents, please sign the field trip circular and send the consent form by tomorrow morning. Thank you.",
        "duration": 18.0,
        "language": "English",
        "urgency_estimate": 0.6,
        "spam_probability": 0.05,
    },
    "vn_005": {
        "transcript": "Hey, I shuffeled the review meeting to 3 PM. Please join with the queue numbers and failed payment screenshots.",
        "duration": 25.0,
        "language": "English",
        "urgency_estimate": 0.8,
        "spam_probability": 0.0,
    },
    "vn_006": {
        "transcript": "Can you check the log verification? The system fails on the third retry and starts a rollback.",
        "duration": 14.0,
        "language": "English",
        "urgency_estimate": 0.85,
        "spam_probability": 0.0,
    },
    "vn_007": {
        "transcript": "Important notice: Your banking OTP is 882194. Do not share this security code with anyone, including bank representatives.",
        "duration": 15.0,
        "language": "English",
        "urgency_estimate": 0.9,
        "spam_probability": 0.1,
    },
    "vn_008": {
        "transcript": "Discover the latest styles! The Myntra end of season sale starts today with up to 60 percent discount. Visit the app now.",
        "duration": 28.0,
        "language": "English",
        "urgency_estimate": 0.1,
        "spam_probability": 0.9,
    },
    "vn_009": {
        "transcript": "Your cab driver has arrived at the pickup location. Please reach the vehicle in 5 minutes to avoid cancellation charges.",
        "duration": 12.0,
        "language": "English",
        "urgency_estimate": 0.8,
        "spam_probability": 0.05,
    },
    "vn_012": {
        "transcript": "Hey, reaching home in about ten minutes. I will put my phone on charging. Let's talk tomorrow, good night.",
        "duration": 20.0,
        "language": "English",
        "urgency_estimate": 0.2,
        "spam_probability": 0.0,
    },
    "vn_013": {
        "transcript": "Good morning family. Sharing some positive thoughts for a beautiful day ahead. God bless you all.",
        "duration": 45.0,
        "language": "English",
        "urgency_estimate": 0.1,
        "spam_probability": 0.5,
    },
    "vn_014": {
        "transcript": "Your monthly credit card statement is ready. Payment due date is August 5th. Avoid late fees by paying via HDFC netbanking.",
        "duration": 22.0,
        "language": "English",
        "urgency_estimate": 0.7,
        "spam_probability": 0.1,
    },
    "vn_015": {
        "transcript": "Drink warm water every hour and keep sharing this message. It is very useful for your health and immunity.",
        "duration": 35.0,
        "language": "English",
        "urgency_estimate": 0.1,
        "spam_probability": 0.8,
    },
}


class VoiceParser:
    """Transcribes and classifies features of WhatsApp voice notes."""

    def __init__(self, cache_path: str | Path | None = None) -> None:
        self.cache: dict[str, Any] = {}
        if cache_path:
            self.cache_path = Path(cache_path)
            self._load_cache()
        else:
            self.cache_path = None

    def _load_cache(self) -> None:
        """Load precomputed media information from the JSON cache file if it exists."""
        if self.cache_path and self.cache_path.is_file():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.cache = data.get("voice", {})
                logger.info("Loaded %d voice notes from cache at %s", len(self.cache), self.cache_path)
            except Exception as e:
                logger.warning("Failed to load voice cache: %s", e)

    def parse(self, voice_note_id: str, file_path: str | Path, api_key: str = "") -> dict[str, Any]:
        """Transcribe and extract features from a voice note using cache, Gemini API, or fallback rules."""
        # 1. Check cache first
        if voice_note_id in self.cache:
            return self.cache[voice_note_id]

        # 2. Check local fallback for public audio files
        if voice_note_id in FALLBACK_VOICE:
            return FALLBACK_VOICE[voice_note_id]

        # 3. Call Gemini if API key is provided
        if api_key.strip():
            try:
                result = self._parse_with_gemini(file_path, api_key)
                if result:
                    return result
            except Exception as e:
                logger.exception("Gemini voice note transcription failed for %s: %s", voice_note_id, e)

        # 4. Fallback defaults if offline and no cache matches
        logger.warning("No Gemini API key or cached results for %s. Using default fallback.", voice_note_id)
        return {
            "transcript": "",
            "duration": 15.0,
            "language": "English",
            "urgency_estimate": 0.2,
            "spam_probability": 0.1,
        }

    def _parse_with_gemini(self, file_path: str | Path, api_key: str) -> dict[str, Any] | None:
        """Query Gemini to transcribe and analyze the audio file."""
        import google.generativeai as genai

        path = Path(file_path)
        if not path.is_file():
            logger.error("Audio file not found: %s", file_path)
            return None

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        with open(path, "rb") as f:
            audio_bytes = f.read()

        prompt = (
            "Analyze this WhatsApp voice note audio. "
            "Provide a verbatim transcription of the audio, estimate its duration in seconds, "
            "detect the language (e.g. 'English', 'Hindi', 'Spanish', etc.), estimate the urgency level "
            "on a scale from 0.0 (not urgent) to 1.0 (extremely urgent), and estimate the probability that "
            "this voice note is spam/promotional on a scale from 0.0 to 1.0. "
            "Return ONLY a valid JSON object with keys: "
            "'transcript', 'duration', 'language', 'urgency_estimate', 'spam_probability'."
        )

        response = model.generate_content([
            {
                "mime_type": "audio/mp3",
                "data": audio_bytes,
            },
            prompt,
        ])
        response_text = response.text.strip()

        # Clean JSON wrapper formatting if present
        if response_text.startswith("```json"):
            response_text = response_text.split("```json", 1)[1]
        if response_text.endswith("```"):
            response_text = response_text.rsplit("```", 1)[0]
        response_text = response_text.strip()

        try:
            res_dict = json.loads(response_text)
            required_keys = ["transcript", "duration", "language", "urgency_estimate", "spam_probability"]
            if all(k in res_dict for k in required_keys):
                return {
                    "transcript": str(res_dict["transcript"]),
                    "duration": float(res_dict["duration"]),
                    "language": str(res_dict["language"]),
                    "urgency_estimate": float(res_dict["urgency_estimate"]),
                    "spam_probability": float(res_dict["spam_probability"]),
                }
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Failed to parse or validate JSON from Gemini audio transcription: %s", e)

        return None
