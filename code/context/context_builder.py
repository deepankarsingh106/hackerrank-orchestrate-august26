"""Module for building the structured context of an incoming message."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MessageContext:
    """Structured context combining message details, user behavior, and historical logs."""

    message: dict[str, Any]
    user: dict[str, Any]
    group: Optional[dict[str, Any]]
    group_member: Optional[dict[str, Any]]
    business: Optional[dict[str, Any]]
    business_history: Optional[dict[str, Any]]
    history: list[dict[str, Any]]
    events: list[dict[str, Any]]
    notification_summary: list[dict[str, Any]]
    image: Optional[dict[str, Any]]
    voice: Optional[dict[str, Any]]
    features: Optional[Any] = None  # To be populated by FeatureEngine
    evidence_message_ids: str = "none"


class ContextBuilder:
    """Builds and resolves context from DataFrames for a specific message."""

    def __init__(self, data: dict[str, pd.DataFrame]) -> None:
        self.data = data
        self.messages = data["messages"]
        self.users = data["users"]
        self.groups = data["groups"]
        self.group_members = data["group_members"]
        self.business_accounts = data["business_accounts"]
        self.business_history = data["business_history"]
        self.message_history = data["message_history"]
        self.message_events = data["message_events"]
        self.images = data["images"]
        self.voice_notes = data["voice_notes"]
        self.notification_summary = data["notification_summary"]

    def build_context(self, message_id: str) -> MessageContext:
        """Resolve message relationships and return a MessageContext object."""
        # Find current message
        message_rows = self.messages[self.messages["message_id"] == message_id]
        if message_rows.empty:
            raise KeyError(f"Message ID '{message_id}' not found in messages dataset.")
        message = message_rows.iloc[0]
        message_dict = message.to_dict()

        # Resolve user
        user_rows = self.users[self.users["user_id"] == message["user_id"]]
        user_dict = user_rows.iloc[0].to_dict() if not user_rows.empty else {}

        # Resolve group
        group_dict = None
        group_member_dict = None
        if pd.notna(message["group_id"]) and str(message["group_id"]).strip() != "":
            group_id = message["group_id"]
            group_rows = self.groups[self.groups["group_id"] == group_id]
            if not group_rows.empty:
                group_dict = group_rows.iloc[0].to_dict()

            member_rows = self.group_members[
                (self.group_members["group_id"] == group_id)
                & (self.group_members["user_id"] == message["user_id"])
            ]
            if not member_rows.empty:
                group_member_dict = member_rows.iloc[0].to_dict()

        # Resolve business
        business_dict = None
        business_history_dict = None
        if pd.notna(message["business_id"]) and str(message["business_id"]).strip() != "":
            business_id = message["business_id"]
            business_rows = self.business_accounts[
                self.business_accounts["business_id"] == business_id
            ]
            if not business_rows.empty:
                business_dict = business_rows.iloc[0].to_dict()

            bh_rows = self.business_history[
                (self.business_history["business_id"] == business_id)
                & (self.business_history["user_id"] == message["user_id"])
            ]
            if not bh_rows.empty:
                business_history_dict = bh_rows.iloc[0].to_dict()

        # Resolve history & events for this specific user
        user_id = message["user_id"]
        history_df = self.message_history[self.message_history["user_id"] == user_id]
        history_list = history_df.to_dict("records")

        # Resolve events related to user's history
        events_df = self.message_events[
            self.message_events["message_id"].isin(history_df["message_id"])
        ]
        events_list = events_df.to_dict("records")

        # Resolve daily notification summaries for user
        summary_df = self.notification_summary[
            self.notification_summary["user_id"] == user_id
        ]
        summary_list = summary_df.to_dict("records")

        # Resolve image metadata
        image_dict = None
        if message["media_type"] == "image" and pd.notna(message["media_id"]):
            image_id = message["media_id"]
            image_rows = self.images[self.images["image_id"] == image_id]
            if not image_rows.empty:
                image_dict = image_rows.iloc[0].to_dict()

        # Resolve voice note metadata
        voice_dict = None
        if message["media_type"] == "voice" and pd.notna(message["media_id"]):
            voice_id = message["media_id"]
            voice_rows = self.voice_notes[self.voice_notes["voice_note_id"] == voice_id]
            if not voice_rows.empty:
                voice_dict = voice_rows.iloc[0].to_dict()

        return MessageContext(
            message=message_dict,
            user=user_dict,
            group=group_dict,
            group_member=group_member_dict,
            business=business_dict,
            business_history=business_history_dict,
            history=history_list,
            events=events_list,
            notification_summary=summary_list,
            image=image_dict,
            voice=voice_dict,
        )