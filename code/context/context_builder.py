import pandas as pd


class ContextBuilder:
    def __init__(self, data):
        self.data = data

        # DataFrames
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

    def build_context(self, message_id):
        message = self.messages[self.messages["message_id"] == message_id].iloc[0]

        user = self.users[
            self.users["user_id"] == message["user_id"]
        ]

        group = self.groups[
            self.groups["group_id"] == message["group_id"]
        ]

        group_member = self.group_members[
            (self.group_members["user_id"] == message["user_id"]) &
            (self.group_members["group_id"] == message["group_id"])
        ]

        business = self.business_accounts[
            self.business_accounts["business_id"] == message["business_id"]
        ]

        business_history = self.business_history[
            (self.business_history["user_id"] == message["user_id"]) &
            (self.business_history["business_id"] == message["business_id"])
        ]

        history = self.message_history[
            self.message_history["user_id"] == message["user_id"]
        ]

        events = self.message_events[
            self.message_events["message_id"].isin(history["message_id"])
        ]

        notification_summary = self.notification_summary[
            self.notification_summary["user_id"] == message["user_id"]
        ]

        image = pd.DataFrame()

        if message["media_type"] == "image":
            image = self.images[
                self.images["image_id"] == message["media_id"]
            ]

        voice = pd.DataFrame()

        if message["media_type"] == "voice":
            voice = self.voice_notes[
                self.voice_notes["voice_note_id"] == message["media_id"]
            ]

        return {
            "message": message.to_dict(),
            "user": user.to_dict("records"),
            "group": group.to_dict("records"),
            "group_member": group_member.to_dict("records"),
            "business": business.to_dict("records"),
            "business_history": business_history.to_dict("records"),
            "history": history.to_dict("records"),
            "events": events.to_dict("records"),
            "notification_summary": notification_summary.to_dict("records"),
            "image": image.to_dict("records"),
            "voice": voice.to_dict("records"),
        }