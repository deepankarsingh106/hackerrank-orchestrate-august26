class IndexBuilder:
    def __init__(self, data):
        self.data = data

    def build(self):
        indexes = {}

        indexes["users"] = (
            self.data["users"]
            .set_index("user_id")
            .to_dict("index")
        )

        indexes["groups"] = (
            self.data["groups"]
            .set_index("group_id")
            .to_dict("index")
        )

        indexes["business"] = (
            self.data["business_accounts"]
            .set_index("business_id")
            .to_dict("index")
        )

        indexes["images"] = (
            self.data["images"]
            .set_index("image_id")
            .to_dict("index")
        )

        indexes["voice"] = (
            self.data["voice_notes"]
            .set_index("voice_note_id")
            .to_dict("index")
        )

        indexes["messages"] = (
            self.data["messages"]
            .set_index("message_id")
            .to_dict("index")
        )

        return indexes