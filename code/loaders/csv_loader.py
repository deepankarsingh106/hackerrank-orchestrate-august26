import pandas as pd
from pathlib import Path

class CSVLoader:

    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)

    def _load_csv(self, filename):
        file_path = self.dataset_path / filename

        if not file_path.exists():           
            raise FileNotFoundError(f"{filename} not found!")

        return pd.read_csv(file_path)

    def load_messages(self):
        return self._load_csv("messages.csv")

    def load_users(self):
        return self._load_csv("users.csv")

    def load_groups(self):
        return self._load_csv("groups.csv")

    def load_group_members(self):
        return self._load_csv("group_members.csv")

    def load_business_accounts(self):
        return self._load_csv("business_accounts.csv")

    def load_business_history(self):
        return self._load_csv("user_business_history.csv")

    def load_message_history(self):
        return self._load_csv("message_history.csv")

    def load_message_events(self):
        return self._load_csv("message_events.csv")

    def load_images(self):
        return self._load_csv("images.csv")

    def load_voice_notes(self):
        return self._load_csv("voice_notes.csv")

    def load_notification_summary(self):
        return self._load_csv("daily_notification_summary.csv")
        
    def load_everything(self):
        return {
            "messages": self.load_messages(),
            "users": self.load_users(),
            "groups": self.load_groups(),
            "group_members": self.load_group_members(),
            "business_accounts": self.load_business_accounts(),
            "business_history": self.load_business_history(),
            "message_history": self.load_message_history(),
            "message_events": self.load_message_events(),
            "images": self.load_images(),
            "voice_notes": self.load_voice_notes(),
            "notification_summary": self.load_notification_summary(),
        }
