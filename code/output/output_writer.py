"""Module for validating and writing the router output CSV."""

from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd
from typing import Any
from config import Action, MessageType

logger = logging.getLogger(__name__)


class OutputWriter:
    """Validates predictions format and writes output.csv."""

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)

    def write(self, predictions: list[dict[str, Any]], expected_message_ids: list[str]) -> None:
        """Validate list of prediction dicts and write them to output_path."""
        df = pd.DataFrame(predictions)

        # 1. Validate column existence and ordering
        required_columns = [
            "message_id",
            "action",
            "message_type",
            "reason",
            "confidence",
            "evidence_message_ids",
        ]
        
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Output is missing required column: '{col}'")

        # Reorder and filter columns
        df = df[required_columns]

        # 2. Validate row count and message ID completeness
        pred_ids = set(df["message_id"].dropna())
        expected_ids = set(expected_message_ids)

        if len(df) != len(expected_message_ids):
            raise ValueError(
                f"Row count mismatch! Expected {len(expected_message_ids)} rows, but got {len(df)} predictions."
            )

        missing_ids = expected_ids - pred_ids
        if missing_ids:
            raise ValueError(
                f"Output is missing predictions for {len(missing_ids)} expected message IDs: {list(missing_ids)[:5]}"
            )

        extra_ids = pred_ids - expected_ids
        if extra_ids:
            raise ValueError(
                f"Output contains {len(extra_ids)} unexpected message IDs not present in input messages: {list(extra_ids)[:5]}"
            )

        # 3. Validate values inside columns
        allowed_actions = {a.value for a in Action}
        allowed_types = {t.value for t in MessageType}

        for idx, row in df.iterrows():
            msg_id = row["message_id"]

            # Action validation
            act = row["action"]
            if act not in allowed_actions:
                raise ValueError(
                    f"Row {idx} (message_id: {msg_id}) has invalid action: '{act}'. Must be one of {allowed_actions}"
                )

            # Message Type validation
            m_type = row["message_type"]
            if m_type not in allowed_types:
                raise ValueError(
                    f"Row {idx} (message_id: {msg_id}) has invalid message_type: '{m_type}'. Must be one of {allowed_types}"
                )

            # Confidence validation
            conf = row["confidence"]
            try:
                conf_val = float(conf)
                if not (0.0 <= conf_val <= 1.0):
                    raise ValueError(f"Confidence {conf_val} is out of bounds [0.0, 1.0].")
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Row {idx} (message_id: {msg_id}) has invalid confidence value: '{conf}'. Error: {e}"
                )

            # Evidence IDs validation
            ev_ids = str(row["evidence_message_ids"]).strip()
            if not ev_ids:
                raise ValueError(
                    f"Row {idx} (message_id: {msg_id}) has empty evidence_message_ids. Write 'none' if no evidence exists."
                )

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to file
        df.to_csv(self.output_path, index=False)
        logger.info("Successfully validated and wrote %d prediction rows to %s", len(df), self.output_path)
        print(f"Predictions written to {self.output_path}")
