"""Module for evaluating router pipeline performance against sample messages."""

from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd
from typing import Any

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluates pipeline prediction accuracy and calibration against solved samples."""

    def __init__(self, sample_csv_path: str | Path) -> None:
        self.sample_csv_path = Path(sample_csv_path)

    def evaluate(self, predict_func: Any) -> dict[str, Any]:
        """Run predictions on all sample messages and compute accuracy metrics."""
        print(f"Loading sample validation data from {self.sample_csv_path}...")
        df_samples = pd.read_csv(self.sample_csv_path)

        total_rows = len(df_samples)
        if total_rows == 0:
            print("No sample messages found for evaluation.")
            return {}

        correct_action = 0
        correct_type = 0
        mismatches = []

        action_actual_counts: dict[str, int] = {}
        action_pred_counts: dict[str, int] = {}

        # Tracking confidence calibration
        correct_confidences = []
        incorrect_confidences = []

        print(f"Evaluating {total_rows} sample messages...")
        for idx, row in df_samples.iterrows():
            msg_id = str(row["message_id"])
            gt_action = str(row["action"]).strip().lower()
            gt_type = str(row["message_type"]).strip().lower()

            # Run prediction through the provided pipeline prediction function
            try:
                pred = predict_func(msg_id, row)
                pred_action = str(pred["action"]).strip().lower()
                pred_type = str(pred["message_type"]).strip().lower()
                pred_conf = float(pred["confidence"])
            except Exception as e:
                logger.exception("Failed to run prediction on sample message %s: %s", msg_id, e)
                print(f"  Error predicting message {msg_id}: {e}")
                continue

            # Update count metrics
            action_actual_counts[gt_action] = action_actual_counts.get(gt_action, 0) + 1
            action_pred_counts[pred_action] = action_pred_counts.get(pred_action, 0) + 1

            action_match = (gt_action == pred_action)
            type_match = (gt_type == pred_type)

            if action_match:
                correct_action += 1
                correct_confidences.append(pred_conf)
            else:
                incorrect_confidences.append(pred_conf)

            if type_match:
                correct_type += 1

            if not action_match or not type_match:
                mismatches.append({
                    "message_id": msg_id,
                    "text": str(row.get("message_text", ""))[:60].replace("\n", " "),
                    "actual_action": gt_action,
                    "pred_action": pred_action,
                    "actual_type": gt_type,
                    "pred_type": pred_type,
                    "reason": pred["reason"],
                    "confidence": pred_conf,
                })

        action_accuracy = correct_action / total_rows
        type_accuracy = correct_type / total_rows

        avg_correct_conf = sum(correct_confidences) / len(correct_confidences) if correct_confidences else 0.0
        avg_incorrect_conf = sum(incorrect_confidences) / len(incorrect_confidences) if incorrect_confidences else 0.0

        print("\n========================================================")
        print("EVALUATION REPORT")
        print("========================================================")
        print(f"Total evaluated messages : {total_rows}")
        print(f"Action Accuracy          : {action_accuracy * 100:.2f}% ({correct_action}/{total_rows})")
        print(f"Message Type Accuracy    : {type_accuracy * 100:.2f}% ({correct_type}/{total_rows})")
        print(f"Avg Correct Confidence   : {avg_correct_conf:.4f}")
        print(f"Avg Incorrect Confidence : {avg_incorrect_conf:.4f}")
        print(f"Confidence Gap           : {avg_correct_conf - avg_incorrect_conf:.4f}")
        print("--------------------------------------------------------")
        print("Actual Action distribution:")
        for act, count in action_actual_counts.items():
            print(f"  {act:<8} : {count}")
        print("Predicted Action distribution:")
        for act, count in action_pred_counts.items():
            print(f"  {act:<8} : {count}")
        
        if mismatches:
            print("--------------------------------------------------------")
            print(f"Sample Mismatch Detail (showing up to 10 of {len(mismatches)} errors):")
            for m in mismatches[:10]:
                print(f"ID: {m['message_id']}")
                print(f"  Txt  : {m['text']}")
                print(f"  Act  : Expected [{m['actual_action']}], Got [{m['pred_action']}]")
                print(f"  Type : Expected [{m['actual_type']}], Got [{m['pred_type']}]")
                print(f"  Conf : {m['confidence']:.3f} | Reason: {m['reason']}")
                print()
        else:
            print("--------------------------------------------------------")
            print("PERFECT MATCH! No classification mismatches found on sample messages.")

        print("========================================================\n")

        return {
            "total_rows": total_rows,
            "action_accuracy": action_accuracy,
            "type_accuracy": type_accuracy,
            "mismatch_count": len(mismatches),
            "avg_correct_confidence": avg_correct_conf,
            "avg_incorrect_confidence": avg_incorrect_conf,
        }
