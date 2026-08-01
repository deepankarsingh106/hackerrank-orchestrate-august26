"""Main execution entry point for the Message Notification Router pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any
import pandas as pd

from config import load_config, validate_required_paths
from loaders.csv_loader import CSVLoader
from context.context_builder import ContextBuilder, MessageContext
from features.feature_engine import FeatureEngine
from media.image_parser import ImageParser
from media.voice_parser import VoiceParser
from retrieval.evidence_retriever import EvidenceRetriever
from rules.rule_engine import RuleEngine
from ai.gemini_router import GeminiRouter
from utils.confidence_calibrator import ConfidenceCalibrator
from output.output_writer import OutputWriter
from evaluation.evaluator import Evaluator

logger = logging.getLogger("router_main")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="WhatsApp Message Notification Router pipeline."
    )
    parser.add_argument(
        "--mode",
        choices=["run", "evaluate"],
        default="run",
        help="Pipeline execution mode: 'run' to generate output.csv (default), or 'evaluate' to run validation.",
    )
    return parser.parse_args()


def main() -> int:
    """Orchestrate pipeline loading, execution, evaluation, and output generation."""
    args = parse_args()

    # 1. Load configuration
    try:
        config = load_config(configure_logs=True)
        validate_required_paths(config)
    except Exception as e:
        print(f"Configuration loading failed: {e}")
        return 1

    # 2. Load dataset
    print("Loading datasets...")
    try:
        loader = CSVLoader(config.paths.dataset_root)
        data = loader.load_everything()
        
        # Load sample validation messages separately for evaluation
        sample_messages_df = pd.read_csv(config.paths.sample_messages_file)
        data["sample_messages"] = sample_messages_df
    except Exception as e:
        logger.exception("Failed to load CSV files: %s", e)
        print(f"Data loading failed: {e}")
        return 1

    # 3. Instantiate pipeline components
    cache_path = config.paths.dataset_root / "media_cache.json"
    context_builder = ContextBuilder(data)
    image_parser = ImageParser(cache_path)
    voice_parser = VoiceParser(cache_path)
    feature_engine = FeatureEngine()
    evidence_retriever = EvidenceRetriever()
    rule_engine = RuleEngine()
    gemini_router = GeminiRouter(
        api_key=config.gemini.api_key, model_name=config.gemini.model_name
    )
    confidence_calibrator = ConfidenceCalibrator()
    output_writer = OutputWriter(config.paths.output_file)

    def predict_message(message_id: str, row: pd.Series) -> dict[str, Any]:
        """Process a single message through all pipeline stages."""
        # 1. Resolve basic relationships
        context = context_builder.build_context(message_id)

        # 2. Parse visual/auditory media if present
        api_key = config.gemini.api_key
        if context.message.get("media_type") == "image" and context.image:
            img_path = config.paths.resolve_media_file(context.image["file_path"])
            media_res = image_parser.parse(context.message["media_id"], img_path, api_key)
            context.image.update(media_res)
        elif context.message.get("media_type") == "voice" and context.voice:
            voice_path = config.paths.resolve_media_file(context.voice["file_path"])
            media_res = voice_parser.parse(context.message["media_id"], voice_path, api_key)
            context.voice.update(media_res)

        # 3. Compute deterministic features
        features = feature_engine.compute_features(context)
        context.features = features

        # 4. Retrieve historical evidence message IDs
        evidence_ids = evidence_retriever.retrieve_evidence(context)
        context.evidence_message_ids = evidence_ids

        # 5. Run Rule Engine routing decision
        decision = rule_engine.evaluate(context, features)

        # 6. Fallback to Gemini Router for low confidence rule matches
        if decision.confidence < config.confidence.rule_engine_auto:
            gemini_decision = gemini_router.route(context, features)
            if gemini_decision:
                decision = gemini_decision

        # 7. Apply confidence calibration
        calibrated = confidence_calibrator.calibrate(decision, features)

        return {
            "message_id": message_id,
            "action": calibrated.action.value,
            "message_type": calibrated.message_type.value,
            "reason": calibrated.reason,
            "confidence": calibrated.confidence,
            "evidence_message_ids": evidence_ids,
        }

    # 4. Mode Execution
    if args.mode == "evaluate":
        # Direct the context builder to search validation samples instead of messages.csv
        context_builder.messages = data["sample_messages"]
        
        evaluator = Evaluator(config.paths.sample_messages_file)
        evaluator.evaluate(predict_message)

    else:
        # Run prediction on messages.csv and write output.csv
        context_builder.messages = data["messages"]
        messages_df = data["messages"]
        total_msgs = len(messages_df)
        
        print(f"Processing {total_msgs} incoming messages...")
        predictions = []
        
        # Display simple inline progress indicator
        for idx, row in messages_df.iterrows():
            msg_id = str(row["message_id"])
            if idx % 20 == 0 and idx > 0:
                print(f"  Processed {idx}/{total_msgs} messages...")
            
            try:
                pred = predict_message(msg_id, row)
                predictions.append(pred)
            except Exception as e:
                logger.exception("Pipeline failed on message ID %s: %s", msg_id, e)
                print(f"CRITICAL: Message routing failed for {msg_id}: {e}")
                return 1

        print(f"Completed routing for all {total_msgs} messages. Validating and writing results...")
        
        # Write and validate final output.csv
        expected_ids = messages_df["message_id"].tolist()
        try:
            output_writer.write(predictions, expected_ids)
        except Exception as e:
            print(f"Output validation failed: {e}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())