"""Utility script to precompute OCR and transcriptions for all media assets using Gemini."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Add project root to path so we can import code modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import load_config
from loaders.csv_loader import CSVLoader
from media.image_parser import ImageParser
from media.voice_parser import VoiceParser


def main() -> int:
    """Precompute all media assets and write them to dataset/media_cache.json."""
    print("Initializing media precomputation utility...")
    
    try:
        config = load_config(configure_logs=False)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1

    api_key = config.gemini.api_key
    if not api_key:
        print("WARNING: GEMINI_API_KEY is not set in environment or .env file.")
        print("This script requires a valid Gemini API key to process files online.")
        print("Exiting without precomputing.")
        return 1

    print(f"Using Gemini Model: {config.gemini.model_name}")
    print(f"Dataset root      : {config.paths.dataset_root}")

    # Load media files
    try:
        loader = CSVLoader(config.paths.dataset_root)
        images_df = loader.load_images()
        voice_df = loader.load_voice_notes()
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        return 1

    cache_file = config.paths.dataset_root / "media_cache.json"
    
    # Load existing cache if any
    cache_data = {"images": {}, "voice": {}}
    if cache_file.is_file():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                cache_data["images"] = loaded.get("images", {})
                cache_data["voice"] = loaded.get("voice", {})
            print(f"Loaded existing cache with {len(cache_data['images'])} images and {len(cache_data['voice'])} voice notes.")
        except Exception as e:
            print(f"Warning: Could not read existing cache file: {e}")

    image_parser = ImageParser()
    voice_parser = VoiceParser()

    # Process images
    print("\n--- Processing Images ---")
    image_count = 0
    for _, row in images_df.iterrows():
        img_id = str(row["image_id"])
        rel_path = str(row["file_path"])
        abs_path = config.paths.resolve_media_file(rel_path)

        if img_id in cache_data["images"]:
            print(f"Image {img_id} already cached, skipping.")
            continue

        print(f"Processing image {img_id} ({rel_path})...")
        if not abs_path.is_file():
            print(f"  Error: File not found at {abs_path}")
            continue

        try:
            res = image_parser._parse_with_gemini(abs_path, api_key)
            if res:
                cache_data["images"][img_id] = res
                image_count += 1
                print(f"  Successfully processed: {res['image_type']}")
        except Exception as e:
            print(f"  Failed to process image {img_id}: {e}")

    # Process voice notes
    print("\n--- Processing Voice Notes ---")
    voice_count = 0
    for _, row in voice_df.iterrows():
        vn_id = str(row["voice_note_id"])
        rel_path = str(row["file_path"])
        abs_path = config.paths.resolve_media_file(rel_path)

        if vn_id in cache_data["voice"]:
            print(f"Voice note {vn_id} already cached, skipping.")
            continue

        print(f"Processing voice note {vn_id} ({rel_path})...")
        if not abs_path.is_file():
            print(f"  Error: File not found at {abs_path}")
            continue

        try:
            res = voice_parser._parse_with_gemini(abs_path, api_key)
            if res:
                cache_data["voice"][vn_id] = res
                voice_count += 1
                print(f"  Successfully processed transcript. Language: {res['language']}")
        except Exception as e:
            print(f"  Failed to process voice note {vn_id}: {e}")

    # Save results
    if image_count > 0 or voice_count > 0:
        try:
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
            print(f"\nCache successfully updated! Written to {cache_file}")
            print(f"Total processed in this run: {image_count} images, {voice_count} voice notes.")
        except Exception as e:
            print(f"Error writing cache file: {e}")
            return 1
    else:
        print("\nAll assets were already present in cache. No updates written.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
