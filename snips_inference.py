"""
SNIPS Dataset Batch Inference Script

This script performs SLU inference on the SNIPS dataset train split and saves
the results to a CSV file with parsed action, object, and location components.

Output CSV contains: filepath, transcription, action, object, location, raw_slu_output
"""

import argparse
import csv
import json
import os
import sys
import warnings
from pathlib import Path

import torch
from tqdm import tqdm

# Set environment variable and suppress warnings before importing SpeechBrain
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

from speechbrain.inference.SLU import EndToEndSLU  # noqa: E402


def parse_arguments():
    """
    Setup command line argument parsing.
    """
    parser = argparse.ArgumentParser(
        description="Perform SLU inference on SNIPS dataset using SpeechBrain models",
        usage="python snips_inference.py --slu_model_path <path> --asr_model_path <path> "
        "--dataset_path <path> --output_csv <path> [--max_samples <num>]",
    )

    parser.add_argument(
        "--slu_model_path",
        type=str,
        required=True,
        help="Path to the SLU model directory",
    )

    parser.add_argument(
        "--asr_model_path",
        type=str,
        required=True,
        help="Path to the ASR model directory",
    )

    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the SNIPS dataset directory",
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Path to the output CSV file",
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all)",
    )

    return parser.parse_args()


def validate_paths(args):
    """
    Validate that all required paths exist.
    """
    paths_to_check = [
        (args.slu_model_path, "SLU model path"),
        (args.asr_model_path, "ASR model path"),
        (args.dataset_path, "Dataset path"),
    ]

    for path, description in paths_to_check:
        if not Path(path).exists():
            print(f"Error: {description} does not exist: {path}")
            sys.exit(1)

    # Check if the train directory exists
    train_data_dir = Path(args.dataset_path) / "train"

    if not train_data_dir.exists():
        # Try to find any wav files in the dataset directory
        wav_files = list(Path(args.dataset_path).rglob("*.wav"))
        if not wav_files:
            print(f"Error: No audio data found in: {args.dataset_path}")
            print("Expected 'train' subdirectory with audio files")
            sys.exit(1)

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_csv).parent
    output_dir.mkdir(parents=True, exist_ok=True)


def load_snips_train_data(dataset_path):
    """
    Load the SNIPS train dataset for inference.

    The SNIPS dataset structure:
    - train/
      - wav/ (contains 00000.wav, 00001.wav, etc.)
      - meta/ (contains 00000.json, 00001.json, etc.)
      - index.jsonl (contains all metadata in one file)
    """
    dataset_path = Path(dataset_path)

    print(f"Loading SNIPS train data from: {dataset_path}")

    samples = []

    # Strategy 1: Look for train split with index.jsonl file
    train_dir = dataset_path / "train"
    if train_dir.exists():
        index_file = train_dir / "index.jsonl"
        if index_file.exists():
            print(f"Found train directory with index file: {index_file}")

            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            data = json.loads(line.strip())

                            # Get the wav file path
                            wav_rel_path = data.get("wav", "")
                            if wav_rel_path:
                                wav_path = train_dir / wav_rel_path
                            else:
                                # Fallback: construct from id
                                item_id = data.get("id", f"{line_num - 1:05d}")
                                wav_path = train_dir / "wav" / f"{item_id}.wav"

                            if wav_path.exists():
                                # Extract entities/slots information
                                entities = data.get("entities", [])
                                slots_list = []
                                if entities:
                                    for entity in entities:
                                        entity_type = entity.get("entity", "")
                                        entity_value = entity.get("value", "")
                                        if entity_type and entity_value:
                                            slots_list.append(
                                                f"{entity_type}:{entity_value}"
                                            )

                                sample = {
                                    "audio_path": str(wav_path),
                                    "transcription": data.get("text", ""),
                                    "intent": data.get("intent", "unknown"),
                                    "slots": " ".join(slots_list),
                                }
                                samples.append(sample)
                            else:
                                print(f"Warning: Audio file not found: {wav_path}")

                        except json.JSONDecodeError as e:
                            print(
                                f"Warning: Could not parse line {line_num} in {index_file}: {e}"
                            )
                            continue

            except Exception as e:
                print(f"Error reading index file {index_file}: {e}")
                sys.exit(1)

        # Strategy 2: Look for wav directory directly
        elif (train_dir / "wav").exists():
            print(f"Found train/wav directory: {train_dir / 'wav'}")
            wav_dir = train_dir / "wav"
            meta_dir = train_dir / "meta"

            wav_files = sorted(wav_dir.glob("*.wav"))

            for wav_file in wav_files:
                # Look for corresponding metadata file
                meta_file = meta_dir / wav_file.with_suffix(".json").name

                transcription = ""
                intent = "unknown"
                slots = ""

                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            transcription = data.get("text", "")
                            intent = data.get("intent", "unknown")

                            # Extract entities/slots
                            entities = data.get("entities", [])
                            slots_list = []
                            if entities:
                                for entity in entities:
                                    entity_type = entity.get("entity", "")
                                    entity_value = entity.get("value", "")
                                    if entity_type and entity_value:
                                        slots_list.append(
                                            f"{entity_type}:{entity_value}"
                                        )
                            slots = " ".join(slots_list)

                    except Exception as e:
                        print(f"Warning: Could not read metadata file {meta_file}: {e}")

                sample = {
                    "audio_path": str(wav_file),
                    "transcription": transcription,
                    "intent": intent,
                    "slots": slots,
                }
                samples.append(sample)

    # Strategy 3: Fallback - look for any wav files in subdirectories
    else:
        print("Looking for wav files in subdirectories...")
        wav_files = list(dataset_path.rglob("*.wav"))

        if wav_files:
            print(f"Found {len(wav_files)} .wav files")

            for wav_file in wav_files:
                sample = {
                    "audio_path": str(wav_file),
                    "transcription": wav_file.stem,  # Use filename as fallback
                    "intent": "unknown",
                    "slots": "",
                }
                samples.append(sample)
        else:
            print("Error: No audio files found in dataset directory")
            sys.exit(1)

    if not samples:
        print("Error: No test samples found")
        sys.exit(1)

    print(f"✓ Loaded {len(samples)} test samples")
    return samples


def load_slu_model(slu_model_path, asr_model_path):
    """
    Load the SLU model from disk.
    """
    try:
        # Convert to absolute path to avoid path concatenation issues
        slu_model_path = Path(slu_model_path).resolve()

        # Check if hyperparams.yaml exists
        hparams_file = slu_model_path / "hyperparams.yaml"
        if not hparams_file.exists():
            print(f"Error: hyperparams.yaml not found at {hparams_file}")
            sys.exit(1)

        print(f"Loading SLU model from: {slu_model_path}")

        slu_model = EndToEndSLU.from_hparams(
            source=str(slu_model_path),
            savedir=str(slu_model_path),
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            # Use local ASR model
            overrides={"asr_model_source": asr_model_path},
        )
        print("✓ SLU model loaded successfully")
        return slu_model

    except Exception as e:
        print(f"Error loading SLU model: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def perform_slu_inference(slu_model, audio_file_path):
    """
    Perform SLU inference on a single audio file.
    """
    try:
        # Use decode_batch method for inference
        model_audio = slu_model.load_audio(str(audio_file_path))

        # Prepare audio for batch processing
        lengths = torch.tensor([1.0])  # Single audio file with relative length 1.0
        predicted_words, predicted_tokens = slu_model.decode_batch(
            model_audio.unsqueeze(0), lengths
        )

        # Extract the first (and only) prediction
        slu_output = predicted_words[0] if predicted_words else ""

        return slu_output

    except Exception as e:
        print(f"Error during SLU inference for {audio_file_path}: {e}")
        return f"ERROR: {str(e)}"


def parse_slu_output(slu_output):
    """
    Parse the SLU output to extract action, object, and location.

    The SLU output can be in different formats:
    1. JSON format: {"action": "activate", "object": "lights", "location": "none"}
    2. Space-separated format: "activate lights none"
    """
    if not slu_output or slu_output.startswith("ERROR:"):
        return "unknown", "unknown", "unknown"

    try:
        # First, try to parse as JSON or JSON-like structure
        if "{" in slu_output and "}" in slu_output:
            import re

            # Extract all quoted strings from the output
            quoted_strings = re.findall(r'"([^"]+)"', slu_output)

            action = "unknown"
            object_val = "unknown"
            location = "unknown"

            # Look for action, object, location values by finding them after their keys
            for i, string in enumerate(quoted_strings):
                if string.lower() in ["action", "action:"]:
                    if i + 1 < len(quoted_strings):
                        action = quoted_strings[i + 1]
                elif string.lower() == "object":
                    if i + 1 < len(quoted_strings):
                        object_val = quoted_strings[i + 1]
                elif string.lower() == "location":
                    if i + 1 < len(quoted_strings):
                        location = quoted_strings[i + 1]

            return action, object_val, location

        # Fall back to space-separated parsing
        else:
            # Split the output into components
            parts = slu_output.strip().split()

            # Default values
            action = "unknown"
            object_val = "unknown"
            location = "unknown"

            # Extract components based on the number of parts
            if len(parts) >= 1:
                action = parts[0]
            if len(parts) >= 2:
                object_val = parts[1]
            if len(parts) >= 3:
                location = parts[2]

            # Handle special cases where there might be multi-word components
            if len(parts) > 3:
                # If there are more than 3 parts, combine the middle parts as object
                # and use the last part as location
                action = parts[0]
                object_val = " ".join(parts[1:-1])
                location = parts[-1]

            return action, object_val, location

    except Exception as e:
        print(f"Warning: Error parsing SLU output '{slu_output}': {e}")
        return "unknown", "unknown", "unknown"


def process_snips_dataset(args, slu_model):
    """
    Process the SNIPS dataset and perform SLU inference.
    """
    print("Loading SNIPS train dataset...")

    # Load the train data
    samples = load_snips_train_data(args.dataset_path)

    # Limit samples if specified
    if args.max_samples:
        original_count = len(samples)
        samples = samples[: args.max_samples]
        print(f"Limited to {len(samples)} samples (from {original_count} total)")

    # Create CSV file and write header
    csv_file = Path(args.output_csv)
    print(f"Writing results to: {csv_file}")

    with open(csv_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = [
            "filepath",
            "transcription",
            "action",
            "object",
            "location",
            "raw_slu_output",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Process each sample
        processed_count = 0
        failed_count = 0

        progress_bar = tqdm(samples, desc="Processing samples")

        for sample in progress_bar:
            try:
                # Extract required fields
                audio_path = sample["audio_path"]
                transcription = sample["transcription"]

                # Check if audio file exists
                if not Path(audio_path).exists():
                    print(f"Warning: Audio file not found: {audio_path}")
                    failed_count += 1
                    continue

                # Perform SLU inference
                slu_output = perform_slu_inference(slu_model, audio_path)

                # Parse the SLU output to extract components
                action, object_val, location = parse_slu_output(slu_output)

                # Write to CSV
                writer.writerow(
                    {
                        "filepath": audio_path,
                        "transcription": transcription,
                        "action": action,
                        "object": object_val,
                        "location": location,
                        "raw_slu_output": slu_output,
                    }
                )

                processed_count += 1

            except Exception as e:
                print(f"Error processing sample: {e}")
                failed_count += 1
                continue

    print(
        f"✓ Processing completed. Processed {processed_count} samples, {failed_count} failed. Results saved to: {csv_file}"
    )


def main():
    """
    Main function to process SNIPS dataset with SLU inference.
    """
    # Parse arguments
    args = parse_arguments()

    # Validate paths
    validate_paths(args)

    print("SNIPS Dataset Batch Inference")
    print(f"Dataset path: {args.dataset_path}")
    print(f"Output CSV: {args.output_csv}")
    if args.max_samples:
        print(f"Max samples: {args.max_samples}")
    print()

    # Load SLU model
    slu_model = load_slu_model(args.slu_model_path, args.asr_model_path)

    # Process dataset
    process_snips_dataset(args, slu_model)

    print("Batch inference completed successfully!")


if __name__ == "__main__":
    main()
