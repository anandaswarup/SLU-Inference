"""
SNIPS Dataset Download Script

This script downloads the SNIPS SLU dataset from HuggingFace and saves it to disk
in a structured format for offline processing.
"""

import argparse
import json
import os
import warnings
from pathlib import Path

import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

# Set environment variable and suppress warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")


def parse_arguments():
    """
    Setup command line argument parsing.
    """
    parser = argparse.ArgumentParser(
        description="Download SNIPS SLU dataset and save to disk",
        usage="Usage: python download_snips_dataset.py --output_dir <path>",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory to save the dataset",
    )

    parser.add_argument(
        "--audio_format",
        type=str,
        default="wav",
        choices=["wav", "flac"],
        help="Audio format to save (default: wav)",
    )

    return parser.parse_args()


def create_directory_structure(output_dir):
    """
    Create the directory structure for the dataset.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for each split (The HF repo has only 'train' split)
    for split in ["train"]:
        (output_path / split / "audio").mkdir(parents=True, exist_ok=True)

    return output_path


def save_audio_file(audio_data, sample_rate, audio_path, audio_format="wav"):
    """
    Save audio data to file.
    """
    try:
        sf.write(audio_path, audio_data, sample_rate, format=audio_format.upper())
        return True
    except Exception as e:
        print(f"Error saving audio {audio_path}: {e}")
        return False


def download_split(dataset_name, split, output_dir, audio_format="wav"):
    """
    Download and save a specific split of the dataset.
    """
    print(f"Downloading {split} split...")

    try:
        # Load the dataset split
        dataset = load_dataset(dataset_name, split=split)

        # Get total samples
        try:
            total_samples = len(dataset)  # type: ignore
            print(f"Found {total_samples} samples in {split} split")
        except (TypeError, AttributeError):
            print(f"Loading {split} split (streaming mode)")
            total_samples = None

        # Create output paths
        split_dir = Path(output_dir) / split
        audio_dir = split_dir / "audio"
        metadata_file = split_dir / "metadata.jsonl"

        # Process samples
        processed_count = 0
        failed_count = 0

        with open(metadata_file, "w", encoding="utf-8") as meta_file:
            # Create progress bar
            if total_samples:
                progress_bar = tqdm(
                    dataset, desc=f"Processing {split}", total=total_samples
                )
            else:
                progress_bar = tqdm(dataset, desc=f"Processing {split}")

            for i, sample in enumerate(progress_bar):
                try:
                    # Extract sample data
                    audio_data = sample["audio"]["array"]
                    sample_rate = sample["audio"]["sampling_rate"]
                    sample_id = sample.get("ID", f"{split}_sample_{i}")

                    # Create audio filename
                    audio_filename = f"{sample_id}.{audio_format}"
                    audio_path = audio_dir / audio_filename

                    # Save audio file
                    if save_audio_file(
                        audio_data, sample_rate, audio_path, audio_format
                    ):
                        # Create metadata entry
                        metadata_entry = {
                            "ID": sample_id,
                            "audio_path": str(audio_path.relative_to(output_dir)),
                            "text": sample.get("text", ""),
                            "intent": sample.get("intent", ""),
                            "scenario": sample.get("scenario", ""),
                            "action": sample.get("action", ""),
                            "sample_rate": sample_rate,
                            "original_index": i,
                        }

                        # Add any additional fields from the sample
                        for key, value in sample.items():
                            if key not in [
                                "audio",
                                "ID",
                                "text",
                                "intent",
                                "scenario",
                                "action",
                            ]:
                                metadata_entry[key] = value

                        # Write metadata
                        meta_file.write(json.dumps(metadata_entry) + "\n")
                        meta_file.flush()

                        processed_count += 1
                    else:
                        failed_count += 1

                except Exception as e:
                    print(f"Error processing sample {i}: {e}")
                    failed_count += 1
                    continue

        print(
            f"✓ {split} split completed: {processed_count} samples saved, {failed_count} failed"
        )
        return processed_count, failed_count

    except Exception as e:
        print(f"Error downloading {split} split: {e}")
        return 0, 0


def create_dataset_info(output_dir, splits_info):
    """
    Create a dataset info file with summary information.
    """
    info_file = Path(output_dir) / "dataset_info.json"

    info = {
        "dataset_name": "MWilinski/snips_slu_v1.0",
        "download_date": str(Path().resolve()),
        "splits": splits_info,
        "total_samples": sum(info["processed"] for info in splits_info.values()),
        "total_failed": sum(info["failed"] for info in splits_info.values()),
        "structure": {
            "audio_dir": "Each split has an 'audio' subdirectory with audio files",
            "metadata_file": "Each split has a 'metadata.jsonl' file with sample information",
            "audio_format": "Audio files are saved in the specified format (wav/flac)",
        },
    }

    with open(info_file, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)

    print(f"Dataset info saved to: {info_file}")


def main():
    """
    Main function to download SNIPS dataset.
    """
    args = parse_arguments()

    splits = ["train"]

    print("=" * 60)
    print("SNIPS SLU Dataset Download")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Splits to download: {', '.join(splits)}")
    print(f"Audio format: {args.audio_format}")
    print()

    # Create directory structure
    output_dir = create_directory_structure(args.output_dir)
    print(f"Created directory structure in: {output_dir}")

    # Download each split
    dataset_name = "MWilinski/snips_slu_v1.0"
    splits_info = {}

    for split in splits:
        processed, failed = download_split(
            dataset_name, split, output_dir, args.audio_format
        )
        splits_info[split] = {"processed": processed, "failed": failed}
        print()

    # Create dataset info file
    create_dataset_info(output_dir, splits_info)

    # Print summary
    print("Download Summary")
    total_processed = sum(info["processed"] for info in splits_info.values())
    total_failed = sum(info["failed"] for info in splits_info.values())

    for split, info in splits_info.items():
        print(f"{split}: {info['processed']} samples, {info['failed']} failed")

    print(f"\nTotal: {total_processed} samples downloaded, {total_failed} failed")
    print(f"Dataset saved to: {output_dir}")


if __name__ == "__main__":
    main()
