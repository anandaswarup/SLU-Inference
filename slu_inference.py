"""
SLU Inference Script using SpeechBrain

This script performs Spoken Language Understanding inference using a pre-trained
SpeechBrain model to extract semantic information from audio files.
"""

import argparse
import os
import sys
import traceback
import warnings
from pathlib import Path

import torch

# Set environment variable and suppress warnings before importing SpeechBrain
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

from speechbrain.inference.SLU import EndToEndSLU  # noqa: E402


def parse_arguments():
    """
    Setup command line argument parsing.
    """
    parser = argparse.ArgumentParser(
        description="Perform SLU inference on audio files using SpeechBrain models",
        usage="python slu_inference.py --slu_model_path <path> --asr_model_path <path>  --wav_file <path>",
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
        "--wav_file", type=str, required=True, help="Path to the input WAV file"
    )

    return parser.parse_args()


def validate_paths(args):
    """
    Validate that all required paths exist.
    """
    paths_to_check = [
        (args.slu_model_path, "SLU model path"),
        (args.asr_model_path, "ASR model path"),
        (args.wav_file, "WAV file"),
    ]

    for path, description in paths_to_check:
        if not Path(path).exists():
            print(f"Error: {description} does not exist: {path}")
            sys.exit(1)


def main():
    """
    Main function to perform SLU inference.
    """
    # Parse arguments
    args = parse_arguments()

    # Validate paths
    validate_paths(args)

    # Create the SLU model using local paths
    try:
        # Convert to absolute path to avoid path concatenation issues
        slu_model_path = Path(args.slu_model_path).resolve()

        # ASR model path
        asr_model_path = str(args.asr_model_path)

        # Check if hyperparams.yaml exists
        hparams_file = slu_model_path / "hyperparams.yaml"
        if not hparams_file.exists():
            print(f"Error: hyperparams.yaml not found at {hparams_file}")
            sys.exit(1)

        slu_model = EndToEndSLU.from_hparams(
            source=str(slu_model_path),
            savedir=str(slu_model_path),
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            # Use local ASR model
            overrides={"asr_model_source": asr_model_path},
        )
        print("SLU Seq2Seq model (+ CRDNN ASR encoder) loaded successfully")
    except Exception as e:
        print(f"Error loading SLU model: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Perform SLU inference
    print("Performing SLU inference...")
    try:
        # For decode_batch to work properly, we need to use model's audio loading
        # This ensures compatibility with the model's internal preprocessing pipeline
        assert slu_model is not None, "SLU model is None"
        model_audio = slu_model.load_audio(args.wav_file)

        # Prepare audio for batch processing
        lengths = torch.tensor([1.0])  # Single audio file with relative length 1.0
        predicted_words, predicted_tokens = slu_model.decode_batch(
            model_audio.unsqueeze(0), lengths
        )
    except Exception as e:
        print(f"Error during SLU inference: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("Inference completed")

    print(f"SLU Output: {predicted_words}")


if __name__ == "__main__":
    main()
