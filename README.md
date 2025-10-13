# SLU Inference with SpeechBrain

This project provides command-line tools for performing Spoken Language Understanding (SLU) inference using SpeechBrain models on different datasets.

## Available Scripts

### 1. Single File Inference (`slu_inference.py`)
- Process individual WAV files
- Extract (action, object, location) tuples

### 2. Fluent Speech Commands Dataset (`fluent_speech_inference.py`)
- Batch processing of Fluent Speech Commands test dataset
- Output: filepath, transcription, action, object, location, raw_slu_output

## Features

- Load pre-trained SLU and ASR models from local disk
- Process WAV audio files for intent recognition
- Support for multiple dataset formats (Fluent Speech Commands, SNIPS)
- Robust parsing of different SLU output formats (JSON, malformed JSON, simple text)
- Robust error handling and multiple inference methods


## Dependencies

- `torch>=1.10.0`
- `torchaudio>=0.10.0`
- `speechbrain>=0.5.0`
- `transformers`
- `sentencepiece`

## Usage

### Basic Usage

```bash
python slu_inference.py \
    --slu_model_path /path/to/slu/model \
    --asr_model_path /path/to/asr/model \
    --tokenizer_path /path/to/tokenizer.model \
    --wav_file /path/to/audio.wav
```

### Arguments

- `--slu_model_path`: Path to the SLU model directory (required)
- `--asr_model_path`: Path to the ASR model directory (required)
- `--tokenizer_path`: Path to the sentence piece tokenizer model file (required)
- `--wav_file`: Path to the input WAV file (required)

### Model Setup

1. **Download the models**: Ensure you have downloaded the SpeechBrain models to your local disk:
   - SLU model: `speechbrain/slu-direct-fluent-speech-commands-librispeech-asr`
   - ASR model (dependency of the SLU model)
   - Tokenizer model

2. **Directory structure**: Your model directories should contain the necessary files:
   ```
   slu_model/
   ├── hyperparams.yaml
   ├── model.ckpt
   └── ...
   
   asr_model/
   ├── hyperparams.yaml
   ├── model.ckpt
   └── ...
   ```