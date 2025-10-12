# SLU Inference with SpeechBrain

This project provides a command-line tool for performing Spoken Language Understanding (SLU) inference using the SpeechBrain `speechbrain/slu-direct-fluent-speech-commands-librispeech-asr` model.

## Features

- Load pre-trained SLU and ASR models from local disk
- Process WAV audio files for intent recognition
- Extract (action, object, location) tuples from speech commands
- Support for Fluent Speech Commands dataset format
- Robust error handling and multiple inference methods

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies

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

### Example Output

```
============================================================
SLU Inference using SpeechBrain
============================================================
Loading SLU model...
✓ SLU model loaded successfully

Loading audio file...
Loaded audio: /path/to/audio.wav
Sample rate: 16000 Hz
Duration: 2.34 seconds

Performing SLU inference...
Used classify_file method
✓ Inference completed

============================================================
RAW OUTPUT:
============================================================
Type: <class 'dict'>
Content: {'intent': 'turn on lights kitchen'}

============================================================
PARSED RESULTS:
============================================================
Action:   turn
Object:   lights
Location: kitchen

Tuple: (turn, lights, kitchen)

============================================================
Inference completed successfully!
============================================================
```

## Supported Intent Formats

The script supports parsing various intent formats from the Fluent Speech Commands dataset:

### Common Actions
- `turn`, `bring`, `activate`, `deactivate`
- `increase`, `decrease`, `change`, `switch`
- `set`, `play`, `stop`, `pause`

### Common Objects
- `lights`, `music`, `volume`, `temperature`
- `heat`, `lamp`, `tv`, `television`
- `radio`, `player`, `air`, `conditioning`

### Common Locations
- `kitchen`, `bedroom`, `living`, `room`
- `bathroom`, `office`, `dining`, `garage`
- `basement`, `upstairs`, `downstairs`

## Audio Requirements

- **Format**: WAV files
- **Sample Rate**: Any (will be handled automatically)
- **Channels**: Mono or stereo (stereo will be converted to mono)
- **Duration**: Any reasonable length

## Error Handling

The script includes robust error handling for:
- Missing files or invalid paths
- Audio loading issues
- Model loading failures
- Multiple inference method fallbacks
- Output parsing errors

## Troubleshooting

### Common Issues

1. **Model loading fails**: Ensure the model paths are correct and contain all necessary files
2. **Audio loading fails**: Check that the WAV file is valid and accessible
3. **Inference methods fail**: The script tries multiple inference approaches automatically
4. **Parsing fails**: Check the raw output to understand the model's response format

### Debug Information

The script provides detailed debug information including:
- Model loading status
- Audio file properties
- Inference method used
- Raw model output
- Parsed results

## License

This project is licensed under the terms specified in the LICENSE file.

## Contributing

Feel free to submit issues and pull requests to improve the script.
