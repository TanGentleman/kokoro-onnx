"""
Kokoro CLI

Prerequisites:
pip install kokoro-onnx sounddevice

Download model files:

wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx
wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json

Place them in the same directory as this script.
"""

import argparse
import json
import os
import soundfile as sf
from kokoro_onnx import Kokoro
from typing import Dict, Any
import numpy as np
import sounddevice as sd

# Constants
ONNX_MODEL = "kokoro-v0_19.onnx"
VOICES_JSON = "voices.json"
DEFAULT_STREAM = True
DEFAULT_SAVE = False
DEFAULT_TEXT = "Hey there! I hope you're doing well!"
DEFAULT_VOICE = "af_sarah"
DEFAULT_SPEED = 1.0
DEFAULT_LANG = "en-us"
OUTPUT_DIR = "outputs"
DEFAULT_OUTPUT = os.path.join(OUTPUT_DIR, "audio.wav")

def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load a JSON file and return its content."""
    with open(file_path, "r") as f:
        return json.load(f)

def get_voices() -> list[str]:
    """Retrieve available voices from the JSON file."""
    return [
        "af",
        "af_bella",
        "af_nicole",
        "af_sarah",
        "af_sky",
        "am_adam",
        "am_michael",
        "bf_emma",
        "bf_isabella",
        "bm_george",
        "bm_lewis",
    ]

def play_audio(samples: np.ndarray, sample_rate: int) -> None:
    """
    Stream audio using sounddevice.
    
    Args:
        samples: Audio samples as numpy array
        sample_rate: Sample rate of the audio
    """
    sd.play(samples, sample_rate)
    sd.wait()  # Wait until audio is finished playing

def process_audio(args, save: bool = DEFAULT_SAVE) -> None:
    """
    Process audio based on the provided arguments.
    
    Args:
        args: Parsed command line arguments.
        save: Flag to determine if the audio should be saved to a file.
    """
    kokoro = Kokoro(ONNX_MODEL, VOICES_JSON)
    samples, sample_rate = kokoro.create(args.text, voice=args.voice, speed=args.speed, lang=args.lang)
    
    if args.stream:
        play_audio(samples, sample_rate)
        print("Audio successfully played")
    
    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        sf.write(args.output, samples, sample_rate)
        print(f"Audio saved to {args.output}")

def handle_args(args) -> None:
    """
    Handle the command line arguments to determine the action.
    
    Args:
        args: Parsed command line arguments.
    """
    if args.list_voices:
        voices = get_voices()
        print("Available voices:")
        for voice in voices:
            print(voice)
        return
    
    if args.voice not in get_voices():
        print(f"Error: Voice {args.voice} not found in voices.json!")
        return
    
    process_audio(args, save=args.save)

def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(description="Generate audio using Kokoro ONNX model.")
    parser.add_argument("--text", type=str, default=DEFAULT_TEXT, help="Text to be converted to speech.")
    parser.add_argument("--voice", type=str, default=DEFAULT_VOICE, help="Voice to be used for speech generation.")
    parser.add_argument("--speed", type=float, default=DEFAULT_SPEED, help="Speed of the generated speech.")
    parser.add_argument("--lang", type=str, default=DEFAULT_LANG, help="Language of the generated speech.")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output file name for the generated audio.")
    parser.add_argument("--stream", action="store_true", default=DEFAULT_STREAM, help="Stream audio instead of saving to file")
    parser.add_argument("--save", action="store_true", default=DEFAULT_SAVE, help="Save the generated audio to a file")
    parser.add_argument("--list-voices", action="store_true", help="List all available voices")
    return parser

def main() -> None:
    """Main function to parse arguments and execute the program."""
    parser = create_parser()
    args = parser.parse_args()
    handle_args(args)

if __name__ == "__main__":
    main()
