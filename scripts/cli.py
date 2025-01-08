"""
Kokoro CLI

Prerequisites:
    - pip install kokoro-onnx sounddevice
    - wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx
    - wget https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.json

Add the two files to the root of the repository.

Usage:
    python scripts/cli.py [options]

Examples:
    python scripts/cli.py --text "Bonjour le monde!" --lang "fr-fr" --speed 1.2 --voice "bf_isabella"

For more information, use:
    python scripts/cli.py --help
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List

import numpy as np
import sounddevice as sd
import soundfile as sf
from kokoro_onnx import Kokoro

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

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


class KokoroError(Exception):
    """Base exception class for Kokoro CLI errors."""

    pass


def check_requirements() -> None:
    """
    Verify that all required model files exist.

    Raises:
        KokoroError: If any required file is missing
    """
    required_files = [ONNX_MODEL, VOICES_JSON]
    for file in required_files:
        if not Path(file).is_file():
            raise KokoroError(
                f"Required file '{file}' not found. Please download from "
                "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/"
            )


def get_voices() -> List[str]:
    """
    Retrieve available voices from the JSON file.

    Returns:
        List of available voice names
    """
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

    Raises:
        KokoroError: If audio playback fails
    """
    try:
        sd.play(samples, sample_rate)
        sd.wait()  # Wait until audio is finished playing
    except Exception as e:
        raise KokoroError(f"Audio playback failed: {str(e)}")


def process_audio(args: argparse.Namespace, save: bool = DEFAULT_SAVE) -> None:
    """
    Process audio based on the provided arguments.

    Args:
        args: Parsed command line arguments
        save: Flag to determine if the audio should be saved to a file

    Raises:
        KokoroError: If audio processing or saving fails
    """
    try:
        kokoro = Kokoro(ONNX_MODEL, VOICES_JSON)
        samples, sample_rate = kokoro.create(
            args.text, voice=args.voice, speed=args.speed, lang=args.lang
        )

        if args.stream:
            play_audio(samples, sample_rate)
            logger.info("Audio successfully played")

        if save:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            sf.write(args.output, samples, sample_rate)
            logger.info(f"Audio saved to {args.output}")

    except Exception as e:
        raise KokoroError(f"Audio processing failed: {str(e)}")


def handle_args(args: argparse.Namespace) -> None:
    """
    Handle the command line arguments to determine the action.

    Args:
        args: Parsed command line arguments

    Raises:
        KokoroError: If voice is invalid or processing fails
    """
    if args.list_voices:
        voices = get_voices()
        print("Available voices:")
        for voice in voices:
            print(voice)
        return

    if args.voice not in get_voices():
        raise KokoroError(f"Voice '{args.voice}' not found in voices.json!")

    process_audio(args, save=args.save)


def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Generate audio using Kokoro ONNX model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--text", type=str, default=DEFAULT_TEXT, help="Text to be converted to speech"
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=DEFAULT_VOICE,
        help="Voice to be used for speech generation",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=DEFAULT_SPEED,
        help="Speed of the generated speech",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default=DEFAULT_LANG,
        help="Language of the generated speech",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Output file name for the generated audio",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        default=DEFAULT_STREAM,
        help="Stream audio instead of saving to file",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        default=DEFAULT_SAVE,
        help="Save the generated audio to a file",
    )
    parser.add_argument(
        "--list-voices", action="store_true", help="List all available voices"
    )
    return parser


def main() -> None:
    """
    Main function to parse arguments and execute the program.
    """
    try:
        check_requirements()
        parser = create_parser()
        args = parser.parse_args()
        handle_args(args)
    except KokoroError as e:
        logger.error(str(e))
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
