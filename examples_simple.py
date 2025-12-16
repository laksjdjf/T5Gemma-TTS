#!/usr/bin/env python
"""
Example usage of the simple TTS inference.
Demonstrates various ways to use inference_simple.py
"""

import subprocess
import sys


def run_example(description, command):
    """Run an example command with description."""
    print(f"\n{'='*60}")
    print(f"Example: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}")
    
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"⚠️  Example failed with code {result.returncode}")
    else:
        print("✓ Example completed successfully")


def main():
    print("T5Gemma-TTS Super Simple Version - Examples")
    print("=" * 60)
    
    examples = [
        (
            "Basic English TTS",
            'python inference_simple.py "Hello, this is a simple text to speech example."'
        ),
        (
            "Japanese TTS",
            'python inference_simple.py "こんにちは、これは音声合成の例です。"'
        ),
        (
            "Chinese TTS",
            'python inference_simple.py "你好，这是一个语音合成示例。"'
        ),
        (
            "Custom output filename",
            'python inference_simple.py "Testing custom output" --output examples/test.wav'
        ),
    ]
    
    print("\nThis script will run 4 examples.")
    print("Each example generates speech and saves it as a WAV file.")
    print("\nPress Ctrl+C to skip all examples, or wait to continue...")
    
    try:
        import time
        time.sleep(3)
    except KeyboardInterrupt:
        print("\n\nSkipping examples. You can run them manually using the commands above.")
        sys.exit(0)
    
    for description, command in examples:
        try:
            run_example(description, command)
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            break
        except Exception as e:
            print(f"Error running example: {e}")
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("Check the generated .wav files in the current directory.")
    print("="*60)


if __name__ == "__main__":
    main()
