"""
Super simple TTS inference script.
Minimal dependencies and easy to use.

Usage:
    python inference_simple.py "Hello, this is a test."
    python inference_simple.py "こんにちは" --output my_audio.wav
"""

import os
import sys
import argparse
import torch
import torchaudio

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError:
    print("Error: transformers is not installed.")
    print("Please install it with: pip install transformers")
    sys.exit(1)

from data.tokenizer import AudioTokenizer
from duration_estimator import estimate_duration
from inference_tts_utils import inference_one_sample, normalize_text_with_lang


def generate_speech(
    text,
    model_dir="Aratako/T5Gemma-TTS-2b-2b",
    output_path="output.wav",
):
    """
    Generate speech from text using T5Gemma-TTS.
    
    Args:
        text: Text to convert to speech
        model_dir: HuggingFace model directory or path
        output_path: Where to save the generated audio
    """
    print(f"Loading model from {model_dir}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Choose dtype based on device capability
    if device == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    
    # Load model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        dtype=dtype,
    ).to(device)
    model.eval()
    
    cfg = model.config
    tokenizer_name = getattr(cfg, "text_tokenizer_name", None) or getattr(cfg, "t5gemma_model_name", None)
    text_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Audio tokenizer
    audio_tokenizer = AudioTokenizer(
        backend="xcodec2",
        model_name=getattr(cfg, "xcodec2_model_name", None),
    )
    codec_audio_sr = audio_tokenizer.sample_rate
    codec_sr = getattr(cfg, "encodec_sr", 50)
    
    # Normalize text
    text, lang_code = normalize_text_with_lang(text, None)
    
    # Estimate duration
    target_duration = estimate_duration(
        target_text=text,
        reference_speech=None,
        reference_transcript=None,
        target_lang=lang_code,
        reference_lang=None,
    )
    print(f"Generating speech (estimated duration: {target_duration:.2f}s)...")
    
    # Simple decode config
    decode_config = {
        "top_k": 30,
        "top_p": 0.9,
        "min_p": 0,
        "temperature": 0.8,
        "stop_repetition": 3,
        "codec_audio_sr": codec_audio_sr,
        "codec_sr": codec_sr,
        "silence_tokens": [],
        "sample_batch_size": 1,
    }
    
    # Generate
    concat_audio, gen_audio = inference_one_sample(
        model=model,
        model_args=cfg,
        text_tokenizer=text_tokenizer,
        audio_tokenizer=audio_tokenizer,
        audio_fn=None,
        target_text=text,
        lang=lang_code,
        device=device,
        decode_config=decode_config,
        prompt_end_frame=0,
        target_generation_length=target_duration,
        prefix_transcript="",
        multi_trial=[],
        repeat_prompt=0,
        return_frames=False,
    )
    
    gen_audio = gen_audio[0].cpu()
    
    # Save
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directory if path includes a directory
        os.makedirs(output_dir, exist_ok=True)
    torchaudio.save(output_path, gen_audio, codec_audio_sr)
    print(f"✓ Audio saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Simple TTS generation")
    parser.add_argument("text", help="Text to convert to speech")
    parser.add_argument("--model", default="Aratako/T5Gemma-TTS-2b-2b", help="Model directory")
    parser.add_argument("--output", default="output.wav", help="Output audio file")
    
    args = parser.parse_args()
    
    try:
        generate_speech(args.text, args.model, args.output)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
