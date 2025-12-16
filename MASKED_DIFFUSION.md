# Masked Diffusion for T5Gemma-TTS

This document describes the Masked Diffusion implementation added to T5Gemma-TTS.

## Overview

Masked Diffusion is a non-autoregressive approach to audio token generation that replaces the standard autoregressive (AR) token-by-token generation with an iterative denoising process. This approach is inspired by:
- Masked Diffusion Language Models (MDLM)
- MaskGIT
- Non-autoregressive models in general

## Key Differences from Autoregressive Model

| Aspect | Autoregressive (T5Gemma) | Masked Diffusion |
|--------|-------------------------|------------------|
| Generation | Sequential, left-to-right | Parallel, iterative refinement |
| Attention | Causal (masked) | Bidirectional |
| Training | Predict next token | Predict masked tokens |
| Inference Speed | Slow (one token at a time) | Faster (multiple tokens per step) |
| Quality Control | Top-k/top-p sampling | Confidence-based unmasking |

## Architecture

The `MaskedDiffusionModel` class inherits the same T5Gemma backbone but modifies:

1. **Training Process**:
   - Randomly masks a portion of audio tokens based on a diffusion schedule
   - Model predicts the original tokens at masked positions
   - Loss computed only on masked tokens
   - Uses bidirectional attention (no causal mask)

2. **Inference Process**:
   - Start with all tokens (except prompt) masked
   - Iteratively unmask tokens over `diffusion_steps` iterations
   - At each step:
     - Predict all masked tokens
     - Compute confidence scores
     - Unmask the most confident predictions
   - Continue until all tokens are unmasked

3. **Masking Schedule**:
   - **Cosine**: `cos(t * π/2)` - smoother transitions (default)
   - **Linear**: `1 - t` - uniform unmasking rate

## Usage

### Training

Use the provided training script for masked diffusion:

```bash
NUM_GPUS=8 examples/training/masked_diffusion_2b-2b.sh
```

Key training parameters:
- `--model_arch masked_diffusion`: Select masked diffusion architecture
- `--diffusion_steps 10`: Number of diffusion steps (default: 10)
- `--mask_schedule cosine`: Masking schedule (cosine or linear)

### Inference

The masked diffusion model uses the same inference interface:

```bash
python inference_commandline.py \
    --model_root . \
    --model_name masked_diffusion \
    --target_text "Your text here"
```

During inference, the model will:
1. Encode the text prompt
2. Initialize target sequence with mask tokens
3. Iteratively predict and unmask tokens over `diffusion_steps` iterations
4. Return the final generated audio tokens

## Configuration Parameters

New parameters added to `config.py`:

```python
--model_arch: ["t5gemma", "masked_diffusion"]  # Select architecture
--diffusion_steps: int (default: 10)            # Number of diffusion iterations
--mask_schedule: ["cosine", "linear"]           # Masking schedule type
```

## Implementation Details

### Masking Strategy

During training, tokens are masked uniformly at random positions based on the current mask ratio. The mask ratio follows the chosen schedule:

- At step 0: mask_ratio = 1.0 (fully masked)
- At step T: mask_ratio = 0.0 (no masking)

### Confidence-Based Unmasking

During inference, tokens are unmasked based on model confidence:
1. Predict all masked positions
2. Compute softmax probabilities
3. Select positions with highest max probability
4. Unmask those positions
5. Keep remaining positions masked for next iteration

### Bidirectional Attention

Unlike the autoregressive model which uses causal attention, masked diffusion uses bidirectional attention. This allows the model to:
- See context from both directions
- Make better predictions for masked tokens
- Potentially improve coherence

## Advantages

1. **Parallel Generation**: Multiple tokens can be unmasked per step
2. **Controllable Quality**: More diffusion steps = higher quality (but slower)
3. **Better Context**: Bidirectional attention captures full context
4. **Flexible**: Can adjust diffusion steps at inference time

## Limitations

1. **Training**: Requires retraining from scratch or fine-tuning with new objective
2. **Memory**: Bidirectional attention may use more memory during training
3. **Speed**: Still requires multiple forward passes (though fewer than AR)
4. **Tuning**: Optimal number of diffusion steps may vary by use case

## Future Improvements

- [ ] Adaptive diffusion steps based on confidence
- [ ] Mixed AR/diffusion hybrid approaches
- [ ] Temperature scheduling during unmasking
- [ ] Multi-codebook support for diffusion
- [ ] Continuous diffusion variants

## References

- [MaskGIT: Masked Generative Image Transformer](https://arxiv.org/abs/2202.04200)
- [Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524)
- [VoiceStar (base architecture)](https://arxiv.org/abs/2505.19462)

## Citation

If you use the Masked Diffusion implementation, please cite both the original T5Gemma-TTS and this extension:

```bibtex
@misc{t5gemma-tts-masked-diffusion,
  title={Masked Diffusion for T5Gemma-TTS},
  author={T5Gemma-TTS Contributors},
  year={2025},
  url={https://github.com/laksjdjf/T5Gemma-TTS}
}
```
