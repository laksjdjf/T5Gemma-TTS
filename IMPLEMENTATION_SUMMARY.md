# Multi-Token Prediction Implementation Summary

## What was implemented?

This PR adds **multi-token prediction** support to T5Gemma-TTS, enabling the model to predict multiple future tokens simultaneously at each decoding step during training.

## Quick Example

### Before (Standard single-token prediction):
```python
# At position t, predict only token at position t
hidden[t] -> predict[t]
```

### After (Multi-token prediction with num_predict_tokens=3):
```python
# At position t, predict tokens at positions t, t+1, and t+2
hidden[t] -> predict[t]     (head 0)
hidden[t] -> predict[t+1]   (head 1)
hidden[t] -> predict[t+2]   (head 2)
```

## Usage

### Training with Multi-Token Prediction

Add these parameters to your training command:

```bash
--num_predict_tokens 3 \
--multi_token_loss_weight "[1.0, 0.5, 0.25]"
```

Or use the example script:
```bash
NUM_GPUS=1 examples/training/t5gemma_2b-2b-ft-multi-token.sh
```

### Default Behavior (Backward Compatible)

Without these parameters, the model behaves exactly as before:
```bash
--num_predict_tokens 1  # Default: standard single-token prediction
```

## Benefits

1. **Better Training Signal**: Model receives gradients from multiple future positions
2. **Improved Representations**: Learning to predict multiple steps ahead can improve learned features
3. **Training Efficiency**: Multiple predictions per forward pass
4. **Future-Ready**: Architecture supports potential speculative decoding in future

## Technical Details

### Files Modified
- `config.py`: Added configuration parameters
- `models/t5gemma.py`: Updated model architecture and training logic
- `hf_export/modeling_t5gemma_voice.py`: Updated HF export model
- `README.md` & `README_ja.md`: Added feature to documentation

### Files Added
- `docs/MULTI_TOKEN_PREDICTION.md`: Comprehensive documentation
- `test_multi_token.py`: Test suite
- `examples/training/t5gemma_2b-2b-ft-multi-token.sh`: Example training script

### Architecture Changes

**Prediction Layer Structure:**
```python
# Before: One head per codebook
predict_layer[codebook_idx] -> Sequential(Linear, GELU, Linear)

# After: Multiple heads per codebook
predict_layer[codebook_idx][prediction_position] -> Sequential(Linear, GELU, Linear)
```

### Loss Computation

For `num_predict_tokens=3`:
- Position 0 (weight=1.0): `loss_0 = cross_entropy(logits[t], target[t])`
- Position 1 (weight=0.5): `loss_1 = cross_entropy(logits[t], target[t+1])`
- Position 2 (weight=0.25): `loss_2 = cross_entropy(logits[t], target[t+2])`
- Total: `loss = (loss_0 * 1.0 + loss_1 * 0.5 + loss_2 * 0.25) / 3`

### Inference

During inference, only the first prediction head (position 0) is used, ensuring:
- Same inference speed as before
- Compatibility with any model (single or multi-token trained)
- No changes to inference API

## Testing

Run the test suite:
```bash
python test_multi_token.py
```

Tests validate:
- ✓ Configuration parameters
- ✓ Model structure for both modes
- ✓ Target alignment logic
- ✓ Loss weight parsing

## Backward Compatibility

✅ **Fully backward compatible**
- Default `num_predict_tokens=1` behaves exactly as before
- Models can be loaded and used without changes
- Inference API unchanged

## Performance Considerations

### Memory Usage
Multi-token prediction increases memory proportionally:
- `num_predict_tokens=1`: baseline memory
- `num_predict_tokens=2`: ~2x prediction head parameters
- `num_predict_tokens=3`: ~3x prediction head parameters

### Training Time
Slight increase in training time per step due to:
- Multiple prediction heads forward pass
- Multiple loss computations
- But provides better gradient signals

### Recommended Settings
- Start with `num_predict_tokens=2` or `3`
- Use decreasing loss weights: `[1.0, 0.5, 0.25]`
- Monitor validation loss to ensure improvement

## Future Work

Potential extensions:
- Speculative decoding for faster inference
- Adaptive prediction horizon
- Per-layer prediction heads
- Token-level confidence estimation

## References

For detailed documentation, see:
- [docs/MULTI_TOKEN_PREDICTION.md](docs/MULTI_TOKEN_PREDICTION.md)

For questions or issues, please open a GitHub issue.
