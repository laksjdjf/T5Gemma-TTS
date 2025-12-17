# Multi-Token Prediction Support

## Overview

Multi-token prediction (also known as multi-step or parallel prediction) is a technique where the model learns to predict multiple future tokens simultaneously at each decoding step, rather than just the next immediate token. This can provide several benefits:

1. **Better Gradient Signal**: The model receives training signals from multiple future positions, potentially leading to better representations.
2. **Improved Training Efficiency**: Multiple predictions per forward pass can improve training throughput.
3. **Future Inference Optimization**: While current inference still generates tokens sequentially, the multi-head structure could enable speculative decoding or other advanced inference techniques in the future.

## Configuration

Two new parameters have been added to control multi-token prediction:

### `--num_predict_tokens` (int, default: 1)
Number of tokens to predict at each decoder step.
- `1`: Standard autoregressive mode (backward compatible)
- `>1`: Multi-token prediction mode

### `--multi_token_loss_weight` (str, default: None)
Loss weights for different prediction positions. Should be a string representation of a list with length equal to `num_predict_tokens`.

Example: `"[1.0, 0.5, 0.25]"` for 3-token prediction
- First value (1.0): weight for predicting the immediate next token
- Second value (0.5): weight for predicting 2 steps ahead
- Third value (0.25): weight for predicting 3 steps ahead

If not specified, all positions receive equal weight (1.0).

## Usage Examples

### Training with Multi-Token Prediction

#### Standard Single-Token Mode (Default/Backward Compatible)
```bash
python main.py \
    --exp_dir ./checkpoints \
    --dataset my_dataset \
    --dataset_dir ./data/my_dataset \
    --num_predict_tokens 1
    # ... other training args
```

#### 2-Token Prediction
```bash
python main.py \
    --exp_dir ./checkpoints \
    --dataset my_dataset \
    --dataset_dir ./data/my_dataset \
    --num_predict_tokens 2 \
    --multi_token_loss_weight "[1.0, 0.5]"
    # ... other training args
```

#### 3-Token Prediction
```bash
python main.py \
    --exp_dir ./checkpoints \
    --dataset my_dataset \
    --dataset_dir ./data/my_dataset \
    --num_predict_tokens 3 \
    --multi_token_loss_weight "[1.0, 0.5, 0.25]"
    # ... other training args
```

### Inference

During inference, the model automatically uses the first prediction head (which predicts the immediate next token), so inference behavior is identical regardless of how many prediction heads were trained. This ensures compatibility and allows you to use models trained with multi-token prediction for standard sequential generation.

```bash
# Inference works the same way for both single and multi-token trained models
python inference_commandline_hf.py \
    --model_dir path/to/model \
    --target_text "Your text here"
```

## Implementation Details

### Model Architecture Changes

The `predict_layer` in `T5GemmaVoiceModel` has been restructured:

**Before (Single-token only):**
```python
self.predict_layer = nn.ModuleList([
    nn.Sequential(...)  # One head per codebook
])
```

**After (Supports multi-token):**
```python
self.predict_layer = nn.ModuleList([
    nn.ModuleList([
        nn.Sequential(...)  # One head per prediction position
        for _ in range(num_predict_tokens)
    ])
    for k in range(n_codebooks)
])
```

### Training Loss Computation

For multi-token prediction with `num_predict_tokens=N`:
- At each decoder position `t`, the model predicts tokens at positions `t`, `t+1`, `t+2`, ..., `t+N-1`
- Prediction head 0: predicts `target[t]` from `hidden[t]`
- Prediction head 1: predicts `target[t+1]` from `hidden[t]`
- Prediction head 2: predicts `target[t+2]` from `hidden[t]`
- etc.

Each prediction head has its own loss, weighted by `multi_token_loss_weight`.

### Backward Compatibility

Models trained with `num_predict_tokens=1` (default) are fully backward compatible with the previous codebase. The prediction layer structure is slightly different (nested ModuleList), but functionally equivalent for single-token mode.

## Best Practices

1. **Start Conservative**: Begin with `num_predict_tokens=2` or `3`. Higher values provide diminishing returns and increase memory usage.

2. **Loss Weighting**: Use decreasing weights for further predictions (e.g., `[1.0, 0.5, 0.25]`) since predicting further ahead is inherently more uncertain.

3. **Memory Considerations**: Multi-token prediction increases model parameters and memory usage proportionally. A model with `num_predict_tokens=3` will have ~3x the prediction head parameters.

4. **Validation**: Monitor validation metrics carefully. Multi-token prediction should improve or maintain performance compared to single-token baselines.

## Testing

A test suite is provided in `test_multi_token.py`:

```bash
python test_multi_token.py
```

This validates:
- Configuration parameters are correctly defined
- Model structure supports both single and multi-token modes
- Target alignment logic is correct

## Troubleshooting

### Issue: Model trained with multi-token won't load on older code
**Solution**: The model structure has changed. Update to the latest code version.

### Issue: Out of memory during training
**Solution**: Reduce `num_predict_tokens` or adjust batch size. Multi-token prediction increases memory usage.

### Issue: No improvement in quality
**Solution**: 
- Try different loss weight configurations
- Ensure you're training for enough steps to see benefits
- Multi-token prediction primarily helps training stability and efficiency, not necessarily final quality

## References

This implementation is inspired by multi-token prediction techniques used in modern language models:
- Speculative Decoding
- Multi-token Generation in Large Language Models
- Auxiliary Loss Training in Sequence Models
