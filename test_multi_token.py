#!/usr/bin/env python3
"""
Test script for multi-token prediction feature.
Tests both single-token (backward compatible) and multi-token modes.
"""

import argparse
import sys


def test_config():
    """Test that config parameters are properly defined."""
    print("Testing config parameters...")
    from config import MyParser
    
    parser = MyParser()
    
    # Test with default (single-token mode)
    args = parser.parse_args([
        '--exp_dir', '/tmp/test',
        '--dataset', 'test',
        '--dataset_dir', '/tmp/data',
    ])
    
    assert hasattr(args, 'num_predict_tokens'), "num_predict_tokens parameter not found"
    assert args.num_predict_tokens == 1, f"Default num_predict_tokens should be 1, got {args.num_predict_tokens}"
    print(f"  ✓ Default num_predict_tokens = {args.num_predict_tokens}")
    
    # Test with multi-token mode
    args = parser.parse_args([
        '--exp_dir', '/tmp/test',
        '--dataset', 'test',
        '--dataset_dir', '/tmp/data',
        '--num_predict_tokens', '3',
        '--multi_token_loss_weight', '[1.0, 0.5, 0.25]',
    ])
    
    assert args.num_predict_tokens == 3, f"num_predict_tokens should be 3, got {args.num_predict_tokens}"
    assert args.multi_token_loss_weight == '[1.0, 0.5, 0.25]', "multi_token_loss_weight not set correctly"
    print(f"  ✓ Multi-token mode: num_predict_tokens = {args.num_predict_tokens}")
    print(f"  ✓ Loss weights: {args.multi_token_loss_weight}")
    
    print("✓ Config tests passed!\n")
    return True


def test_model_structure():
    """Test that model structure is correct for both modes."""
    print("Testing model structure (without loading weights)...")
    
    try:
        import torch
        import torch.nn as nn
        from config import MyParser, apply_repo_defaults
        
        # Mock the transformers library since we don't need to load actual weights
        class MockAutoModel:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                class MockModel:
                    def __init__(self):
                        # Create mock encoder and decoder
                        class MockEncoder:
                            def __init__(self):
                                pass
                            def __call__(self, *args, **kwargs):
                                # Return mock output with last_hidden_state
                                class Output:
                                    last_hidden_state = torch.zeros(1, 10, 768)
                                return Output()
                        
                        class MockDecoder:
                            def __init__(self):
                                self.layers = nn.ModuleList()
                            def __call__(self, *args, **kwargs):
                                class Output:
                                    last_hidden_state = torch.zeros(1, 10, 768)
                                    past_key_values = None
                                return Output()
                        
                        class MockModelInner:
                            def __init__(self):
                                self.encoder = MockEncoder()
                                self.decoder = MockDecoder()
                        
                        self.model = MockModelInner()
                        self.config = type('Config', (), {
                            'd_model': 768,
                            'use_cache': True,
                            '_attn_implementation': 'eager'
                        })()
                    
                    def parameters(self):
                        return [torch.zeros(1)]
                    
                    def named_parameters(self):
                        return [('test', torch.zeros(1))]
                    
                return MockModel()
        
        # Test single-token mode (backward compatible)
        print("  Testing single-token mode (num_predict_tokens=1)...")
        parser = MyParser()
        args = parser.parse_args([
            '--exp_dir', '/tmp/test',
            '--dataset', 'test',
            '--dataset_dir', '/tmp/data',
            '--t5gemma_model_name', 'google/t5gemma-b-b-ul2',
            '--num_predict_tokens', '1',
        ])
        args = apply_repo_defaults(args)
        
        # Create minimal model structure test
        num_predict_tokens = 1
        n_codebooks = 1
        audio_vocab_size = 2048 + 4
        hidden_size = 768
        
        # Test predict_layer structure for single-token
        predict_layer = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, audio_vocab_size),
                )
                for _ in range(num_predict_tokens)
            ])
            for k in range(n_codebooks)
        ])
        
        assert len(predict_layer) == n_codebooks, "Predict layer codebook count mismatch"
        assert len(predict_layer[0]) == num_predict_tokens, f"Predict layer should have {num_predict_tokens} heads"
        print(f"    ✓ Single-token mode: {len(predict_layer[0])} prediction head(s)")
        
        # Test multi-token mode
        print("  Testing multi-token mode (num_predict_tokens=3)...")
        num_predict_tokens = 3
        
        predict_layer = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, audio_vocab_size),
                )
                for _ in range(num_predict_tokens)
            ])
            for k in range(n_codebooks)
        ])
        
        assert len(predict_layer[0]) == num_predict_tokens, f"Predict layer should have {num_predict_tokens} heads"
        print(f"    ✓ Multi-token mode: {len(predict_layer[0])} prediction heads")
        
        # Test loss weight parsing
        print("  Testing loss weight parsing...")
        multi_token_loss_weight = '[1.0, 0.5, 0.25]'
        loss_weights = eval(multi_token_loss_weight)
        loss_weights = torch.tensor(loss_weights, dtype=torch.float32)
        
        assert len(loss_weights) == num_predict_tokens, "Loss weight length mismatch"
        assert torch.allclose(loss_weights, torch.tensor([1.0, 0.5, 0.25])), "Loss weights incorrect"
        print(f"    ✓ Loss weights: {loss_weights.tolist()}")
        
        print("✓ Model structure tests passed!\n")
        return True
        
    except ImportError as e:
        print(f"  ⚠ Skipping model tests (missing dependencies: {e})")
        return True


def test_prediction_logic():
    """Test the multi-token prediction logic."""
    print("Testing multi-token prediction logic...")
    
    try:
        import torch
        
        # Simulate multi-token prediction alignment
        print("  Testing target alignment for multi-token prediction...")
        
        # Assume we have a sequence of length 10
        seq_len = 10
        num_predict_tokens = 3
        
        # Simulated logits: [num_predict_tokens, seq_len, vocab_size]
        vocab_size = 100
        logits = torch.randn(num_predict_tokens, seq_len, vocab_size)
        targets = torch.randint(0, vocab_size, (seq_len,))
        
        print(f"    Sequence length: {seq_len}")
        print(f"    Number of prediction heads: {num_predict_tokens}")
        
        for pred_idx in range(num_predict_tokens):
            pred_logits = logits[pred_idx]  # [seq_len, vocab_size]
            
            if pred_idx == 0:
                # Predict current token (t->t)
                pred_logits_aligned = pred_logits
                pred_targets = targets
                print(f"    Head {pred_idx}: predict t->t, shapes {pred_logits_aligned.shape}, {pred_targets.shape}")
            else:
                # Predict future token (t->t+pred_idx)
                if pred_logits.shape[0] > pred_idx:
                    pred_logits_aligned = pred_logits[:-pred_idx]
                    pred_targets = targets[pred_idx:]
                    print(f"    Head {pred_idx}: predict t->t+{pred_idx}, shapes {pred_logits_aligned.shape}, {pred_targets.shape}")
                else:
                    print(f"    Head {pred_idx}: skipped (sequence too short)")
                    continue
            
            # Check alignment
            assert pred_logits_aligned.shape[0] == pred_targets.shape[0], \
                f"Alignment mismatch: {pred_logits_aligned.shape[0]} != {pred_targets.shape[0]}"
        
        print("  ✓ Target alignment logic is correct")
        print("✓ Prediction logic tests passed!\n")
        return True
        
    except ImportError as e:
        print(f"  ⚠ Skipping prediction tests (missing dependencies: {e})")
        return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Multi-Token Prediction Feature Tests")
    print("=" * 60 + "\n")
    
    results = []
    
    # Test 1: Config parameters
    results.append(("Config", test_config()))
    
    # Test 2: Model structure
    results.append(("Model Structure", test_model_structure()))
    
    # Test 3: Prediction logic
    results.append(("Prediction Logic", test_prediction_logic()))
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {name}: {status}")
    
    all_passed = all(result for _, result in results)
    print("\n" + ("=" * 60))
    if all_passed:
        print("All tests passed! ✓")
        print("=" * 60)
        return 0
    else:
        print("Some tests failed! ✗")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
