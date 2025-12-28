"""
Full stack test: Compare our RoPE + Attention implementation with HuggingFace Llama
Tests both forward and backward passes for correctness
"""

import torch
import torch.nn.functional as F
from rope_attn_pytorch import precompute_freqs_cis, apply_rotary_emb, attention_pytorch

try:
    from transformers.models.llama.modeling_llama import (
        LlamaRotaryEmbedding,
        apply_rotary_pos_emb as hf_apply_rotary_pos_emb,
    )
    HF_AVAILABLE = True
except ImportError:
    print("WARNING: transformers not installed. Run: pip install transformers")
    HF_AVAILABLE = False


class TestConfig:
    """Test configuration"""
    def __init__(self, batch, heads, seq_len, head_dim, theta=10000.0, causal=True, device='cuda'):
        self.batch = batch
        self.heads = heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.theta = theta
        self.causal = causal
        self.device = device

    def __repr__(self):
        return (f"TestConfig(batch={self.batch}, heads={self.heads}, "
                f"seq_len={self.seq_len}, head_dim={self.head_dim}, "
                f"theta={self.theta}, causal={self.causal})")


def create_random_inputs(config, requires_grad=True, dtype=torch.float32):
    """Create random Q, K, V tensors for testing"""
    torch.manual_seed(42)

    q = torch.randn(
        config.batch, config.heads, config.seq_len, config.head_dim,
        device=config.device, dtype=dtype, requires_grad=requires_grad
    )
    k = torch.randn(
        config.batch, config.heads, config.seq_len, config.head_dim,
        device=config.device, dtype=dtype, requires_grad=requires_grad
    )
    v = torch.randn(
        config.batch, config.heads, config.seq_len, config.head_dim,
        device=config.device, dtype=dtype, requires_grad=requires_grad
    )

    return q, k, v


def compute_numerical_error(tensor1, tensor2, name=""):
    """Compute and display numerical errors between two tensors"""
    abs_diff = torch.abs(tensor1 - tensor2)
    rel_diff = abs_diff / (torch.abs(tensor2) + 1e-8)

    error_info = {
        'max_abs_error': abs_diff.max().item(),
        'mean_abs_error': abs_diff.mean().item(),
        'max_rel_error': rel_diff.max().item(),
        'mean_rel_error': rel_diff.mean().item(),
    }

    print(f"\n{'='*60}")
    print(f"Numerical Error Report: {name}")
    print(f"{'='*60}")
    print(f"Max Absolute Error:  {error_info['max_abs_error']:.2e}")
    print(f"Mean Absolute Error: {error_info['mean_abs_error']:.2e}")
    print(f"Max Relative Error:  {error_info['max_rel_error']:.2e}")
    print(f"Mean Relative Error: {error_info['mean_rel_error']:.2e}")
    print(f"{'='*60}\n")

    return error_info


def get_hf_rope_embeddings(config):
    """Get RoPE embeddings using HuggingFace implementation"""
    rope = LlamaRotaryEmbedding(
        dim=config.head_dim,
        max_position_embeddings=config.seq_len,
        base=config.theta,
        device=config.device,
    )

    # HF returns cos, sin in shape (seq_len, head_dim)
    position_ids = torch.arange(config.seq_len, device=config.device).unsqueeze(0)
    cos, sin = rope(torch.zeros(1, config.seq_len, config.head_dim, device=config.device), position_ids)

    # cos, sin shape: (1, seq_len, head_dim)
    cos = cos.squeeze(0)  # (seq_len, head_dim)
    sin = sin.squeeze(0)  # (seq_len, head_dim)

    return cos, sin


def hf_apply_rope_to_qk(q, k, cos, sin):
    """
    Apply RoPE using HuggingFace's apply_rotary_pos_emb

    Args:
        q: (batch, heads, seq_len, head_dim)
        k: (batch, heads, seq_len, head_dim)
        cos: (seq_len, head_dim)
        sin: (seq_len, head_dim)

    Returns:
        q_rotated, k_rotated: (batch, heads, seq_len, head_dim)
    """
    # HF expects cos, sin in shape (batch, seq_len, head_dim) or (1, seq_len, head_dim)
    cos = cos.unsqueeze(0)  # (1, seq_len, head_dim)
    sin = sin.unsqueeze(0)  # (1, seq_len, head_dim)

    # HF expects q, k in shape (batch, seq_len, heads, head_dim)
    # We need to transpose
    batch, heads, seq_len, head_dim = q.shape
    q_t = q.transpose(1, 2)  # (batch, seq_len, heads, head_dim)
    k_t = k.transpose(1, 2)  # (batch, seq_len, heads, head_dim)

    # Apply RoPE
    q_rotated, k_rotated = hf_apply_rotary_pos_emb(q_t, k_t, cos, sin)

    # Transpose back
    q_rotated = q_rotated.transpose(1, 2)  # (batch, heads, seq_len, head_dim)
    k_rotated = k_rotated.transpose(1, 2)  # (batch, heads, seq_len, head_dim)

    return q_rotated, k_rotated


def test_precompute_freqs_cis(config, atol=1e-6):
    """Test 1: Compare our freqs computation with HuggingFace"""
    print(f"\n{'#'*60}")
    print(f"TEST 1: precompute_freqs_cis")
    print(f"Config: {config}")
    print(f"{'#'*60}")

    # Our implementation
    freqs_cos_ours, freqs_sin_ours = precompute_freqs_cis(
        dim=config.head_dim,
        seq_len=config.seq_len,
        theta=config.theta,
        device=config.device
    )

    # HuggingFace implementation
    freqs_cos_hf, freqs_sin_hf = get_hf_rope_embeddings(config)

    # Compare
    cos_error = compute_numerical_error(freqs_cos_ours, freqs_cos_hf, "freqs_cos")
    sin_error = compute_numerical_error(freqs_sin_ours, freqs_sin_hf, "freqs_sin")

    # Check if errors are within tolerance
    passed = (cos_error['max_abs_error'] < atol and sin_error['max_abs_error'] < atol)

    if passed:
        print(f"✓ TEST 1 PASSED: freqs computation matches HuggingFace (atol={atol})")
    else:
        print(f"✗ TEST 1 FAILED: freqs computation differs from HuggingFace")

    return passed, freqs_cos_ours, freqs_sin_ours, freqs_cos_hf, freqs_sin_hf


def test_apply_rotary_emb_forward(config, freqs_cos_ours, freqs_sin_ours,
                                   freqs_cos_hf, freqs_sin_hf, atol=1e-5):
    """Test 2: Compare RoPE application (forward only)"""
    print(f"\n{'#'*60}")
    print(f"TEST 2: apply_rotary_emb (Forward)")
    print(f"Config: {config}")
    print(f"{'#'*60}")

    # Create random Q, K
    q, k, _ = create_random_inputs(config, requires_grad=False)

    # Our implementation
    q_rotated_ours = apply_rotary_emb(q, freqs_cos_ours, freqs_sin_ours)
    k_rotated_ours = apply_rotary_emb(k, freqs_cos_ours, freqs_sin_ours)

    # HuggingFace implementation
    q_rotated_hf, k_rotated_hf = hf_apply_rope_to_qk(q, k, freqs_cos_hf, freqs_sin_hf)

    # Compare
    q_error = compute_numerical_error(q_rotated_ours, q_rotated_hf, "Q rotated")
    k_error = compute_numerical_error(k_rotated_ours, k_rotated_hf, "K rotated")

    # Check if errors are within tolerance
    passed = (q_error['max_abs_error'] < atol and k_error['max_abs_error'] < atol)

    if passed:
        print(f"✓ TEST 2 PASSED: RoPE application matches HuggingFace (atol={atol})")
    else:
        print(f"✗ TEST 2 FAILED: RoPE application differs from HuggingFace")

    return passed


def test_apply_rotary_emb_backward(config, freqs_cos_ours, freqs_sin_ours, atol=1e-4):
    """Test 3: Test RoPE backward pass using gradcheck"""
    print(f"\n{'#'*60}")
    print(f"TEST 3: apply_rotary_emb (Backward - Gradcheck)")
    print(f"Config: {config}")
    print(f"{'#'*60}")

    # Create small random input for gradcheck (gradcheck is slow)
    small_config = TestConfig(
        batch=2, heads=2, seq_len=8, head_dim=config.head_dim,
        theta=config.theta, causal=config.causal, device=config.device
    )

    q, _, _ = create_random_inputs(small_config, requires_grad=True, dtype=torch.float64)

    freqs_cos_small, freqs_sin_small = precompute_freqs_cis(
        dim=small_config.head_dim,
        seq_len=small_config.seq_len,
        theta=small_config.theta,
        device=small_config.device
    )
    freqs_cos_small = freqs_cos_small.to(torch.float64)
    freqs_sin_small = freqs_sin_small.to(torch.float64)

    # Define function for gradcheck
    def rope_func(x):
        return apply_rotary_emb(x, freqs_cos_small, freqs_sin_small)

    # Run gradcheck
    try:
        passed = torch.autograd.gradcheck(
            rope_func, q, eps=1e-6, atol=atol, rtol=1e-3,
            raise_exception=False
        )

        if passed:
            print(f"✓ TEST 3 PASSED: RoPE gradcheck succeeded (atol={atol})")
        else:
            print(f"✗ TEST 3 FAILED: RoPE gradcheck failed")

    except Exception as e:
        print(f"✗ TEST 3 FAILED with exception: {e}")
        passed = False

    return passed


def compute_reference_attention(q, k, v, causal, sm_scale, freqs_cos_hf, freqs_sin_hf):
    """
    Compute reference attention output using HuggingFace RoPE + manual attention
    This is the ground truth for comparison
    """
    # Apply HF RoPE
    q_rotated, k_rotated = hf_apply_rope_to_qk(q, k, freqs_cos_hf, freqs_sin_hf)

    # Compute attention manually
    # attn_scores = Q @ K^T * sm_scale
    attn_scores = torch.einsum('bhqd,bhkd->bhqk', q_rotated, k_rotated) * sm_scale

    # Apply causal mask
    if causal:
        seq_len = q.shape[2]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1)
        causal_mask = mask.unsqueeze(0).unsqueeze(0)
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))

    # Softmax
    attn_weights = F.softmax(attn_scores, dim=-1)

    # Apply mask to weights
    if causal:
        attn_weights = attn_weights.masked_fill(causal_mask, 0.0)

    # Output
    output = torch.einsum('bhqk,bhkd->bhqd', attn_weights, v)

    return output


def test_attention_forward(config, freqs_cos_ours, freqs_sin_ours,
                           freqs_cos_hf, freqs_sin_hf, atol=1e-5, rtol=1e-4):
    """Test 4: Full attention forward pass"""
    print(f"\n{'#'*60}")
    print(f"TEST 4: Attention Forward (Full Stack)")
    print(f"Config: {config}")
    print(f"{'#'*60}")

    # Create random inputs
    q, k, v = create_random_inputs(config, requires_grad=False)

    sm_scale = 1.0 / (config.head_dim ** 0.5)

    # Our implementation
    output_ours = attention_pytorch(q, k, v, config.causal, sm_scale, freqs_cos_ours, freqs_sin_ours)

    # Reference implementation
    output_ref = compute_reference_attention(q, k, v, config.causal, sm_scale, freqs_cos_hf, freqs_sin_hf)

    # Compare
    error = compute_numerical_error(output_ours, output_ref, "Attention Output")

    # Check if errors are within tolerance
    passed = (error['max_abs_error'] < atol or error['max_rel_error'] < rtol)

    if passed:
        print(f"✓ TEST 4 PASSED: Attention forward matches reference (atol={atol}, rtol={rtol})")
    else:
        print(f"✗ TEST 4 FAILED: Attention forward differs from reference")

    return passed


def test_attention_backward(config, freqs_cos_ours, freqs_sin_ours,
                            freqs_cos_hf, freqs_sin_hf, atol=1e-4, rtol=1e-3):
    """Test 5: Full attention backward pass"""
    print(f"\n{'#'*60}")
    print(f"TEST 5: Attention Backward (Full Stack)")
    print(f"Config: {config}")
    print(f"{'#'*60}")

    # Create random inputs with gradient tracking
    q_ours, k_ours, v_ours = create_random_inputs(config, requires_grad=True)
    q_ref, k_ref, v_ref = create_random_inputs(config, requires_grad=True)

    sm_scale = 1.0 / (config.head_dim ** 0.5)

    # Our implementation
    output_ours = attention_pytorch(q_ours, k_ours, v_ours, config.causal, sm_scale,
                                    freqs_cos_ours, freqs_sin_ours)

    # Reference implementation
    output_ref = compute_reference_attention(q_ref, k_ref, v_ref, config.causal, sm_scale,
                                             freqs_cos_hf, freqs_sin_hf)

    # Create same grad_output for both
    grad_output = torch.randn_like(output_ours)

    # Backward pass
    output_ours.backward(grad_output)
    output_ref.backward(grad_output)

    # Compare gradients
    dq_error = compute_numerical_error(q_ours.grad, q_ref.grad, "dQ")
    dk_error = compute_numerical_error(k_ours.grad, k_ref.grad, "dK")
    dv_error = compute_numerical_error(v_ours.grad, v_ref.grad, "dV")

    # Check if errors are within tolerance
    passed = (
        (dq_error['max_abs_error'] < atol or dq_error['max_rel_error'] < rtol) and
        (dk_error['max_abs_error'] < atol or dk_error['max_rel_error'] < rtol) and
        (dv_error['max_abs_error'] < atol or dv_error['max_rel_error'] < rtol)
    )

    if passed:
        print(f"✓ TEST 5 PASSED: Attention backward matches reference (atol={atol}, rtol={rtol})")
    else:
        print(f"✗ TEST 5 FAILED: Attention backward differs from reference")

    return passed


def run_all_tests():
    """Run all tests with multiple configurations"""
    if not HF_AVAILABLE:
        print("ERROR: HuggingFace transformers not available. Cannot run tests.")
        return

    print("\n" + "="*60)
    print("FULL STACK ROPE + ATTENTION TEST SUITE")
    print("="*60)

    # Test configurations
    test_configs = [
        TestConfig(batch=2, heads=4, seq_len=128, head_dim=64, theta=10000.0, causal=True),
        TestConfig(batch=4, heads=8, seq_len=256, head_dim=128, theta=10000.0, causal=False),
        TestConfig(batch=1, heads=2, seq_len=512, head_dim=64, theta=500000.0, causal=True),
    ]

    all_results = []

    for i, config in enumerate(test_configs):
        print(f"\n{'='*60}")
        print(f"TEST SUITE {i+1}/{len(test_configs)}")
        print(f"{'='*60}")

        results = {'config': str(config)}

        # Test 1: precompute_freqs_cis
        passed, freqs_cos_ours, freqs_sin_ours, freqs_cos_hf, freqs_sin_hf = test_precompute_freqs_cis(config)
        results['test1_freqs'] = passed

        # Test 2: apply_rotary_emb forward
        passed = test_apply_rotary_emb_forward(config, freqs_cos_ours, freqs_sin_ours,
                                               freqs_cos_hf, freqs_sin_hf)
        results['test2_rope_forward'] = passed

        # Test 3: apply_rotary_emb backward
        passed = test_apply_rotary_emb_backward(config, freqs_cos_ours, freqs_sin_ours)
        results['test3_rope_backward'] = passed

        # Test 4: attention forward
        passed = test_attention_forward(config, freqs_cos_ours, freqs_sin_ours,
                                       freqs_cos_hf, freqs_sin_hf)
        results['test4_attn_forward'] = passed

        # Test 5: attention backward
        passed = test_attention_backward(config, freqs_cos_ours, freqs_sin_ours,
                                        freqs_cos_hf, freqs_sin_hf)
        results['test5_attn_backward'] = passed

        all_results.append(results)

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for i, results in enumerate(all_results):
        print(f"\nTest Suite {i+1}:")
        print(f"  Config: {results['config']}")
        print(f"  Test 1 (freqs):          {'PASS' if results['test1_freqs'] else 'FAIL'}")
        print(f"  Test 2 (RoPE forward):   {'PASS' if results['test2_rope_forward'] else 'FAIL'}")
        print(f"  Test 3 (RoPE backward):  {'PASS' if results['test3_rope_backward'] else 'FAIL'}")
        print(f"  Test 4 (Attn forward):   {'PASS' if results['test4_attn_forward'] else 'FAIL'}")
        print(f"  Test 5 (Attn backward):  {'PASS' if results['test5_attn_backward'] else 'FAIL'}")

    # Overall result
    all_passed = all(
        all(results[key] for key in results if key != 'config')
        for results in all_results
    )

    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED - Please review errors above")
    print("="*60 + "\n")

    return all_passed


if __name__ == "__main__":
    run_all_tests()
