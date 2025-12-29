"""
Test script to verify Triton RoPE fused Flash Attention against PyTorch baseline
"""

import torch
import torch.nn.functional as F
from rope_attn_pytorch import precompute_freqs_cis, attention_pytorch
from flash_attn_rope_triton import attention as attention_triton


def test_rope_attention():
    """Test RoPE attention: Triton vs PyTorch"""

    # Test configurations
    configs = [
        # (BATCH, H, N_CTX, HEAD_DIM, causal)
        (2, 4, 128, 64, True),
        (1, 8, 256, 128, False),
        (4, 16, 512, 64, True),
    ]

    torch.manual_seed(42)
    device = 'cuda'
    dtype = torch.float16
    theta = 10000.0

    for BATCH, H, N_CTX, HEAD_DIM, causal in configs:
        print(f"\n{'='*80}")
        print(f"Testing: BATCH={BATCH}, H={H}, N_CTX={N_CTX}, HEAD_DIM={HEAD_DIM}, causal={causal}")
        print(f"{'='*80}")

        # Generate random inputs
        q = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, HEAD_DIM), dtype=dtype, device=device, requires_grad=True)

        # Clone for separate gradient computation
        q_ref = q.clone().detach().requires_grad_(True)
        k_ref = k.clone().detach().requires_grad_(True)
        v_ref = v.clone().detach().requires_grad_(True)

        # Precompute RoPE frequencies
        freqs_cos, freqs_sin = precompute_freqs_cis(HEAD_DIM, N_CTX, theta, device=device)

        # Scale factor
        sm_scale = 1.0 / (HEAD_DIM ** 0.5)

        # ========== Forward Pass ==========
        print("\nForward Pass:")

        # PyTorch reference
        out_ref = attention_pytorch(q_ref, k_ref, v_ref, causal, sm_scale, freqs_cos, freqs_sin)

        # Triton implementation
        out_tri = attention_triton(q, k, v, causal, sm_scale, freqs_cos, freqs_sin)

        # Compare outputs
        max_diff = torch.max(torch.abs(out_tri - out_ref))
        mean_diff = torch.mean(torch.abs(out_tri - out_ref))

        print(f"Output max diff: {max_diff.item():.6e}")
        print(f"Output mean diff: {mean_diff.item():.6e}")

        # Check if forward pass is close
        try:
            torch.testing.assert_close(out_tri, out_ref, rtol=1e-2, atol=1e-2)
            print("✓ Forward pass: PASSED")
        except AssertionError as e:
            print(f"✗ Forward pass: FAILED\n{e}")
            continue

        # ========== Backward Pass ==========
        print("\nBackward Pass:")

        # Generate random gradient
        grad_out = torch.randn_like(out_ref)

        # PyTorch reference backward
        out_ref.backward(grad_out)
        dq_ref = q_ref.grad.clone()
        dk_ref = k_ref.grad.clone()
        dv_ref = v_ref.grad.clone()

        # Triton backward
        out_tri.backward(grad_out)
        dq_tri = q.grad.clone()
        dk_tri = k.grad.clone()
        dv_tri = v.grad.clone()

        # Compare gradients
        print("\ndQ comparison:")
        dq_max_diff = torch.max(torch.abs(dq_tri - dq_ref))
        dq_mean_diff = torch.mean(torch.abs(dq_tri - dq_ref))
        print(f"  max diff: {dq_max_diff.item():.6e}")
        print(f"  mean diff: {dq_mean_diff.item():.6e}")

        print("\ndK comparison:")
        dk_max_diff = torch.max(torch.abs(dk_tri - dk_ref))
        dk_mean_diff = torch.mean(torch.abs(dk_tri - dk_ref))
        print(f"  max diff: {dk_max_diff.item():.6e}")
        print(f"  mean diff: {dk_mean_diff.item():.6e}")

        print("\ndV comparison:")
        dv_max_diff = torch.max(torch.abs(dv_tri - dv_ref))
        dv_mean_diff = torch.mean(torch.abs(dv_tri - dv_ref))
        print(f"  max diff: {dv_max_diff.item():.6e}")
        print(f"  mean diff: {dv_mean_diff.item():.6e}")

        # Check if backward pass is close
        all_passed = True
        try:
            torch.testing.assert_close(dq_tri, dq_ref, rtol=1e-2, atol=1e-2)
            print("\n✓ dQ: PASSED")
        except AssertionError as e:
            print(f"\n✗ dQ: FAILED\n{e}")
            all_passed = False

        try:
            torch.testing.assert_close(dk_tri, dk_ref, rtol=1e-2, atol=1e-2)
            print("✓ dK: PASSED")
        except AssertionError as e:
            print(f"✗ dK: FAILED\n{e}")
            all_passed = False

        try:
            torch.testing.assert_close(dv_tri, dv_ref, rtol=1e-2, atol=1e-2)
            print("✓ dV: PASSED")
        except AssertionError as e:
            print(f"✗ dV: FAILED\n{e}")
            all_passed = False

        if all_passed:
            print(f"\n{'='*80}")
            print(f"✓✓✓ ALL TESTS PASSED for this configuration ✓✓✓")
            print(f"{'='*80}")
        else:
            print(f"\n{'='*80}")
            print(f"✗✗✗ SOME TESTS FAILED for this configuration ✗✗✗")
            print(f"{'='*80}")


if __name__ == "__main__":
    print("Testing Triton RoPE Fused Flash Attention Implementation")
    print("="*80)
    test_rope_attention()
    print("\n" + "="*80)
    print("Testing complete!")
