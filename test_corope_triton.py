"""
Test script to compare Co-RoPE PyTorch and Triton implementations

This script validates the Triton implementation against the PyTorch reference.
"""

import torch
import sys
from corope_attn_gqa_pytorch import attention_pytorch
from flash_attn_co_rope_gqa_triton import attention as attention_triton

def test_corope_triton_vs_pytorch():
    """Test Co-RoPE Triton implementation against PyTorch reference"""
    
    print("="*60)
    print("Co-RoPE Triton vs PyTorch Comparison Test")
    print("="*60)
    
    # Test configuration
    B, H_Q, H_K, N, D = 2, 8, 4, 128, 64
    group_size = H_Q // H_K
    theta = 10000.0
    sm_scale = D ** -0.5
    causal = True
    device = 'cuda'
    dtype = torch.float16
    
    print(f"\nTest Configuration:")
    print(f"  Batch: {B}, Q_heads: {H_Q}, K_heads: {H_K}, Seq_len: {N}, Head_dim: {D}")
    print(f"  Group size: {group_size}")
    print(f"  Theta: {theta}")
    print(f"  Scale: {sm_scale:.6f}")
    print(f"  Causal: {causal}")
    print(f"  Dtype: {dtype}")
    
    # Create input tensors
    torch.manual_seed(42)
    q = torch.randn(B, H_Q, N, D, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(B, H_K, N, D, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(B, H_K, N, D, device=device, dtype=dtype, requires_grad=True)
    
    print(f"\nInput shapes:")
    print(f"  Q: {q.shape}")
    print(f"  K: {k.shape}")
    print(f"  V: {v.shape}")
    
    # PyTorch reference
    print(f"\n[1/2] Running PyTorch reference implementation...")
    try:
        out_pytorch = attention_pytorch(q, k, v, causal, sm_scale, theta)
        print(f"  ✓ PyTorch output shape: {out_pytorch.shape}")
        print(f"  ✓ PyTorch output range: [{out_pytorch.min().item():.4f}, {out_pytorch.max().item():.4f}]")
    except Exception as e:
        print(f"  ✗ PyTorch failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Triton implementation
    print(f"\n[2/2] Running Triton implementation...")
    try:
        out_triton = attention_triton(q, k, v, causal, sm_scale, theta, False)
        print(f"  ✓ Triton output shape: {out_triton.shape}")
        print(f"  ✓ Triton output range: [{out_triton.min().item():.4f}, {out_triton.max().item():.4f}]")
    except Exception as e:
        print(f"  ✗ Triton failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Compare results
    print(f"\n{'='*60}")
    print("Results Comparison")
    print(f"{'='*60}")
    
    diff = (out_pytorch - out_triton).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    
    print(f"\nAbsolute Difference:")
    print(f"  Max:  {max_diff:.2e}")
    print(f"  Mean: {mean_diff:.2e}")
    
    # Relative difference (avoid division by zero)
    rel_diff = diff / (out_pytorch.abs() + 1e-8)
    max_rel_diff = rel_diff.max().item()
    mean_rel_diff = rel_diff.mean().item()
    
    print(f"\nRelative Difference:")
    print(f"  Max:  {max_rel_diff:.2e}")
    print(f"  Mean: {mean_rel_diff:.2e}")
    
    # Pass/Fail criteria
    atol = 1e-2  # Absolute tolerance for fp16
    rtol = 1e-1  # Relative tolerance
    
    print(f"\nTolerance Thresholds:")
    print(f"  Absolute: {atol:.2e}")
    print(f"  Relative: {rtol:.2e}")
    
    passed = (max_diff < atol) or (max_rel_diff < rtol)
    
    print(f"\n{'='*60}")
    if passed:
        print("✅ TEST PASSED: Triton matches PyTorch within tolerance!")
    else:
        print("❌ TEST FAILED: Triton output differs from PyTorch")
        print(f"\n   Max absolute diff ({max_diff:.2e}) >= threshold ({atol:.2e})")
        print(f"   Max relative diff ({max_rel_diff:.2e}) >= threshold ({rtol:.2e})")
    print(f"{'='*60}\n")
    
    return passed


def test_gqa_configurations():
    """Test multiple GQA configurations"""
    
    print("\n" + "="*60)
    print("Testing Multiple GQA Configurations")
    print("="*60)
    
    configs = [
        ("MHA", 8, 8),
        ("GQA-2", 8, 4),
        ("GQA-4", 8, 2),
        ("MQA", 8, 1),
    ]
    
    results = []
    
    for name, h_q, h_k in configs:
        print(f"\n[{name}] Q_heads={h_q}, K_heads={h_k}, group_size={h_q//h_k}")
        
        B, N, D = 2, 64, 64
        theta = 10000.0
        sm_scale = D ** -0.5
        
        torch.manual_seed(42)
        q = torch.randn(B, h_q, N, D, device='cuda', dtype=torch.float16)
        k = torch.randn(B, h_k, N, D, device='cuda', dtype=torch.float16)
        v = torch.randn(B, h_k, N, D, device='cuda', dtype=torch.float16)
        
        try:
            out_pytorch = attention_pytorch(q, k, v, True, sm_scale, theta)
            out_triton = attention_triton(q, k, v, True, sm_scale, theta, False)
            
            diff = (out_pytorch - out_triton).abs().max().item()
            passed = diff < 1e-2
            
            results.append((name, passed, diff))
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {status}: max_diff = {diff:.2e}")
            
        except Exception as e:
            results.append((name, False, float('inf')))
            print(f"  ❌ ERROR: {e}")
    
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for name, passed, diff in results:
        status = "✅" if passed else "❌"
        print(f"  {status} {name:10s}: {diff:.2e}")
    
    all_passed = all(p for _, p, _ in results)
    print(f"\n{'='*60}")
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print(f"{'='*60}\n")
    
    return all_passed


if __name__ == "__main__":
    print("\n" + "🧪 "*30)
    print("Co-RoPE Triton Implementation Test Suite")
    print("🧪 "*30 + "\n")
    
    # Run main test
    test1_passed = test_corope_triton_vs_pytorch()
    
    # Run GQA configuration tests
    test2_passed = test_gqa_configurations()
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL TEST SUMMARY")
    print("="*60)
    print(f"  Basic Test:     {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"  GQA Tests:      {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print("="*60)
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED! 🎉\n")
        sys.exit(0)
    else:
        print("\n❌ SOME TESTS FAILED ❌\n")
        sys.exit(1)

