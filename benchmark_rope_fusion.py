"""
Benchmark: Fused RoPE + Flash Attention vs. No RoPE Flash Attention
Compare performance to show the benefit of fusing RoPE into the kernel
"""

import torch
import triton

# Import both versions
from flash_attn_v2_triton import attention as attention_no_rope
from flash_attn_rope_triton import attention as attention_fused_rope

def benchmark_config(BATCH, N_HEADS, SEQ_LEN, HEAD_DIM, CAUSAL=True, THETA=10000.0, device='cuda', dtype=torch.float16):
    """Benchmark a single configuration"""
    
    # Create inputs
    q = torch.randn(BATCH, N_HEADS, SEQ_LEN, HEAD_DIM, device=device, dtype=dtype, requires_grad=False)
    k = torch.randn(BATCH, N_HEADS, SEQ_LEN, HEAD_DIM, device=device, dtype=dtype, requires_grad=False)
    v = torch.randn(BATCH, N_HEADS, SEQ_LEN, HEAD_DIM, device=device, dtype=dtype, requires_grad=False)
    
    sm_scale = 1.0 / (HEAD_DIM ** 0.5)
    
    # Warmup
    for _ in range(10):
        _ = attention_no_rope(q, k, v, CAUSAL, sm_scale, False)  # Use positional args
        _ = attention_fused_rope(q, k, v, CAUSAL, sm_scale, THETA, False)  # Use positional args
    
    torch.cuda.synchronize()
    
    # Benchmark No RoPE version
    fn_no_rope = lambda: attention_no_rope(q, k, v, CAUSAL, sm_scale, False)  # Use positional args
    ms_no_rope = triton.testing.do_bench(fn_no_rope, warmup=10, rep=100)
    
    # Benchmark Fused RoPE version
    fn_fused_rope = lambda: attention_fused_rope(q, k, v, CAUSAL, sm_scale, THETA, False)  # Use positional args
    ms_fused_rope = triton.testing.do_bench(fn_fused_rope, warmup=10, rep=100)
    
    # Calculate speedup
    speedup = ms_no_rope / ms_fused_rope
    overhead_pct = ((ms_fused_rope - ms_no_rope) / ms_no_rope) * 100
    
    return ms_no_rope, ms_fused_rope, speedup, overhead_pct


def main():
    print("\n" + "="*80)
    print("Benchmark: Fused RoPE + Flash Attention vs. No RoPE Flash Attention")
    print("="*80)
    
    # Test configurations
    configs = [
        # (BATCH, N_HEADS, SEQ_LEN, HEAD_DIM, name)
        (2, 8, 512, 64, "Small (B=2, H=8, S=512, D=64)"),
        (4, 16, 1024, 64, "Medium (B=4, H=16, S=1024, D=64)"),
        (8, 32, 2048, 64, "Large (B=8, H=32, S=2048, D=64)"),
        (2, 8, 512, 128, "Small D=128 (B=2, H=8, S=512, D=128)"),
        (4, 16, 1024, 128, "Medium D=128 (B=4, H=16, S=1024, D=128)"),
    ]
    
    print("\nConfiguration | No RoPE (ms) | Fused RoPE (ms) | Overhead | Note")
    print("-" * 80)
    
    results = []
    for BATCH, N_HEADS, SEQ_LEN, HEAD_DIM, name in configs:
        try:
            ms_no_rope, ms_fused_rope, speedup, overhead_pct = benchmark_config(
                BATCH, N_HEADS, SEQ_LEN, HEAD_DIM
            )
            
            # Determine if overhead is acceptable
            if overhead_pct < 5:
                note = "✓ Excellent"
            elif overhead_pct < 10:
                note = "✓ Good"
            elif overhead_pct < 20:
                note = "⚠ Acceptable"
            else:
                note = "✗ High overhead"
            
            print(f"{name:20s} | {ms_no_rope:11.3f} | {ms_fused_rope:14.3f} | {overhead_pct:+7.1f}% | {note}")
            results.append((name, ms_no_rope, ms_fused_rope, overhead_pct))
            
        except Exception as e:
            print(f"{name:20s} | ERROR: {str(e)[:50]}")
    
    print("-" * 80)
    
    # Summary
    if results:
        avg_overhead = sum(r[3] for r in results) / len(results)
        print(f"\nSummary:")
        print(f"  Average Overhead: {avg_overhead:+.1f}%")
        print(f"  Total Configs Tested: {len(results)}")
        
        if avg_overhead < 10:
            print(f"  Result: ✓ RoPE fusion adds minimal overhead (<10%)")
        elif avg_overhead < 20:
            print(f"  Result: ⚠ RoPE fusion adds moderate overhead (10-20%)")
        else:
            print(f"  Result: ✗ RoPE fusion adds significant overhead (>20%)")
    
    print("\n" + "="*80)
    print("Note: Overhead = (Fused RoPE time - No RoPE time) / No RoPE time")
    print("      The goal is to minimize overhead by fusing RoPE into the kernel.")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

