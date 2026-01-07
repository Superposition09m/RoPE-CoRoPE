"""
GQA 对拍与性能对比脚本 (Plain Attention)
对比 Triton GQA 实现与 PyTorch 参考实现
"""

import torch
import triton
import sys
import os

# 确保能导入当前目录下的模块
sys.path.insert(0, os.path.dirname(__file__))

from flash_attn_co_rope_gqa_triton import attention as attention_triton
from attn_gqa_pytorch import attention_pytorch


def bench_gqa_comparison():
    print("=" * 70)
    print("GQA 对拍与性能对比 (Plain Attention)")
    print("=" * 70)

    # 1. 配置参数 (GQA 典型配置)
    B = 2
    H_Q = 8
    H_KV = 2
    N = 1024
    D = 128
    causal = True
    sm_scale = 1.0 / (D ** 0.5)
    
    device = 'cuda'
    dtype = torch.float16
    
    print(f"配置: B={B}, H_Q={H_Q}, H_KV={H_KV}, N={N}, D={D}, causal={causal}")
    print(f"Dtype: {dtype}, Device: {device}")

    # 2. 准备数据
    torch.manual_seed(42)
    q = torch.randn(B, H_Q, N, D, device=device, dtype=dtype)
    k = torch.randn(B, H_KV, N, D, device=device, dtype=dtype)
    v = torch.randn(B, H_KV, N, D, device=device, dtype=dtype)

    # 3. 正确性验证 (Forward)
    print("\n[Step 1] 正确性验证...")
    
    # PyTorch 输出
    o_pytorch = attention_pytorch(q, k, v, causal, sm_scale)
    
    # Triton 输出
    try:
        # 显式传入 warp_specialize=False 避免非Blackwell架构编译失败
        o_triton = attention_triton(q, k, v, causal, sm_scale, False)
        
        diff = (o_pytorch - o_triton).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        
        print(f"  Max Diff:  {max_diff:.6f}")
        print(f"  Mean Diff: {mean_diff:.6f}")
        
        if max_diff < 1e-2:
            print("  ✅ 正确性通过 (Max Diff < 0.01)")
        else:
            print("  ❌ 正确性不通过，请检查实现")
    except Exception as e:
        print(f"  ❌ Triton 运行失败: {e}")
        o_triton = None

    # 4. 性能测试 (使用 do_bench 排除编译时间)
    print("\n[Step 2] 性能测速 (排除 Autotune 编译时间)...")
    
    # 定义测试函数
    fn_pytorch = lambda: attention_pytorch(q, k, v, causal, sm_scale)
    fn_triton = lambda: attention_triton(q, k, v, causal, sm_scale, False) if o_triton is not None else None

    # PyTorch 测速
    print("  PyTorch 运行中...")
    ms_pytorch = triton.testing.do_bench(fn_pytorch)
    print(f"  PyTorch: {ms_pytorch:.3f} ms")

    # Triton 测速
    if fn_triton is not None:
        print("  Triton 运行中 (含自动 Autotune)...")
        ms_triton = triton.testing.do_bench(fn_triton)
        print(f"  Triton:  {ms_triton:.3f} ms")
        
        speedup = ms_pytorch / ms_triton
        print(f"  Speedup: {speedup:.2f}x")
    else:
        print("  Triton 测速跳过（失败）")

    print("\n" + "=" * 70)

if __name__ == "__main__":
    bench_gqa_comparison()


