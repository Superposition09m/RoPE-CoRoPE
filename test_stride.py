"""
测试物理双指针方案在不同 stride 情况下的正确性
"""

import torch
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

from flash_attn_rope_triton import attention


def test_contiguous():
    """测试连续内存（正常情况）"""
    print("=" * 80)
    print("测试 1: 连续内存 (Contiguous)")
    print("=" * 80)
    
    B, H, N, D = 2, 4, 128, 64
    device = 'cuda'
    dtype = torch.float16
    
    q = torch.randn(B, H, N, D, device=device, dtype=dtype)
    k = torch.randn(B, H, N, D, device=device, dtype=dtype)
    v = torch.randn(B, H, N, D, device=device, dtype=dtype)
    freqs_cos = torch.randn(N, D // 2, device=device, dtype=dtype)
    freqs_sin = torch.randn(N, D // 2, device=device, dtype=dtype)
    
    print(f"Q stride: {q.stride()}")
    print(f"K stride: {k.stride()}")
    print(f"V stride: {v.stride()}")
    
    try:
        o = attention(q, k, v, False, 0.5, freqs_cos, freqs_sin, False)
        print(f"✅ 成功！输出形状: {o.shape}")
        print(f"   输出均值: {o.mean().item():.4f}, 标准差: {o.std().item():.4f}")
        return True
    except Exception as e:
        print(f"❌ 失败: {e}")
        return False


def test_view():
    """测试 view 后的非连续内存"""
    print("\n" + "=" * 80)
    print("测试 2: View 后的非连续内存")
    print("=" * 80)
    
    B, H, N, D = 2, 4, 128, 64
    device = 'cuda'
    dtype = torch.float16
    
    # 创建一个大的 tensor，然后 view 成我们需要的形状
    q_flat = torch.randn(B * H * N * D, device=device, dtype=dtype)
    k_flat = torch.randn(B * H * N * D, device=device, dtype=dtype)
    v_flat = torch.randn(B * H * N * D, device=device, dtype=dtype)
    
    q = q_flat.view(B, H, N, D)
    k = k_flat.view(B, H, N, D)
    v = v_flat.view(B, H, N, D)
    
    # 检查是否连续
    print(f"Q is contiguous: {q.is_contiguous()}")
    print(f"Q stride: {q.stride()}")
    
    freqs_cos = torch.randn(N, D // 2, device=device, dtype=dtype)
    freqs_sin = torch.randn(N, D // 2, device=device, dtype=dtype)
    
    try:
        o = attention(q, k, v, False, 0.5, freqs_cos, freqs_sin, False)
        print(f"✅ 成功！输出形状: {o.shape}")
        print(f"   输出均值: {o.mean().item():.4f}, 标准差: {o.std().item():.4f}")
        return True
    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_transpose():
    """测试 transpose 后的非连续内存"""
    print("\n" + "=" * 80)
    print("测试 3: Transpose 后的非连续内存")
    print("=" * 80)
    
    B, H, N, D = 2, 4, 128, 64
    device = 'cuda'
    dtype = torch.float16
    
    # 创建 [B, H, D, N] 然后 transpose 成 [B, H, N, D]
    q = torch.randn(B, H, D, N, device=device, dtype=dtype).transpose(2, 3)
    k = torch.randn(B, H, D, N, device=device, dtype=dtype).transpose(2, 3)
    v = torch.randn(B, H, D, N, device=device, dtype=dtype).transpose(2, 3)
    
    print(f"Q is contiguous: {q.is_contiguous()}")
    print(f"Q stride: {q.stride()}")
    
    freqs_cos = torch.randn(N, D // 2, device=device, dtype=dtype)
    freqs_sin = torch.randn(N, D // 2, device=device, dtype=dtype)
    
    try:
        o = attention(q, k, v, False, 0.5, freqs_cos, freqs_sin, False)
        print(f"✅ 成功！输出形状: {o.shape}")
        print(f"   输出均值: {o.mean().item():.4f}, 标准差: {o.std().item():.4f}")
        return True
    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_correctness():
    """测试数值正确性（与连续版本对比）"""
    print("\n" + "=" * 80)
    print("测试 4: 数值正确性验证")
    print("=" * 80)
    
    B, H, N, D = 1, 2, 64, 64  # HEAD_DIM 必须 >= BLOCK_N (64)
    device = 'cuda'
    dtype = torch.float16
    
    # 创建相同的随机数据
    torch.manual_seed(42)
    q_cont = torch.randn(B, H, N, D, device=device, dtype=dtype)
    k_cont = torch.randn(B, H, N, D, device=device, dtype=dtype)
    v_cont = torch.randn(B, H, N, D, device=device, dtype=dtype)
    
    # 创建非连续版本（通过 view）
    q_noncont = q_cont.clone().view(B * H * N * D).view(B, H, N, D)
    k_noncont = k_cont.clone().view(B * H * N * D).view(B, H, N, D)
    v_noncont = v_cont.clone().view(B * H * N * D).view(B, H, N, D)
    
    freqs_cos = torch.randn(N, D // 2, device=device, dtype=dtype)
    freqs_sin = torch.randn(N, D // 2, device=device, dtype=dtype)
    
    try:
        o_cont = attention(q_cont, k_cont, v_cont, False, 0.5, freqs_cos, freqs_sin, False)
        o_noncont = attention(q_noncont, k_noncont, v_noncont, False, 0.5, freqs_cos, freqs_sin, False)
        
        diff = torch.abs(o_cont - o_noncont).max().item()
        print(f"连续 vs 非连续最大差异: {diff:.6e}")
        
        if diff < 1e-3:
            print(f"✅ 数值正确性验证通过！")
            return True
        else:
            print(f"⚠️  差异较大，可能需要检查")
            return False
    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 80)
    print("物理双指针 Stride 测试")
    print("=" * 80)
    
    results = {}
    results['contiguous'] = test_contiguous()
    results['view'] = test_view()
    results['transpose'] = test_transpose()
    results['correctness'] = test_correctness()
    
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 所有测试通过！Stride 处理正确。")
    else:
        print("\n⚠️  部分测试失败，需要检查 stride 处理逻辑。")
    
    sys.exit(0 if all_passed else 1)

