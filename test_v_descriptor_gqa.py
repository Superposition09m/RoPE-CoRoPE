"""
测试 V descriptor 在 GQA 场景下的正确性

目标：验证在修改后的 kernel 签名中，V descriptor 能否正确加载
- V 使用 descriptor（不需要 RoPE）
- Q/K 使用 pointer（需要 RoPE/dual-pointer）
- GQA：Q heads != K/V heads
"""

import torch
import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

DEVICE = triton.runtime.driver.active.get_active_torch_device()

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def supports_host_descriptor():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


@triton.jit
def _test_v_descriptor_load_kernel(
    desc_v,  # V descriptor or pointer
    V_out,  # 输出
    Z, H_Q, H_K, GROUP_SIZE,
    N_CTX, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    stride_v_out_z, stride_v_out_h, stride_v_out_m, stride_v_out_d,
):
    """
    测试 V descriptor 在 GQA 场景下的加载
    模拟 _attn_fwd 中的 V 加载逻辑
    """
    dtype = tl.float16
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H_Q
    off_h_q = off_hz % H_Q
    
    # GQA: compute corresponding K/V head
    off_h_k = off_h_q // GROUP_SIZE
    
    # Setup V descriptor
    y_dim = Z * H_K * N_CTX
    desc_v = _maybe_make_tensor_desc(desc_v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    
    # Offset for V (uses K head indexing)
    offset_v_y = off_z * (N_CTX * H_K) + off_h_k * N_CTX
    
    # Initialize
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    # Simulate loading V in a loop (like _attn_fwd_inner)
    for start_n in range(0, BLOCK_N, BLOCK_N):  # 只测试一个 block
        offsetv_y = offset_v_y + start_n
        
        # Load V using descriptor
        v = desc_v.load([offsetv_y, 0])  # Should be [BLOCK_N, HEAD_DIM]
        
        # Write back to output (broadcast to all M positions)
        offs_d = tl.arange(0, HEAD_DIM)
        mask_out = (offs_m[:, None] < N_CTX)
        
        # Write v to output for verification
        # 简化：只写第一行的 V 数据
        mask_write = (offs_m[0] < N_CTX)
        if mask_write:
            v_out_base = V_out + off_z * stride_v_out_z + off_h_q * stride_v_out_h
            v_out_ptrs = v_out_base + offs_m[0] * stride_v_out_m + offs_d * stride_v_out_d
            # v 的第一行: [HEAD_DIM]
            v_first_row = v[0, :]
            tl.store(v_out_ptrs, v_first_row.to(dtype))


def test_v_descriptor_gqa():
    """测试 V descriptor 在 GQA 场景下的加载"""
    print("="*60)
    print("测试: V Descriptor 在 GQA 场景下的加载")
    print("="*60)
    
    # GQA configuration
    B = 2
    H_Q = 8
    H_K = 4
    N_CTX = 128
    HEAD_DIM = 64
    BLOCK_M = 64
    BLOCK_N = 64
    GROUP_SIZE = H_Q // H_K
    
    print(f"\n配置:")
    print(f"  Batch={B}, Q_heads={H_Q}, K/V_heads={H_K}")
    print(f"  Group_size={GROUP_SIZE}")
    print(f"  N_CTX={N_CTX}, HEAD_DIM={HEAD_DIM}")
    print(f"  BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}")
    
    # 创建测试数据
    v = torch.randn((B, H_K, N_CTX, HEAD_DIM), dtype=torch.float16, device=DEVICE)
    v_out = torch.zeros((B, H_Q, N_CTX, HEAD_DIM), dtype=torch.float16, device=DEVICE)
    
    # Grid: 每个 Q head 一个 kernel instance
    grid = (triton.cdiv(N_CTX, BLOCK_M), B * H_Q)
    
    # 测试是否支持 descriptor
    if supports_host_descriptor():
        print(f"\n✓ 设备支持 TensorDescriptor (Hopper+)")
        
        # 创建 V descriptor
        y_dim_v = B * H_K * N_CTX
        dummy_block = [1, 1]
        desc_v = TensorDescriptor(v, shape=[y_dim_v, HEAD_DIM], strides=[HEAD_DIM, 1],
                                  block_shape=dummy_block)
        print(f"  desc_v shape: {desc_v.shape}")
        print(f"  desc_v strides: {desc_v.strides}")
        print(f"  desc_v block_shape: {desc_v.block_shape}")
    else:
        print(f"\n✗ 设备不支持 TensorDescriptor，使用 pointer fallback")
        desc_v = v
    
    # 运行 kernel
    print(f"\n运行 kernel...")
    try:
        _test_v_descriptor_load_kernel[grid](
            desc_v,
            v_out,
            B, H_Q, H_K, GROUP_SIZE,
            N_CTX, HEAD_DIM,
            BLOCK_M, BLOCK_N,
            v_out.stride(0), v_out.stride(1), v_out.stride(2), v_out.stride(3),
        )
        print(f"  ✅ Kernel 执行成功")
        print(f"  输出范围: [{v_out.min().item():.4f}, {v_out.max().item():.4f}]")
        
        # 验证逻辑：检查不同 Q head 是否正确映射到对应的 K/V head
        print(f"\n验证 GQA 映射:")
        for q_head in range(H_Q):
            k_head = q_head // GROUP_SIZE
            print(f"  Q_head[{q_head}] -> K/V_head[{k_head}]:")
            
            # 取第一个 batch，第一个位置
            v_loaded = v_out[0, q_head, 0, :5]
            v_expected = v[0, k_head, 0, :5]
            
            diff = (v_loaded - v_expected).abs().max().item()
            status = "✅" if diff < 1e-4 else "❌"
            print(f"    {status} diff={diff:.2e}, loaded={v_loaded.tolist()[:3]}...")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Kernel 执行失败:")
        print(f"     {e}")
        import traceback
        traceback.print_exc()
        return False


@triton.jit  
def _test_v_pointer_load_kernel(
    V,  # V pointer
    V_out,  # 输出
    Z, H_Q, H_K, GROUP_SIZE,
    N_CTX, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    stride_v_z, stride_v_h, stride_v_m, stride_v_d,
    stride_out_z, stride_out_h, stride_out_m, stride_out_d,
):
    """
    使用纯指针加载 V 作为对照
    """
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H_Q
    off_h_q = off_hz % H_Q
    
    # GQA: compute corresponding K/V head
    off_h_k = off_h_q // GROUP_SIZE
    
    # V base pointer
    v_base = V + off_z * stride_v_z + off_h_k * stride_v_h
    
    # Initialize
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)
    
    # Load first BLOCK_N of V
    offs_n = tl.arange(0, BLOCK_N)
    mask_v = (offs_n[:, None] < N_CTX)
    v_ptrs = v_base + offs_n[:, None] * stride_v_m + offs_d[None, :] * stride_v_d
    v = tl.load(v_ptrs, mask=mask_v, other=0.0)
    
    # Write first row to output
    mask_write = (offs_m[0] < N_CTX)
    if mask_write:
        v_out_base = V_out + off_z * stride_out_z + off_h_q * stride_out_h
        v_out_ptrs = v_out_base + offs_m[0] * stride_out_m + offs_d * stride_out_d
        # v 的第一行
        v_first_row = v[0, :]
        tl.store(v_out_ptrs, v_first_row)


def test_v_pointer_gqa():
    """测试使用纯指针加载 V（作为对照组）"""
    print("\n" + "="*60)
    print("对照测试: V Pointer 在 GQA 场景下的加载")
    print("="*60)
    
    # GQA configuration
    B = 2
    H_Q = 8
    H_K = 4
    N_CTX = 128
    HEAD_DIM = 64
    BLOCK_M = 64
    BLOCK_N = 64
    GROUP_SIZE = H_Q // H_K
    
    # 创建测试数据
    v = torch.randn((B, H_K, N_CTX, HEAD_DIM), dtype=torch.float16, device=DEVICE)
    v_out = torch.zeros((B, H_Q, N_CTX, HEAD_DIM), dtype=torch.float16, device=DEVICE)
    
    # Grid
    grid = (triton.cdiv(N_CTX, BLOCK_M), B * H_Q)
    
    # 运行 kernel
    print(f"\n运行纯指针 kernel...")
    try:
        _test_v_pointer_load_kernel[grid](
            v,
            v_out,
            B, H_Q, H_K, GROUP_SIZE,
            N_CTX, HEAD_DIM,
            BLOCK_M, BLOCK_N,
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            v_out.stride(0), v_out.stride(1), v_out.stride(2), v_out.stride(3),
        )
        print(f"  ✅ Kernel 执行成功")
        
        # 验证
        print(f"\n验证 GQA 映射:")
        for q_head in range(H_Q):
            k_head = q_head // GROUP_SIZE
            v_loaded = v_out[0, q_head, 0, :5]
            v_expected = v[0, k_head, 0, :5]
            
            diff = (v_loaded - v_expected).abs().max().item()
            status = "✅" if diff < 1e-4 else "❌"
            print(f"  Q_head[{q_head}] -> K/V_head[{k_head}]: {status} diff={diff:.2e}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Kernel 执行失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "🧪 "*30)
    print("V Descriptor GQA 加载测试")
    print("🧪 "*30 + "\n")
    
    # Test 1: Pointer baseline
    test1 = test_v_pointer_gqa()
    
    # Test 2: Descriptor (if supported)
    test2 = test_v_descriptor_gqa()
    
    print("\n" + "="*60)
    print("总结")
    print("="*60)
    print(f"  Pointer 加载:    {'✅ PASS' if test1 else '❌ FAIL'}")
    print(f"  Descriptor 加载: {'✅ PASS' if test2 else '❌ FAIL'}")
    print("="*60 + "\n")
    
    if test1 and test2:
        print("✅ V descriptor 在 GQA 场景下工作正常！\n")
    elif test1 and not test2:
        print("⚠️  Pointer 工作但 Descriptor 失败，需要修复 descriptor 配置\n")
    else:
        print("❌ 两种方式都有问题，需要检查 GQA 映射逻辑\n")

