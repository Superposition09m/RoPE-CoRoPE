"""
Co-RoPE 组件级测试
类似于 test_descriptor_to_pointer_rope.py 的风格
逐个验证 Co-RoPE Triton 实现的各个关键组件
"""

import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()

def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"

def supports_host_descriptor():
    return is_cuda() and torch.cuda.get_device_capability()[0] >= 9


# ========================================
# 测试 1: inv_freq 计算
# ========================================
@triton.jit
def _test_inv_freq_kernel(
    inv_freq_out,
    theta,
    HEAD_DIM: tl.constexpr,
):
    """测试 inv_freq 的动态计算"""
    half_dim: tl.constexpr = HEAD_DIM // 2
    offs_d = tl.arange(0, half_dim)
    
    # 计算 inv_freq = 1.0 / (theta ** (2*offs_d / HEAD_DIM))
    # 注意：PyTorch 使用 arange(0, HEAD_DIM, 2)，即 [0, 2, 4, 6, ...]
    exponent = (2 * offs_d).to(tl.float32) / HEAD_DIM
    inv_freq = 1.0 / tl.exp(exponent * tl.log(theta))
    
    # 写回
    tl.store(inv_freq_out + offs_d, inv_freq)


def test_inv_freq():
    """测试 inv_freq 计算的正确性"""
    print("="*60)
    print("测试 1: inv_freq 动态计算")
    print("="*60)
    
    HEAD_DIM = 64
    theta = 10000.0
    half_dim = HEAD_DIM // 2
    
    # Triton 计算
    inv_freq_triton = torch.zeros(half_dim, dtype=torch.float32, device=DEVICE)
    _test_inv_freq_kernel[(1,)](inv_freq_triton, theta, HEAD_DIM)
    
    # PyTorch 参考
    inv_freq_ref = 1.0 / (theta ** (torch.arange(0, HEAD_DIM, 2, device=DEVICE).float() / HEAD_DIM))
    
    # 对比
    diff = (inv_freq_triton - inv_freq_ref).abs().max().item()
    print(f"  Triton: {inv_freq_triton[:5].tolist()}")
    print(f"  PyTorch: {inv_freq_ref[:5].tolist()}")
    print(f"  Max Diff: {diff:.2e}")
    
    if diff < 1e-5:
        print("  ✅ PASS: inv_freq 计算正确")
        return True
    else:
        print("  ❌ FAIL: inv_freq 计算有误")
        return False


# ========================================
# 测试 2: Phase 1 里程计算
# ========================================
@triton.jit
def _test_mileage_phase1_kernel(
    Q, K,
    a_tt_out,
    sm_scale,
    N_CTX, HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    stride_q_seq, stride_q_dim,
    stride_k_seq, stride_k_dim,
):
    """测试 Phase 1 的里程计算逻辑"""
    pid = tl.program_id(0)
    
    half_dim: tl.constexpr = HEAD_DIM // 2
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d_first = tl.arange(0, half_dim)
    offs_d_second = offs_d_first + half_dim
    
    # Load Q
    mask_q = (offs_m[:, None] < N_CTX)
    q1_ptrs = Q + offs_m[:, None] * stride_q_seq + offs_d_first[None, :] * stride_q_dim
    q2_ptrs = Q + offs_m[:, None] * stride_q_seq + offs_d_second[None, :] * stride_q_dim
    q1 = tl.load(q1_ptrs, mask=mask_q, other=0.0)
    q2 = tl.load(q2_ptrs, mask=mask_q, other=0.0)
    
    # Initialize diagonal cumulative mileage
    a_tt = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # Scan all K blocks
    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n_curr = start_n + tl.arange(0, BLOCK_N)
        mask_k = (offs_n_curr[:, None] < N_CTX)
        
        # Load K
        k1_ptrs = K + offs_n_curr[:, None] * stride_k_seq + offs_d_first[None, :] * stride_k_dim
        k2_ptrs = K + offs_n_curr[:, None] * stride_k_seq + offs_d_second[None, :] * stride_k_dim
        k1 = tl.load(k1_ptrs, mask=mask_k, other=0.0)
        k2 = tl.load(k2_ptrs, mask=mask_k, other=0.0)
        
        # Raw dot product (no RoPE)
        qk_raw = tl.dot(q1, tl.trans(k1)) + tl.dot(q2, tl.trans(k2))
        
        # z = sigmoid(qk * sm_scale)
        z_block = tl.sigmoid(qk_raw * sm_scale)
        
        # Accumulate only diagonal and below
        mask_diagonal = offs_m[:, None] >= offs_n_curr[None, :]
        z_masked = tl.where(mask_diagonal, z_block, 0.0)
        
        # Sum across K dimension
        a_tt = a_tt + tl.sum(z_masked, axis=1)
    
    # Write back
    mask_out = (offs_m < N_CTX)
    tl.store(a_tt_out + offs_m, a_tt, mask=mask_out)


def test_mileage_phase1():
    """测试 Phase 1 里程计算是否与 PyTorch 一致"""
    print("\n" + "="*60)
    print("测试 2: Phase 1 里程计算")
    print("="*60)
    
    N_CTX = 128
    HEAD_DIM = 64
    BLOCK_M = 64
    BLOCK_N = 64
    sm_scale = (HEAD_DIM ** -0.5)
    
    # 创建测试数据
    q = torch.randn((N_CTX, HEAD_DIM), dtype=torch.float16, device=DEVICE)
    k = torch.randn((N_CTX, HEAD_DIM), dtype=torch.float16, device=DEVICE)
    
    # Triton 计算
    a_tt_triton = torch.zeros(N_CTX, dtype=torch.float32, device=DEVICE)
    grid = (triton.cdiv(N_CTX, BLOCK_M),)
    _test_mileage_phase1_kernel[grid](
        q, k, a_tt_triton, sm_scale,
        N_CTX, HEAD_DIM, BLOCK_M, BLOCK_N,
        q.stride(0), q.stride(1),
        k.stride(0), k.stride(1),
    )
    
    # PyTorch 参考
    qk = torch.matmul(q, k.T) * sm_scale  # [N_CTX, N_CTX]
    z = torch.sigmoid(qk)
    
    # 对角线累积：a_tt[i] = sum_{j=0}^{i} z[i,j]
    # 使用 tril mask
    mask_tril = torch.tril(torch.ones(N_CTX, N_CTX, device=DEVICE, dtype=torch.bool))
    z_masked = torch.where(mask_tril, z, 0.0)
    a_tt_ref = z_masked.sum(dim=1)
    
    # 对比
    diff = (a_tt_triton - a_tt_ref).abs().max().item()
    print(f"  Triton a_tt 范围: [{a_tt_triton.min().item():.4f}, {a_tt_triton.max().item():.4f}]")
    print(f"  PyTorch a_tt 范围: [{a_tt_ref.min().item():.4f}, {a_tt_ref.max().item():.4f}]")
    print(f"  Max Diff: {diff:.2e}")
    
    if diff < 1e-3:
        print("  ✅ PASS: Phase 1 里程计算正确")
        return True
    else:
        print("  ❌ FAIL: Phase 1 里程计算有误")
        # 详细对比前几个元素
        print(f"\n  详细对比 (前10个元素):")
        for i in range(min(10, N_CTX)):
            print(f"    a_tt[{i}]: Triton={a_tt_triton[i].item():.6f}, PyTorch={a_tt_ref[i].item():.6f}, diff={abs(a_tt_triton[i].item()-a_tt_ref[i].item()):.2e}")
        return False


# ========================================
# 测试 3: tl.cumsum 行为验证
# ========================================
@triton.jit
def _test_cumsum_kernel(
    X,
    Y_out,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """测试 tl.cumsum 的行为"""
    pid = tl.program_id(0)
    
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    # Load
    x = tl.load(X + offs_m[:, None] * BLOCK_N + offs_n[None, :])
    
    # Cumsum along axis=1
    y = tl.cumsum(x, axis=1)
    
    # Store
    tl.store(Y_out + offs_m[:, None] * BLOCK_N + offs_n[None, :], y)


def test_cumsum_behavior():
    """验证 tl.cumsum 的行为与 torch.cumsum 一致"""
    print("\n" + "="*60)
    print("测试 3: tl.cumsum 行为验证")
    print("="*60)
    
    BLOCK_M = 4
    BLOCK_N = 8
    
    # 创建测试数据
    x = torch.randn((BLOCK_M, BLOCK_N), dtype=torch.float32, device=DEVICE)
    
    # Triton 计算
    y_triton = torch.zeros_like(x)
    _test_cumsum_kernel[(1,)](x, y_triton, BLOCK_M, BLOCK_N)
    
    # PyTorch 参考
    y_ref = torch.cumsum(x, dim=1)
    
    # 对比
    diff = (y_triton - y_ref).abs().max().item()
    print(f"  Input:\n{x}")
    print(f"\n  Triton cumsum:\n{y_triton}")
    print(f"\n  PyTorch cumsum:\n{y_ref}")
    print(f"\n  Max Diff: {diff:.2e}")
    
    if diff < 1e-5:
        print("  ✅ PASS: tl.cumsum 行为正确")
        return True
    else:
        print("  ❌ FAIL: tl.cumsum 行为异常")
        return False


# ========================================
# 测试 4: Co-RoPE 能量场计算
# ========================================
@triton.jit
def _test_energy_field_kernel(
    Q1, Q2, K1, K2,
    delta_a, inv_freq,
    qk_out,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """测试 Co-RoPE 能量场和相位调制的计算"""
    half_dim: tl.constexpr = HEAD_DIM // 2
    
    # Load data
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, half_dim)
    
    q1 = tl.load(Q1 + offs_m[:, None] * half_dim + offs_d[None, :])
    q2 = tl.load(Q2 + offs_m[:, None] * half_dim + offs_d[None, :])
    k1 = tl.load(K1 + offs_n[:, None] * half_dim + offs_d[None, :])
    k2 = tl.load(K2 + offs_n[:, None] * half_dim + offs_d[None, :])
    
    delta = tl.load(delta_a + offs_m[:, None] * BLOCK_N + offs_n[None, :])
    inv_f = tl.load(inv_freq + offs_d)
    
    # Compute phi
    phi = delta[:, :, None] * inv_f[None, None, :]  # [BLOCK_M, BLOCK_N, half_dim]
    cos_phi = tl.cos(phi)
    sin_phi = tl.sin(phi)
    
    # Compute energy fields
    E_A = q1[:, None, :] * k1[None, :, :] + q2[:, None, :] * k2[None, :, :]
    E_B = q2[:, None, :] * k1[None, :, :] - q1[:, None, :] * k2[None, :, :]
    
    # Co-RoPE score
    qk = tl.sum(E_A * cos_phi - E_B * sin_phi, axis=2)
    
    # Store
    tl.store(qk_out + offs_m[:, None] * BLOCK_N + offs_n[None, :], qk)


def test_energy_field():
    """测试 Co-RoPE 能量场计算"""
    print("\n" + "="*60)
    print("测试 4: Co-RoPE 能量场和相位调制")
    print("="*60)
    
    BLOCK_M = 4
    BLOCK_N = 8
    HEAD_DIM = 64
    half_dim = HEAD_DIM // 2
    theta = 10000.0
    
    # 创建测试数据
    q = torch.randn((BLOCK_M, HEAD_DIM), dtype=torch.float32, device=DEVICE)
    k = torch.randn((BLOCK_N, HEAD_DIM), dtype=torch.float32, device=DEVICE)
    delta_a = torch.randn((BLOCK_M, BLOCK_N), dtype=torch.float32, device=DEVICE)
    
    q1, q2 = q[:, :half_dim], q[:, half_dim:]
    k1, k2 = k[:, :half_dim], k[:, half_dim:]
    
    inv_freq = 1.0 / (theta ** (torch.arange(0, HEAD_DIM, 2, device=DEVICE).float() / HEAD_DIM))
    
    # Triton 计算
    qk_triton = torch.zeros((BLOCK_M, BLOCK_N), dtype=torch.float32, device=DEVICE)
    _test_energy_field_kernel[(1,)](
        q1.contiguous(), q2.contiguous(),
        k1.contiguous(), k2.contiguous(),
        delta_a.contiguous(), inv_freq,
        qk_triton,
        BLOCK_M, BLOCK_N, HEAD_DIM,
    )
    
    # PyTorch 参考
    phi = delta_a.unsqueeze(-1) * inv_freq.view(1, 1, -1)  # [BLOCK_M, BLOCK_N, half_dim]
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    
    E_A = q1.unsqueeze(1) * k1.unsqueeze(0) + q2.unsqueeze(1) * k2.unsqueeze(0)
    E_B = q2.unsqueeze(1) * k1.unsqueeze(0) - q1.unsqueeze(1) * k2.unsqueeze(0)
    
    qk_ref = (E_A * cos_phi - E_B * sin_phi).sum(dim=-1)
    
    # 对比
    diff = (qk_triton - qk_ref).abs().max().item()
    print(f"  Triton qk 范围: [{qk_triton.min().item():.4f}, {qk_triton.max().item():.4f}]")
    print(f"  PyTorch qk 范围: [{qk_ref.min().item():.4f}, {qk_ref.max().item():.4f}]")
    print(f"  Max Diff: {diff:.2e}")
    
    if diff < 1e-4:
        print("  ✅ PASS: 能量场计算正确")
        return True
    else:
        print("  ❌ FAIL: 能量场计算有误")
        print(f"\n  详细对比 (第一行):")
        print(f"    Triton:  {qk_triton[0, :].tolist()}")
        print(f"    PyTorch: {qk_ref[0, :].tolist()}")
        return False


# ========================================
# 测试 5: 增量累积 + tl.cumsum
# ========================================
@triton.jit
def _test_incremental_cumsum_kernel(
    Q1, Q2, K1, K2,
    a_tt_in,
    delta_a_out,
    sm_scale,
    N_CTX, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
    stride_q_seq, stride_q_dim,
    stride_k_seq, stride_k_dim,
):
    """测试增量累积 + tl.cumsum 的组合逻辑"""
    pid = tl.program_id(0)
    
    half_dim: tl.constexpr = HEAD_DIM // 2
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d_first = tl.arange(0, half_dim)
    
    # Load Q
    mask_q = (offs_m[:, None] < N_CTX)
    q1 = tl.load(Q1 + offs_m[:, None] * half_dim + offs_d_first[None, :], mask=mask_q, other=0.0)
    q2 = tl.load(Q2 + offs_m[:, None] * half_dim + offs_d_first[None, :], mask=mask_q, other=0.0)
    
    # Load a_tt
    a_tt = tl.load(a_tt_in + offs_m)
    
    # Simulate Phase 2: loop over K blocks
    a_cum = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    block_idx = 0
    for start_n in range(0, N_CTX, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_k = (offs_n[:, None] < N_CTX)
        
        # Load K
        k1 = tl.load(K1 + offs_n[:, None] * half_dim + offs_d_first[None, :], mask=mask_k, other=0.0)
        k2 = tl.load(K2 + offs_n[:, None] * half_dim + offs_d_first[None, :], mask=mask_k, other=0.0)
        
        # Compute mileage
        qk_mile = tl.dot(q1, tl.trans(k1)) + tl.dot(q2, tl.trans(k2))
        z_tile = tl.sigmoid(qk_mile * sm_scale)
        
        # Cumsum within block
        z_cumsum = tl.cumsum(z_tile, axis=1)
        
        # Current accumulated mileage
        a_current = a_cum[:, None] + z_cumsum
        
        # Compute delta_a
        delta = a_tt[:, None] - a_current
        
        # Store first block's delta_a for verification
        if block_idx == 0:
            mask_store = (offs_m[:, None] < N_CTX) & (offs_n[None, :] < BLOCK_N)
            tl.store(delta_a_out + offs_m[:, None] * BLOCK_N + offs_n[None, :], delta, mask=mask_store)
        
        # Update cumulative
        a_cum = a_cum + tl.sum(z_tile, axis=1)
        
        block_idx += 1


def test_incremental_cumsum():
    """测试增量累积 + tl.cumsum 的正确性"""
    print("\n" + "="*60)
    print("测试 5: 增量累积 + tl.cumsum")
    print("="*60)
    
    N_CTX = 128
    HEAD_DIM = 64
    BLOCK_M = 64
    BLOCK_N = 64
    half_dim = HEAD_DIM // 2
    sm_scale = (HEAD_DIM ** -0.5)
    
    # 创建测试数据
    q = torch.randn((N_CTX, HEAD_DIM), dtype=torch.float32, device=DEVICE)
    k = torch.randn((N_CTX, HEAD_DIM), dtype=torch.float32, device=DEVICE)
    q1, q2 = q[:, :half_dim].contiguous(), q[:, half_dim:].contiguous()
    k1, k2 = k[:, :half_dim].contiguous(), k[:, half_dim:].contiguous()
    
    # 先计算 a_tt (使用 PyTorch)
    qk = torch.matmul(q, k.T) * sm_scale
    z = torch.sigmoid(qk)
    mask_tril = torch.tril(torch.ones(N_CTX, N_CTX, device=DEVICE, dtype=torch.bool))
    a_tt_ref = torch.where(mask_tril, z, 0.0).sum(dim=1)
    
    # Triton 计算
    delta_a_triton = torch.zeros((N_CTX, BLOCK_N), dtype=torch.float32, device=DEVICE)
    grid = (triton.cdiv(N_CTX, BLOCK_M),)
    _test_incremental_cumsum_kernel[grid](
        q1, q2, k1, k2,
        a_tt_ref,
        delta_a_triton,
        sm_scale,
        N_CTX, BLOCK_M, BLOCK_N, HEAD_DIM,
        half_dim, 1,
        half_dim, 1,
    )
    
    # PyTorch 参考 (只验证第一个block)
    # 计算第一个 block 的 cumsum
    z_first_block = z[:, :BLOCK_N]
    z_cumsum_ref = torch.cumsum(z_first_block, dim=1)
    a_current_ref = z_cumsum_ref  # 第一个 block，a_cum=0
    delta_a_ref = a_tt_ref[:, None] - a_current_ref
    
    # 对比
    diff = (delta_a_triton - delta_a_ref).abs().max().item()
    print(f"  Max Diff: {diff:.2e}")
    
    if diff < 1e-4:
        print("  ✅ PASS: 增量累积 + cumsum 逻辑正确")
        return True
    else:
        print("  ❌ FAIL: 增量累积 + cumsum 逻辑有误")
        print(f"\n  详细对比 (第一行):")
        print(f"    Triton:  {delta_a_triton[0, :].tolist()}")
        print(f"    PyTorch: {delta_a_ref[0, :].tolist()}")
        return False


# ========================================
# Main Test Suite
# ========================================
if __name__ == "__main__":
    print("\n" + "🧪 "*30)
    print("Co-RoPE 组件级测试套件")
    print("🧪 "*30 + "\n")
    
    results = []
    
    # Test 1: inv_freq
    results.append(("inv_freq 计算", test_inv_freq()))
    
    # Test 2: Phase 1 mileage
    results.append(("Phase 1 里程计算", test_mileage_phase1()))
    
    # Test 3: tl.cumsum
    results.append(("tl.cumsum 行为", test_cumsum_behavior()))
    
    # Test 4: Energy field
    results.append(("能量场计算", test_energy_field()))
    
    # Test 5: Incremental cumsum
    results.append(("增量累积 + cumsum", test_incremental_cumsum()))
    
    # Summary
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(p for _, p in results)
    print("\n" + "="*60)
    if all_passed:
        print("✅ 所有组件测试通过！")
        print("📝 下一步：集成到完整的 Co-RoPE Attention kernel")
    else:
        print("❌ 部分组件测试失败，需要修复")
    print("="*60 + "\n")

