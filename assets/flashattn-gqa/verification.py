"""
GQA 正确性对拍脚本 (Plain GQA) - 支持 Forward & Backward
对比 flash_attn_co_rope_gqa_triton.py 与 attn_gqa_pytorch.py
"""

import torch
import sys
import os

# 确保能导入当前目录下的模块
sys.path.insert(0, os.path.dirname(__file__))

from triton_gqa import attention as attention_triton
from attn_gqa_pytorch import attention_pytorch

def verify_gqa(dtype=torch.float32, atol=1e-2, rtol=1e-4):
    print("=" * 60)
    print(f"GQA 正确性验证 (Forward & Backward) | Dtype: {dtype}")
    print("=" * 60)

    # 使用与原始 flash_attn_v2 测试一致的参数
    # 原始代码设计假设 N_CTX >= 128
    B = 1
    H_Q = 4
    H_KV = 2  # GROUP_SIZE = 2
    N = 128    # 改为 128，符合原始代码的设计假设
    D = 64     # 改为 64，与原始测试一致
    causal = True
    sm_scale = 1.0 / (D ** 0.5)
    
    device = 'cuda'
    
    print(f"配置: B={B}, H_Q={H_Q}, H_KV={H_KV}, N={N}, D={D}, causal={causal}")

    # 2. 构造输入 (开启 requires_grad)
    torch.manual_seed(1234)
    q = torch.randn(B, H_Q, N, D, device=device, dtype=dtype).requires_grad_(True)
    k = torch.randn(B, H_KV, N, D, device=device, dtype=dtype).requires_grad_(True)
    v = torch.randn(B, H_KV, N, D, device=device, dtype=dtype).requires_grad_(True)
    
    # 3. 运行 PyTorch 参考实现
    print("\n[PyTorch] 计算中...")
    o_pytorch = attention_pytorch(q, k, v, causal, sm_scale)
    # 计算梯度
    dout = torch.randn_like(o_pytorch)
    o_pytorch.backward(dout)
    dq_ref, dk_ref, dv_ref = q.grad.clone(), k.grad.clone(), v.grad.clone()
    
    # 清除梯度准备下一步
    q.grad.zero_()
    k.grad.zero_()
    v.grad.zero_()

    # 4. 运行 Triton 实现
    print("[Triton] 计算中...")
    try:
        # GQA 路径下显式传入 warp_specialize=False
        o_triton = attention_triton(q, k, v, causal, sm_scale, False)
        # 计算梯度
        o_triton.backward(dout)
        dq_tri, dk_tri, dv_tri = q.grad.clone(), k.grad.clone(), v.grad.clone()
    except Exception as e:
        print(f"  ❌ Triton 运行失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. 对比结果
    print("\n" + "-" * 40)
    print("Forward 对比:")
    fwd_diff = (o_pytorch - o_triton).abs().max().item()
    print(f"  Max Diff: {fwd_diff:.2e}")
    
    print("\nBackward 对比:")
    dq_diff = (dq_ref - dq_tri).abs().max().item()
    dk_diff = (dk_ref - dk_tri).abs().max().item()
    dv_diff = (dv_ref - dv_tri).abs().max().item()
    print(f"  dQ Max Diff: {dq_diff:.2e}")
    print(f"  dK Max Diff: {dk_diff:.2e}")
    print(f"  dV Max Diff: {dv_diff:.2e}")

    # 综合检查
    passed = True
    import math
    if math.isnan(fwd_diff) or fwd_diff > atol: passed = False
    if math.isnan(dq_diff) or math.isnan(dk_diff) or math.isnan(dv_diff): passed = False
    if dq_diff > atol or dk_diff > atol or dv_diff > atol: passed = False
    
    if passed:
        print(f"\n✨ [PASS] 对拍成功！")
    else:
        print(f"\n⚠️ [FAIL] 对拍差异过大！")
    print("-" * 40)

if __name__ == "__main__":
    # FP16 是推荐的 dtype，精度符合预期
    verify_gqa(dtype=torch.float16, atol=5e-3)
    
    # FP32 的数值误差会稍大一些（因为 kernel 内部使用混合精度）
    verify_gqa(dtype=torch.float32, atol=1e-2)
