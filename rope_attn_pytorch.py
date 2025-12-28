"""
PyTorch implementation of RoPE attention for benchmarking
This is used as a baseline to compare with Triton implementation
"""

import torch
import torch.nn.functional as F


def precompute_freqs_cis(dim, seq_len, theta, device='cuda'):
    """
    Precompute cos and sin values for RoPE
    Since this is PyTorch version, we cache the freqs instead of computing them in a fused kernel

    Args:
        dim: head_dim, must be even
        seq_len: sequence length
        theta: base for frequency computation (e.g., 10000.0)
        device: device to create tensors on

    Returns:
        freqs_cos: (seq_len, dim) - cos values, each frequency repeated twice
        freqs_sin: (seq_len, dim) - sin values, each frequency repeated twice
    """
    assert dim % 2 == 0, f"dim must be even, got {dim}"

    # Compute frequencies: theta_i = base^(-2i/dim), i in [0, dim//2 - 1]
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    # freqs shape: (dim // 2,)

    # Position indices: m in [0, seq_len - 1]
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    # t shape: (seq_len,)

    # Compute m * theta_i
    freqs = torch.outer(t, freqs)  # (seq_len, dim // 2)

    # Compute cos and sin
    freqs_cos = torch.cos(freqs)  # (seq_len, dim // 2)
    freqs_sin = torch.sin(freqs)  # (seq_len, dim // 2)

    # Repeat each value twice: (seq_len, dim // 2) -> (seq_len, dim)
    freqs_cos = torch.repeat_interleave(freqs_cos, 2, dim=1)
    freqs_sin = torch.repeat_interleave(freqs_sin, 2, dim=1)

    return freqs_cos, freqs_sin


def rotate_half(x):
    """
    Rearrange x to (-x_2, x_1, -x_4, x_3, ..., -x_d, x_{d-1})
    This swaps and negates pairs for the rotation operation

    Args:
        x: (..., dim) where dim is even

    Returns:
        rotated: (..., dim)
    """
    x1 = x[..., ::2]   # Even indices: x_0, x_2, x_4, ...
    x2 = x[..., 1::2]  # Odd indices: x_1, x_3, x_5, ...

    # Interleave: (-x_1, x_0, -x_3, x_2, ...)
    rotated = torch.stack([-x2, x1], dim=-1).flatten(-2)
    return rotated


def apply_rotary_emb(x, freqs_cos, freqs_sin):
    """
    Apply rotary position embedding
    Formula: x * cos(m*theta) + rotate_half(x) * sin(m*theta)

    Args:
        x: (BATCH, H, N_CTX, HEAD_DIM)
        freqs_cos: (N_CTX, HEAD_DIM)
        freqs_sin: (N_CTX, HEAD_DIM)

    Returns:
        rotated: (BATCH, H, N_CTX, HEAD_DIM)
    """
    return x * freqs_cos + rotate_half(x) * freqs_sin


class _attention_pytorch(torch.autograd.Function):
    """
    Plain PyTorch Attention with manual backward pass
    Interface compatible with flash_attn_v2_triton.py
    """
    
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, freqs_cos, freqs_sin):
        """
        Args:
            q: (BATCH, H, N_CTX, HEAD_DIM)
            k: (BATCH, H, N_CTX, HEAD_DIM)
            v: (BATCH, H, N_CTX, HEAD_DIM)
            causal: bool
            sm_scale: float, scaling factor for attention scores
            freqs_cos: (N_CTX, HEAD_DIM) - precomputed cos values for RoPE
            freqs_sin: (N_CTX, HEAD_DIM) - precomputed sin values for RoPE
        Returns:
            output: (BATCH, H, N_CTX, HEAD_DIM)
        """
        # Apply RoPE to Q and K (V does not need RoPE)
        q = apply_rotary_emb(q, freqs_cos, freqs_sin)
        k = apply_rotary_emb(k, freqs_cos, freqs_sin)

        # Compute attention scores: Q @ K^T * sm_scale
        attn_scores = torch.einsum('bhqd,bhkd->bhqk', q, k) * sm_scale
        
        # Apply causal mask if needed
        if causal:
            N_CTX = q.shape[2]
            mask = torch.triu(torch.ones(N_CTX, N_CTX, device=q.device, dtype=torch.bool), diagonal=1)
            causal_mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, N_CTX, N_CTX)
            attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # Apply mask to weights (set masked positions to 0)
        if causal:
            attn_weights = attn_weights.masked_fill(causal_mask, 0.0)
        
        # Attention output: attn_weights @ V
        output = torch.einsum('bhqk,bhkd->bhqd', attn_weights, v)

        # Save for backward
        ctx.save_for_backward(q, k, v, attn_weights, freqs_cos, freqs_sin)
        ctx.sm_scale = sm_scale
        ctx.causal = causal

        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Args:
            grad_output: (BATCH, H, N_CTX, HEAD_DIM) - gradient w.r.t. output
        Returns:
            dq, dk, dv: (BATCH, H, N_CTX, HEAD_DIM)
            dcausal: None
            dsm_scale: None
            dfreqs_cos: None
            dfreqs_sin: None
        """
        q, k, v, attn_weights, freqs_cos, freqs_sin = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        causal = ctx.causal
        
        # Prepare causal mask if needed
        causal_mask = None
        if causal:
            N_CTX = q.shape[2]
            mask = torch.triu(torch.ones(N_CTX, N_CTX, device=q.device, dtype=torch.bool), diagonal=1)
            causal_mask = mask.unsqueeze(0).unsqueeze(0)
        
        # Step 1: dV = attn_weights^T @ grad_output
        dv = torch.einsum('bhqk,bhqd->bhkd', attn_weights, grad_output)
        
        # Step 2: dS = grad_output @ V^T (gradient w.r.t. attn_weights)
        ds = torch.einsum('bhqd,bhkd->bhqk', grad_output, v)
        
        # Step 3: Softmax backward
        d_softmax_sum = torch.sum(ds * attn_weights, dim=-1, keepdim=True)
        dp = attn_weights * (ds - d_softmax_sum)
        
        # Apply mask to dp (gradient w.r.t. attn_scores)
        if causal:
            dp = dp.masked_fill(causal_mask, 0.0)
        
        # Step 4: dQ = dp @ K * sm_scale
        dq = torch.einsum('bhqk,bhkd->bhqd', dp, k) * sm_scale

        # Step 5: dK = dp^T @ Q * sm_scale
        dk = torch.einsum('bhqk,bhqd->bhkd', dp, q) * sm_scale

        # Apply inverse RoPE to gradients (use -freqs_sin for inverse rotation)
        dq = apply_rotary_emb(dq, freqs_cos, -freqs_sin)
        dk = apply_rotary_emb(dk, freqs_cos, -freqs_sin)

        return dq, dk, dv, None, None, None, None


attention_pytorch = _attention_pytorch.apply

