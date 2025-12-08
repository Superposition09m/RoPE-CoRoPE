import torch
import torch.nn.functional as F
import math

from utils import print_red_warning, calc_sim, assert_similar

class CoRoPE(torch.nn.Module):
    """
    Co-RoPE Attention Block:

    x: (B, T, n_embd), B is batch size, T is sequence length, n_embd is embedding dimension
    q: (B, T, n_head, head_dim), then transpose to (B, n_head, T, head_dim) for attention computation
    k: (B, T, n_kv_head, head_dim), then transpose to (B, n_kv_head, T, head_dim) for attention computation
    v: (B, T, n_kv_head, head_dim), then transpose to (B, n_kv_head, T, head_dim) for attention computation
    
    """
    def __init__(self, n_embd: int, n_head: int, n_kv_head: int = None):
        super().__init__()
        self.n_head = n_head
        self.n_kv_head = n_kv_head if n_kv_head is not None else n_head  # Default MHA, also supports GQA
        self.n_embd = n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        
        # Q projection: n_head heads
        self.q_proj = torch.nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        
        # K/V projection: n_kv_head heads (possibly fewer)
        self.k_proj = torch.nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.v_proj = torch.nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        
        # Output projection: input is n_head * head_dim (because GQA will broadcast)
        self.o_proj = torch.nn.Linear(self.n_head * self.head_dim, self.n_embd, bias=False)
        self.scale_factor = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x: torch.Tensor, causal: bool = True):
        B, T, _ = x.shape

        # Q projection: (B, T, n_embd) -> (B, T, n_head * head_dim)
        q = self.q_proj(x)
        q = q.view(B, T, self.n_head, self.head_dim)
        q = q.transpose(1, 2)  # (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        
        # K/V projection: (B, T, n_embd) -> (B, T, n_kv_head * head_dim)
        k = self.k_proj(x)
        k = k.view(B, T, self.n_kv_head, self.head_dim)
        k = k.transpose(1, 2)  # (B, T, n_kv_head, head_dim) -> (B, n_kv_head, T, head_dim)
        
        v = self.v_proj(x)
        v = v.view(B, T, self.n_kv_head, self.head_dim)
        v = v.transpose(1, 2)  # (B, T, n_kv_head, head_dim) -> (B, n_kv_head, T, head_dim)
        
        # GQA broadcast k and v to match q
        if self.n_kv_head < self.n_head:
            num_groups = self.n_head // self.n_kv_head
            k = k.repeat_interleave(num_groups, dim=1)  # (B, n_head, T, head_dim)
            v = v.repeat_interleave(num_groups, dim=1)  # (B, n_head, T, head_dim)

        # Compute attention scores
        # q: (B, n_head, T, head_dim)
        # k: (B, n_head, T, head_dim)
        # attn_scores: (B, n_head, T, T), the last two dim [q, k]
        attn_scores = torch.einsum('bhqd,bhkd->bhqk', q, k)
        attn_scores = attn_scores * self.scale_factor
        if causal:
            mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

        # attn_scores: (B, n_head, T, T) -> attn_weights: (B, n_head, T, T)
        attn_weights = F.softmax(attn_scores, dim=-1)# probabilities sum to 1 for all kv pairs
        # v: (B, n_head, T, head_dim)
        # attn_weights: (B, n_head, T, T)
        # y: (B, n_head, T, head_dim)
        y = torch.einsum('bhqk,bhkd->bhqd', attn_weights, v)

        # Reshape back: (B, n_head, T, head_dim) -> (B, T, n_head * head_dim)
        y = y.transpose(1, 2).contiguous()
        y = y.view(B, T, -1)
        y = self.o_proj(y)
        return y



def simple_test1():
    """Test the attention block forward pass
    """
    print("=" * 60)
    print("Testing CoRoPE Attention Block Forward Pass")
    print("=" * 60)
    
    # Test parameters
    B, T, n_embd = 2, 8, 64
    n_head = 4
    
    # Test 1: Standard MHA (n_kv_head = n_head)
    print("\n--- Test 1: Standard Multi-Head Attention (MHA) ---")
    model_mha = CoRoPE(n_embd=n_embd, n_head=n_head, n_kv_head=None)
    x = torch.randn(B, T, n_embd)
    
    output = model_mha(x, causal=True)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == (B, T, n_embd), f"Expected output shape {(B, T, n_embd)}, got {output.shape}"
    print("✓ MHA forward pass: PASSED")
    
    # Test 2: GQA (n_kv_head < n_head)
    print("\n--- Test 2: Group-Query Attention (GQA) ---")
    n_kv_head = 2
    model_gqa = CoRoPE(n_embd=n_embd, n_head=n_head, n_kv_head=n_kv_head)
    output_gqa = model_gqa(x, causal=True)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_gqa.shape}")
    assert output_gqa.shape == (B, T, n_embd), f"Expected output shape {(B, T, n_embd)}, got {output_gqa.shape}"
    print("✓ GQA forward pass: PASSED")
    
    # Test 3: Non-causal attention
    print("\n--- Test 3: Non-causal Attention ---")
    output_nc = model_mha(x, causal=False)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output_nc.shape}")
    assert output_nc.shape == (B, T, n_embd), f"Expected output shape {(B, T, n_embd)}, got {output_nc.shape}"
    print("✓ Non-causal forward pass: PASSED")
    
    # Test 4: Check output is not NaN or Inf
    print("\n--- Test 4: Output Validity Check ---")
    assert not torch.isnan(output).any(), "Output contains NaN!"
    assert not torch.isinf(output).any(), "Output contains Inf!"
    print("✓ Output validity: PASSED")
    
    # Test 5: Different sequence lengths
    print("\n--- Test 5: Different Sequence Lengths ---")
    T2 = 16
    x2 = torch.randn(B, T2, n_embd)
    output2 = model_mha(x2, causal=True)
    print(f"Input shape: {x2.shape}")
    print(f"Output shape: {output2.shape}")
    assert output2.shape == (B, T2, n_embd), f"Expected output shape {(B, T2, n_embd)}, got {output2.shape}"
    print("✓ Variable sequence length: PASSED")
    
    print("\n" + "=" * 60)
    print("All forward pass tests PASSED! ✓")
    print("=" * 60)
    


def simple_test2():
    """
    Test the handwritten backward pass
    """
    # TODO: Implement backward pass test
    pass

if __name__ == "__main__":
    simple_test1()
