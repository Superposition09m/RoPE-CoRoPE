# Co-RoPE
# Prerequisites
## Methodology

## Environment:
```
PyTorch: 2.9.1+cu128
CUDA Version: 12.8
GPU: 4x NVIDIA H200
Triton: 3.5.1
NumPy: 2.3.5
einops: 0.8.1
flash-attn: 2.8.3
```
Install Env:
```bash
conda create -n corope python==3.12
conda activate corope
pip install torch triton
#use flash-attn as comparison
pip install packaging ninja psutil
pip install flash-attn --no-build-isolation
```

# Coding Plan

## Step 1: Implement Co-RoPE in PyTorch

We start from plain attention mechanism and then implement Co-RoPE in PyTorch.

## Step 2: Implement Co-RoPE in Triton

## Step 3: Benchmark the Performance

## Step 4: Incorporate Co-RoPE Attention into a Transformer Model, Test the Performance