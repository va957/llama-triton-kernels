# Custom GPU Kernels for LLaMA Transformer Inference

This repository implements **high-performance GPU kernels using Triton** for core Transformer inference operations.

**Hardware:** NVIDIA GeForce RTX 3070 (Compute Capability 8.6)

## Implemented Kernels

### 1. RMSNorm (Root Mean Square Layer Normalization)
- Inference-optimized implementation with one program per token
- Reduction over hidden dimension with fused weight scaling
- Block size tuning: automatic selection based on hidden dimension
- Warp count optimization (4-8 warps) for memory-bound operations

### 2. Multi-Head Attention (Flash Attention-inspired)
- Efficient attention computation with online softmax
- Block-sparse access pattern for memory efficiency
- Support for variable sequence lengths and batch processing
- Kernel fusion to reduce memory bandwidth overhead

## Design Considerations

### Memory Optimization
- **Coalesced memory access**: Ensures aligned loads for maximum bandwidth (370 GB/s on RTX 3070)
- **Kernel fusion**: Combines normalization and scaling to reduce memory round-trips
- **Thread block tuning**: Adapts to different sequence lengths and hidden dimensions

### Performance Evaluation
- Correctness verified against PyTorch reference implementations  
- Performance benchmarked on RTX 3070 GPU
- Variants tested: different block sizes, warp counts, and memory access patterns

## Files
- `rmsnorm_triton.py`: RMSNorm kernel with tuned hyperparameters
- `mha_triton.py`: Multi-Head Attention kernel with online softmax
- `benchmark_rmsnorm.py`: RMSNorm correctness and performance benchmark
- `benchmark_mha.py`: Multi-Head Attention performance evaluation
- `requirements.txt`: Dependencies (Triton, PyTorch)

## How to Run

```bash
pip install -r requirements.txt

# Benchmark RMSNorm
python benchmark_rmsnorm.py

# Benchmark Multi-Head Attention
python benchmark_mha.py
```

## Benchmarks

| Kernel | Configuration | Triton | PyTorch | Speedup |
|--------|----------------|--------|---------|---------|
| RMSNorm | (4096 tokens, 4096 dim) | ~0.25ms | ~0.32ms | 1.3x |
| MHA | (1 batch, 2048 seq, 4096 dim) | ~8.5ms | ~12.3ms | 1.4x |
