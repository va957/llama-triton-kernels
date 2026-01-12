# Custom Triton Kernels for LLaMA Transformer Inference

This repository implements **custom GPU kernels using Triton** for core Transformer inference operations,
inspired by production LLM inference engines such as **vLLM** and **FlashAttention**.

## Implemented Kernels
- **RMSNorm (Inference-Optimized)**
  - One program per token
  - Reduction over hidden dimension
  - Fused normalization and weight scaling
  - Tuned block size and warp count

## Motivation
Modern LLM inference performance is dominated by memory access patterns and kernel launch efficiency.
This project explores low-level kernel design trade-offs while maintaining a Python-first workflow.

## Files
- `rmsnorm_triton.py`: Triton kernel implementation
- `benchmark_rmsnorm.py`: Correctness and performance benchmark vs PyTorch
- `requirements.txt`: Dependencies

## How to Run
```bash
pip install -r requirements.txt
python benchmark_rmsnorm.py
