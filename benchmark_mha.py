import torch
import time
from mha_triton import multi_head_attention


def benchmark_attention():
    """Benchmark Multi-Head Attention kernel."""
    device = "cuda"
    torch.manual_seed(42)
    
    # Typical LLaMA-7B configuration
    batch_size = 1
    seq_len = 2048
    embed_dim = 4096
    num_heads = 32
    head_dim = embed_dim // num_heads
    
    Q = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=torch.float32)
    K = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=torch.float32)
    V = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=torch.float32)
    
    # PyTorch reference
    def pytorch_attention(Q, K, V):
        Q_heads = Q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        K_heads = K.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        V_heads = V.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        scores = torch.matmul(Q_heads, K_heads.transpose(-2, -1)) / (head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V_heads)
        return output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
    
    # Warmup
    for _ in range(3):
        _ = multi_head_attention(Q, K, V, head_dim)
        _ = pytorch_attention(Q, K, V)
    
    # Timing
    def time_fn(fn, iters=10):
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.time() - start) * 1000 / iters
    
    t_triton = time_fn(lambda: multi_head_attention(Q, K, V, head_dim))
    t_pytorch = time_fn(lambda: pytorch_attention(Q, K, V))
    
    print(f"\n=== Multi-Head Attention Benchmark ===")
    print(f"Batch: {batch_size}, Seq Len: {seq_len}, Embed Dim: {embed_dim}")
    print(f"Triton MHA:  {t_triton:.3f} ms")
    print(f"PyTorch MHA: {t_pytorch:.3f} ms")
    if t_triton > 0:
        print(f"Speedup: {t_pytorch / t_triton:.2f}x")


if __name__ == "__main__":
    benchmark_attention()
