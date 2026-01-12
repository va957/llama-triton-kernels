import torch
import time
from rmsnorm_triton import rmsnorm_triton


def rmsnorm_torch(x, w, eps=1e-6):
    var = x.pow(2).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(var + eps) * w


def benchmark():
    device = "cuda"
    torch.manual_seed(0)

    num_tokens = 4096
    hidden_dim = 4096

    x = torch.randn(num_tokens, hidden_dim, device=device)
    w = torch.randn(hidden_dim, device=device)

    # Warmup
    for _ in range(10):
        rmsnorm_triton(x, w)
        rmsnorm_torch(x, w)

    # Correctness
    y_triton = rmsnorm_triton(x, w)
    y_torch = rmsnorm_torch(x, w)

    max_err = (y_triton - y_torch).abs().max().item()
    print(f"Max error: {max_err:.6e}")

    # Timing
    def time_fn(fn, iters=100):
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.time() - start) * 1000 / iters

    t_triton = time_fn(lambda: rmsnorm_triton(x, w))
    t_torch = time_fn(lambda: rmsnorm_torch(x, w))

    print(f"Triton RMSNorm: {t_triton:.3f} ms")
    print(f"PyTorch RMSNorm: {t_torch:.3f} ms")
    print(f"Speedup: {t_torch / t_triton:.2f}x")


if __name__ == "__main__":
    benchmark()
