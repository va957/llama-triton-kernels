import torch
import triton
import triton.language as tl


@triton.jit
def rmsnorm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    stride_x,
    stride_y,
    N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    offs = tl.arange(0, BLOCK_SIZE)
    x_ptrs = x_ptr + pid * stride_x + offs
    mask = offs < N

    x = tl.load(x_ptrs, mask=mask, other=0.0)
    mean_sq = tl.sum(x * x, axis=0) / N
    rstd = tl.rsqrt(mean_sq + eps)

    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    y = x * rstd * w

    tl.store(y_ptr + pid * stride_y + offs, y, mask=mask)


def rmsnorm_triton(x: torch.Tensor, weight: torch.Tensor, eps=1e-6):
    """
    x: (num_tokens, hidden_dim)
    weight: (hidden_dim,)
    """
    assert x.is_cuda and weight.is_cuda
    assert x.shape[1] == weight.shape[0]

    y = torch.empty_like(x)
    num_tokens, hidden_dim = x.shape

    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)

    rmsnorm_kernel[(num_tokens,)](
        x,
        weight,
        y,
        x.stride(0),
        y.stride(0),
        hidden_dim,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4 if BLOCK_SIZE <= 1024 else 8,
    )
    return y
