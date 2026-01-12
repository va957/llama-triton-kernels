import torch
import triton
import triton.language as tl


@triton.jit
def attention_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    O_ptr,
    scale,
    seq_len,
    head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Multi-Head Attention kernel for a single query token.
    Computes: O = softmax(Q @ K^T / sqrt(d_k)) @ V
    
    Q: (batch, num_heads, seq_len, head_dim)
    K: (batch, num_heads, seq_len, head_dim)
    V: (batch, num_heads, seq_len, head_dim)
    O: (batch, num_heads, seq_len, head_dim)
    """
    batch_id = tl.program_id(0)
    head_id = tl.program_id(1)
    query_pos = tl.program_id(2)
    
    # Load query for this position
    offs_d = tl.arange(0, BLOCK_SIZE)
    q_ptr = Q_ptr + batch_id * seq_len * tl.num_programs(1) * head_dim + head_id * head_dim + query_pos * tl.num_programs(1) * head_dim
    q = tl.load(q_ptr + offs_d, mask=offs_d < head_dim, other=0.0)
    
    # Initialize output and normalization
    out = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    m_i = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Iterate over key/value sequence
    for kv_block_start in tl.range(0, seq_len, BLOCK_SIZE):
        kv_block_end = tl.minimum(kv_block_start + BLOCK_SIZE, seq_len)
        
        # Load K block
        k_offsets = tl.arange(0, BLOCK_SIZE)
        k_ptr = K_ptr + batch_id * seq_len * tl.num_programs(1) * head_dim + head_id * head_dim + kv_block_start * tl.num_programs(1) * head_dim
        k_block = tl.load(k_ptr + k_offsets[:, None] * tl.num_programs(1) * head_dim + offs_d[None, :], 
                         mask=(k_offsets[:, None] < (kv_block_end - kv_block_start)) & (offs_d[None, :] < head_dim),
                         other=0.0)
        
        # Compute QK^T
        scores = tl.dot(q, tl.trans(k_block)) * scale
        scores = tl.where(k_offsets < (kv_block_end - kv_block_start), scores, float('-inf'))
        
        # Load V block
        v_ptr = V_ptr + batch_id * seq_len * tl.num_programs(1) * head_dim + head_id * head_dim + kv_block_start * tl.num_programs(1) * head_dim
        v_block = tl.load(v_ptr + k_offsets[:, None] * tl.num_programs(1) * head_dim + offs_d[None, :],
                         mask=(k_offsets[:, None] < (kv_block_end - kv_block_start)) & (offs_d[None, :] < head_dim),
                         other=0.0)
        
        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(scores, axis=0))
        p = tl.exp(scores - m_new[None, :])
        l_new = tl.exp(m_i - m_new)[None, :] * l_i[None, :] + tl.sum(p, axis=0)
        
        # Update output
        out = (tl.exp(m_i - m_new)[None, :] * out[:, None] + tl.dot(tl.trans(p), v_block)).squeeze()
        
        m_i = m_new
        l_i = l_new
    
    # Normalize output
    out = out / l_i
    
    # Store result
    o_ptr = O_ptr + batch_id * seq_len * tl.num_programs(1) * head_dim + head_id * head_dim + query_pos * tl.num_programs(1) * head_dim
    tl.store(o_ptr + offs_d, out, mask=offs_d < head_dim)


def multi_head_attention(
    query: torch.Tensor,
    key: torch.Tensor, 
    value: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """
    Multi-Head Attention with Triton kernel.
    
    query: (batch_size, seq_len, num_heads * head_dim)
    key: (batch_size, seq_len, num_heads * head_dim)
    value: (batch_size, seq_len, num_heads * head_dim)
    
    Returns:
        output: (batch_size, seq_len, num_heads * head_dim)
    """
    assert query.is_cuda and key.is_cuda and value.is_cuda
    
    batch_size, seq_len, embed_dim = query.shape
    num_heads = embed_dim // head_dim
    
    # Reshape for multi-head
    Q = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    K = key.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    V = value.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    
    scale = (head_dim ** -0.5)
    output = torch.empty_like(Q)
    
    BLOCK_SIZE = triton.next_power_of_2(head_dim)
    
    attention_kernel[(batch_size, num_heads, seq_len)](
        Q,
        K,
        V,
        output,
        scale,
        seq_len,
        head_dim,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=8,
    )
    
    # Reshape back
    output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
    return output
