import torch
import torch.nn as nn
from flash_attn.flash_attn_interface import flash_attn_unpadded_func

class Rotary(nn.Module):
    # Assuming you have a Rotary embedding implementation
    def __init__(self, head_dim):
        super().__init__()
        # Initialize rotary embeddings here

    def forward(self, x):
        # Compute rotary embeddings (cos and sin)
        cos = torch.cos(x)  # Placeholder
        sin = torch.sin(x)  # Placeholder
        return cos, sin

def apply_rotary_emb(x, cos, sin):
    # Apply rotary embeddings to tensor x
    return (x * cos) + (torch.cat([-x[..., 1::2], x[..., ::2]], dim=-1) * sin)

class FlashCausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0, "Embedding dimension must be divisible by number of heads."

        # Combined projection for query, key, value
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x, attention_mask=None):
        """
        Args:
            x: Input tensor of shape (B, T, C)
            attention_mask: Optional mask tensor of shape (B, T)
        Returns:
            Tensor of shape (B, T, C)
        """
        B, T, C = x.size()  # Batch size, Sequence length, Embedding dimension

        # Project inputs to query, key, value
        qkv = self.c_attn(x)  # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)  # Each is (B, T, C)

        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, n_head, T, head_dim)

        # Apply rotary embeddings
        cos, sin = self.rotary(q)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Flash Attention expects tensors of shape (B, n_head, T, head_dim)
        # FlashAttention handles causal masking internally
        y = flash_attn_unpadded_func(q, k, v, dropout_p=0.0, softmax_scale=1.0, causal=True)

        # y has shape (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        # Output projection
        y = self.c_proj(y)  # (B, T, C)
        return y
