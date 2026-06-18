import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
  def __init__(self, dim, num_heads=4):
    super().__init__()
    assert dim % num_heads == 0, 'dim must be divisible by num_heads'
    self.dim = dim
    self.num_heads = num_heads
    self.head_dim = dim // num_heads

    self.norm1 = nn.LayerNorm(dim)
    self.norm2 = nn.LayerNorm(dim)

    self.w_q = nn.Linear(dim, dim)
    self.w_k = nn.Linear(dim, dim)
    self.w_v = nn.Linear(dim, dim)
    self.out_proj = nn.Linear(dim, dim)
    self.ffn = nn.Sequential(
        nn.Linear(dim, dim * 4),
        nn.GELU(),
        nn.Linear(dim * 4, dim)
    )
  def forward(self, x):
    batch, channels, height, width = x.shape
    residual = x
    x = x.flatten(2).transpose(1, 2)
    seq_len = x.shape[1]

    x = self.norm1(x)
    q = self.w_q(x)
    k = self.w_k(x)
    v = self.w_v(x)

    queries = q.view(batch, seq_len, self.num_heads, channels // self.num_heads).transpose(1, 2) # (batch, num_heads, seq_len, head_dim)
    keys = k.view(batch, seq_len, self.num_heads, channels // self.num_heads).transpose(1, 2)
    values = v.view(batch, seq_len, self.num_heads, channels // self.num_heads).transpose(1, 2)

    attn_scores = queries @ keys.transpose(-2, -1) # (batch, num_heads, seq_len, seq_len)
    attn_scores = F.softmax(attn_scores / (self.head_dim**0.5), dim=-1)

    attn_weights = (attn_scores @ values).transpose(1, 2).reshape(batch, seq_len, channels)
    x = self.out_proj(attn_weights).transpose(1, 2)
    x = x.view(batch, channels, height, width)
    x = x + residual

    residual = x
    x = x.flatten(2).transpose(1, 2)
    x = self.norm2(x)
    x = self.ffn(x)
    x = x.transpose(1, 2).view(batch, channels, height, width)
    x = x + residual

    return x