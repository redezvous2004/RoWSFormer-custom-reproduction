import torch
import torch.nn as nn
import torch.nn.functional as F

class WindowAttention(nn.Module):
  def __init__(self, dim, window_size=8, num_heads=4):
    super().__init__()
    assert dim % num_heads == 0, 'dim must be divisible by num_heads'
    self.dim = dim
    self.window_size = window_size
    self.num_heads = num_heads
    self.head_dim = dim // num_heads

    self.w_q = nn.Linear(dim, dim)
    self.w_k = nn.Linear(dim, dim)
    self.w_v = nn.Linear(dim, dim)
    self.out_proj = nn.Linear(dim, dim)

    Wh = window_size
    Ww = window_size
    self.rel_pos_table = nn.Parameter(
        torch.zeros((2 * Wh - 1) * (2 * Ww - 1), num_heads)
    )
    nn.init.trunc_normal_(self.rel_pos_table, std=0.02)

    coords_h = torch.arange(Wh)
    coords_w = torch.arange(Ww)
    coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # (2, Wh, Ww)
    coords_flat = torch.flatten(coords, 1)  # (2, Wh*Ww)

    # Pairwise differences: (2, Wh*Ww, Wh*Ww)
    rel_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
    rel_coords[0] += Wh - 1   # shift to [0, 2Wh-2]
    rel_coords[1] += Ww - 1   # shift to [0, 2Ww-2]
    rel_coords[0] *= 2 * Ww - 1

    rel_pos_index = rel_coords.sum(0)
    self.register_buffer("rel_pos_index", rel_pos_index)

  def _get_rel_pos_bias(self, batch_windows):

    seq_len = self.window_size * self.window_size

    rel_bias = self.rel_pos_table[self.rel_pos_index.view(-1)].view(seq_len, seq_len, self.num_heads)
    rel_bias = rel_bias.permute(2, 0, 1).contiguous().unsqueeze(0)  # (1, num_heads, W*W, W*W)
    return rel_bias

  def forward(self, x): # x: (batch * num_windows, window_size * window_size, channels)
    batch_windows, seq_len, channels = x.shape
    q = self.w_q(x)
    k = self.w_k(x)
    v = self.w_v(x)

    queries = q.view(batch_windows, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # (b, n_heads, seq_len, head_dim)
    keys = k.view(batch_windows, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
    values = v.view(batch_windows, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    attn_scores = queries @ keys.transpose(-2, -1) # (batch, n_heads, seq_len, seq_len)

    rel_bias = self._get_rel_pos_bias(batch_windows)       # (1, n_heads, seq_len, seq_len)
    attn_scores = attn_scores + rel_bias                    # broadcast to (batch, n_heads, seq_len, seq_len)

    attn_weights = F.softmax(attn_scores / (self.head_dim**0.5), dim=-1)

    attn = (attn_weights @ values).transpose(1, 2)
    final_attn = attn.contiguous().view(batch_windows, seq_len, channels)
    return self.out_proj(final_attn)