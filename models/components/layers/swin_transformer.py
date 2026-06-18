import torch
import torch.nn as nn
from window_attention import WindowAttention

def window_partition(x, window_size):
  # Non overlapping window
  batch, num_features, height, width = x.shape
  x = x.view(batch, num_features, height // window_size, window_size, width // window_size, window_size)
  windows = x.permute(0, 2, 4, 1, 3, 5).contiguous()

  windows = windows.view(-1, num_features, window_size, window_size) # (batch * num_windows, num_features, window_size, window_size)
  return windows

def window_reverse(windows, window_size, height, width):
  batch_window, num_features, _, _ = windows.shape
  batch = batch_window // ((height // window_size) * (width // window_size))
  x = windows.view(batch, height // window_size, width // window_size, num_features, window_size, window_size)
  x = x.permute(0, 3, 1, 4, 2, 5)
  x = x.contiguous().view(batch, num_features, height, width)
  return x

class SwinTransformer(nn.Module):
  def __init__(self, model_dim, num_heads=4, window_size=8, shift_size=0):
    super().__init__()
    self.model_dim = model_dim
    self.num_heads = num_heads
    self.window_size = window_size
    self.shift_size = shift_size

    self.norm1 = nn.LayerNorm(model_dim)
    self.norm2 = nn.LayerNorm(model_dim)

    self.wmha = WindowAttention(model_dim, window_size, num_heads)
    self.ffn = nn.Sequential(
        nn.Linear(model_dim, model_dim * 4),
        nn.GELU(),
        nn.Linear(model_dim * 4, model_dim)
    )
  def forward(self, x):
    batch, channels, height, width = x.shape
    residual = x
    # Reshape to (batch, height * width, channels)
    x = x.flatten(2).transpose(1, 2)
    if self.shift_size > 0:
      x = x.view(batch, height, width, channels)
      x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
      x = x.view(batch, height * width, channels)
    x_windows = window_partition(
        x.view(batch, height, width, channels).permute(0, 3, 1, 2).contiguous(),
        self.window_size
    ) # (batch * num_windows, channels, height, width)
    x_windows = x_windows.flatten(2).transpose(1, 2)
    # W-MSA
    x_windows = self.norm1(x_windows)
    attn_windows = self.wmha(x_windows)

    attn_windows = attn_windows.transpose(1, 2).view(-1, channels, self.window_size, self.window_size)
    x = window_reverse(attn_windows, self.window_size, height, width) # (batch, channels, height, width)

    if self.shift_size > 0:
      x = x.permute(0, 2, 3, 1).contiguous()
      x = torch.roll(x, shifts=(self.window_size, self.window_size), dims=(1, 2))
      x = x.permute(0, 3, 1, 2).contiguous() # (batch, channels, height, width)

    x = x + residual

    residual = x
    x = x.flatten(2).transpose(1, 2)
    x = self.norm2(x)
    x = self.ffn(x)
    x = x.transpose(1, 2).view(batch, channels, height, width)
    return x + residual