import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.transformer import Transformer
from torch_dct import dct_2d

class FrequencyEnhancedBlock(nn.Module):
  """
  per-patch DCT → frequency magnitude → average → FC → channel attention weights.
  """
  def __init__(self, dim, patch_size=4):
    super().__init__()
    self.dim = dim
    self.patch_size = patch_size

    self.fc = nn.Sequential(
        nn.Linear(dim, dim // 4),
        nn.ReLU(inplace=True),
        nn.Linear(dim // 4, dim),
        nn.Sigmoid()
    )

  def forward(self, x):
    batch, channels, height, width = x.shape
    ph, pw = self.patch_size, self.patch_size

    pad_h = (ph - height % ph) % ph
    pad_w = (pw - width % pw) % pw
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

    _, _, H, W = x.shape
    num_patches_h = H // ph
    num_patches_w = W // pw
    num_patches = num_patches_h * num_patches_w

    # Reshape (batch, channels, num_patches_h, ph, num_patches_w, pw)
    x_p = x.view(batch, channels, num_patches_h, ph, num_patches_w, pw)
    # Permute (batch, channels, num_patches_h, num_patches_w, ph, pw)
    x_p = x_p.permute(0, 1, 2, 4, 3, 5).contiguous()
    # (batch, channels, num_patches, ph*pw)
    x_p = x_p.view(batch, channels, num_patches, ph * pw)
    x_p = x_p.permute(0, 2, 1, 3).contiguous()
    # Reshape (batch*num_patches*channels, 1, 1, ph*pw)
    x_p = x_p.view(batch * num_patches * channels, 1, 1, ph * pw)

    freq = dct_2d(x_p, norm='ortho')  # (B*P*C, 1, 1, ph*pw)
    freq = freq.squeeze(1).squeeze(1)                   # (B*P*C, ph*pw)
    freq_mag = torch.abs(freq)                           # (B*P*C, ph*pw)

    freq_mag = freq_mag.view(batch, num_patches, channels, ph * pw)
    freq_agg = freq_mag.mean(dim=1)                      # (batch, channels, ph*pw)
    freq_agg = freq_agg.mean(dim=-1)                     # (batch, channels)

    freq_weights = self.fc(freq_agg).view(batch, channels, 1, 1)

    x = x[:, :, :height, :width]
    x = x * freq_weights

    return x

class FETB(nn.Module):
  def __init__(self, dim, num_heads=4, num_blocks=2):
    super().__init__()

    self.blocks = nn.ModuleList([
        Transformer(dim=dim, num_heads=num_heads)
        for _ in range(num_blocks)
    ])
    self.feb = FrequencyEnhancedBlock(dim)

  def forward(self, x):
    for block in self.blocks:
      x = block(x)
    x = self.feb(x)
    return x