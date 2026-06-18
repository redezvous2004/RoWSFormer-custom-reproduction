from math import sqrt, ceil
import torch
import torch.nn as nn
import torch.nn.functional as F

class WatermarkEncoder(nn.Module):
  def __init__(self, watermark_length=64, out_dim=32, l1=256):
    super().__init__()
    self.watermark_length = watermark_length
    self.out_dim = out_dim
    self.l1 = l1
    self.grid_h = int(ceil(sqrt(l1)))
    self.grid_w = int(ceil(l1 / self.grid_h))
    self.grid_size = self.grid_h * self.grid_w
    self.linear = nn.Linear(watermark_length, self.grid_size)
    self.conv = nn.Conv2d(1, out_dim, kernel_size=3, padding=1)
  def forward(self, watermark, target_size):
    # watermark's shape (batch, watermark_length)
    batch = watermark.shape[0]
    x = self.linear(watermark)  # (batch, grid_size)
    x = x.view(batch, 1, self.grid_h, self.grid_w)  # (batch, 1, grid_h, grid_w)
    x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)  # (batch, 1, H, W)
    return self.conv(x)  # (batch, out_dim, H, W)