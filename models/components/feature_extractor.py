import torch
import torch.nn as nn

class LocalFeatureExtractor(nn.Module):
  def __init__(self, in_channels=3, out_dim=64):
    super().__init__()
    self.in_channels = in_channels
    self.out_dims = out_dim
    # Simple convolution to extract local features
    self.conv = nn.Conv2d(in_channels, out_dim, kernel_size=3, padding=1, stride=1)
  def forward(self, x):
    # x's shape: (batch, 3, height, width)
    extracted_features = self.conv(x) # (batch, out_dims, height, width)
    return extracted_features