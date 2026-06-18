import torch
import torch.nn as nn
from layers.swin_transfomer import SwinTransformer

class LocallyChannelEnhancedBlock(nn.Module):
  def __init__(self, dim, expansion=2):
    super().__init__()
    hidden_dim = dim * expansion

    self.expand_layer = nn.Conv2d(dim, hidden_dim, kernel_size=1)
    self.deepwise_layer = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
    self.compress_layer = nn.Conv2d(hidden_dim, dim, kernel_size=1)
    self.pooling_layer = nn.AdaptiveAvgPool2d(1)
    self.mlp = nn.Sequential(
        nn.Linear(dim, dim // 4),
        nn.ReLU(inplace=True),
        nn.Linear(dim // 4, dim)
    )
    self.nl_learning_layer = nn.GELU()

  def forward(self, x):
    x_img = x

    x = self.nl_learning_layer(self.expand_layer(x))
    x = self.nl_learning_layer(self.deepwise_layer(x))
    x = self.compress_layer(x)

    batch, channels, height, width = x.shape
    channel_attn_weights = self.pooling_layer(x).view(batch, channels)
    channel_attn_weights = self.mlp(channel_attn_weights).view(batch, channels, 1, 1)

    x_bias = x * channel_attn_weights

    return x_bias + x_img

class LCESTB(nn.Module):
  def __init__(self, dim, num_heads=4, window_size=8, shift_size=0):
    super().__init__()
    self.swintrans_block = SwinTransformer(
        model_dim= dim,
        num_heads= num_heads,
        window_size= window_size,
        shift_size= shift_size
    )
    self.lceb = LocallyChannelEnhancedBlock(dim=dim)

  def forward(self, x):
    x = self.lceb(self.swintrans_block(x))
    return x