import torch
import torch.nn as nn
from models.components.feature_extractors import LocalFeatureExtractor
from models.components import LCESTB
from models.components.FETB import FETB
from models.components.LCESTB import LCESTB

class Decoder(nn.Module):
  def __init__(self, in_channels=3, dim=64, num_stages=3, num_heads=4, window_size=8, watermark_length=64):
    super().__init__()
    self.num_stages = num_stages
    self.dim = dim
    self.watermark_length = watermark_length

    self.extractor = LocalFeatureExtractor(in_channels=in_channels, out_dim=dim)
    self.down_blocks = nn.ModuleList()
    self.down_samples = nn.ModuleList()

    for i in range(num_stages):
      layer_dim = dim * (2 ** i)
      self.down_blocks.append(
          nn.Sequential(
            LCESTB(layer_dim, num_heads=num_heads, window_size=window_size, shift_size=0),
            LCESTB(layer_dim, num_heads=num_heads, window_size=window_size, shift_size=4)
          )
      )

      if i < num_stages - 1:
        self.down_samples.append(
            nn.Sequential(
              nn.Conv2d(layer_dim, layer_dim * 2, kernel_size=4, stride=2, padding=1),
              nn.BatchNorm2d(layer_dim * 2) # stabilize gradient flow
            )
        )
    bottleneck_dim = dim * (2 ** (num_stages - 1))
    self.bottleneck = FETB(bottleneck_dim, num_heads=num_heads, num_blocks=2)

    self.info_conv = nn.Sequential(
        nn.Conv2d(bottleneck_dim, bottleneck_dim, kernel_size=3, padding=1),
        nn.BatchNorm2d(bottleneck_dim),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((4, 4))
    )
    fc_indim = bottleneck_dim * 4 * 4
    self.fc = nn.Sequential(
        nn.Linear(fc_indim, fc_indim // 2),
        nn.ReLU(inplace=True),
        nn.Linear(fc_indim // 2, watermark_length)
    )

    # Initialize weights to prevent vanishing gradients
    for m in self.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

  def forward(self, x):
    batch = x.shape[0]

    x = self.extractor(x)
    for block, sample in zip(self.down_blocks, list(self.down_samples) + [None]):
      x = block[0](x)
      x = block[1](x)

      if sample is not None:
        x = sample(x)

    x = self.bottleneck(x) # (batch, bottleneck_dim, H/4, W/4)
    x = self.info_conv(x) # (batch, bottleneck_dim, 4, 4)
    x = x.reshape(batch, -1) # (batch, bottleneck_dim * 4 * 4)

    watermark = self.fc(x) # (batch, 64)

    return watermark