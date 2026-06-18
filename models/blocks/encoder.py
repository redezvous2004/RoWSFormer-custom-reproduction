import torch
import torch.nn as nn
from models.components.feature_extractor import LocalFeatureExtractor
from models.components.FETB import FETB
from models.components.LCESTB import LCESTB
from models.components.watermark_encoder import WatermarkEncoder

class Encoder(nn.Module):
  # UNet-like architecture
  def __init__(self, in_channels=3, dim=64, num_stages=3, num_heads=4, window_size=8, watermark_length=64):
    super().__init__()
    self.num_stages = num_stages
    self.dim = dim

    self.extractor = LocalFeatureExtractor(in_channels=in_channels, out_dim=dim)
    self.watermark_encoder = WatermarkEncoder(
        watermark_length=watermark_length,
        out_dim=dim,
        l1=512
    )

    self.down_blocks = nn.ModuleList()
    self.down_samples = nn.ModuleList()
    for i in range(num_stages):
      down_dim = dim * (2 ** i)
      self.down_blocks.append(
          nn.Sequential(
              LCESTB(dim=down_dim, num_heads=num_heads, window_size=window_size, shift_size=0),
              LCESTB(dim=down_dim, num_heads=num_heads, window_size=window_size, shift_size=4)
          )
      )
      if i < num_stages - 1:
        self.down_samples.append(
            nn.Conv2d(in_channels=down_dim, out_channels=down_dim*2, kernel_size=4, stride=2, padding=1)
        )

    bottleneck_dim = dim * (2 ** (num_stages - 1))
    self.bottleneck = FETB(bottleneck_dim, num_heads=num_heads, num_blocks=2)

    self.up_blocks = nn.ModuleList()
    self.up_samples = nn.ModuleList()
    for i in range(num_stages - 1, -1, -1):
      up_dim = dim * (2 ** i)
      if i < num_stages - 1:
        self.up_samples.append(
            nn.ConvTranspose2d(in_channels= up_dim * 2, out_channels=up_dim, kernel_size=2, stride=2)
        )
      concat_dim = up_dim * 2 + dim  # FIX: watermark outputs dim (64) not dim//2 (32)

      self.up_blocks.append(
          nn.Sequential(
              nn.Conv2d(concat_dim, up_dim, kernel_size=3, padding=1),
              LCESTB(up_dim, num_heads=num_heads, window_size=window_size, shift_size=0),
              LCESTB(up_dim, num_heads=num_heads, window_size=window_size, shift_size=4)
          )
      )
    self.output = nn.Conv2d(dim, in_channels, kernel_size=3, padding=1)
  def forward(self, image, watermark):
    '''
      Args:
          image: (batch, 3, height, width)
          watermark: (batch, watermark_length)
    '''
    batch, channels, height, width = image.shape
    # Initial features
    x = self.extractor(image) # (batch, 64, height, width)

    residual_conns = []
    cur_h, cur_w = height, width
    for block, sample in zip(self.down_blocks, list(self.down_samples) + [None]):
      x = block[0](x)
      x = block[1](x)
      residual_conns.append(x)

      if sample is not None:
        x = sample(x)
        cur_h, cur_w = cur_h // 2, cur_w // 2
    x = self.bottleneck(x)

    residual_conns = residual_conns[::-1]

    for i, (sample, block) in enumerate(zip([None] + list(self.up_samples), self.up_blocks)):
      if sample is not None:
        x = sample(x)
        cur_h, cur_w = cur_h * 2, cur_w * 2
      watermark_features = self.watermark_encoder(
          watermark,
          target_size=(cur_h, cur_w)
      )
      x = torch.cat([x, residual_conns[i], watermark_features], dim=1)

      x = block[0](x)
      x = block[1](x)
      x = block[2](x)

    delta = self.output(x)
    watermarked_image = image + delta
    if not self.training:
      watermarked_image = torch.clamp(watermarked_image, 0.0, 1.0)
    return watermarked_image