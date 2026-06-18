import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import random
import numpy as np
import kornia

class NoiseLayer(nn.Module):
  def __init__(self, attack_types=None):
    super().__init__()
    if attack_types is None:
      self.attack_types = [
          'none',
          # Non-geometric
          'gaussian_noise',
          'salt_and_pepper_noise',
          'gaussian_blur',
          'median_blur',
          # Geometric
          'cropout',
          'dropout',
          'rotation',
          'scaling',
      ]
    else:
      self.attack_types = attack_types
  def set_attack_types(self, attack_types):
    self.attack_types = attack_types
  def gaussian_noise(self, x, sigma=None):
    '''
      Training phase: 0.001 -> 0.04
      Test phase: 0.01 -> 0.05
    '''
    if sigma is None:
      sigma = torch.rand(1).item() * 0.039 + 0.001

    noise = torch.randn_like(x) * sigma

    attacked_img = x + noise
    return torch.clamp(attacked_img, 0, 1)

  def salt_and_pepper_noise(self, x, ratio=None):
    '''
      Training phase: 0.001 -> 0.04
      Test phase: 0.01 -> 0.05
    '''
    if ratio is None:
      ratio = torch.rand(1).item() * 0.039 + 0.001

    batch, channels, height, width = x.shape
    attacked_img = x.clone()

    probs = torch.rand((batch, 1, height, width), device=x.device)
    attacked_img = torch.where(probs < (ratio / 2.0), torch.tensor(1.0, device=x.device), attacked_img)
    attacked_img = torch.where(probs > (1.0 - ratio / 2.0), torch.tensor(0.0, device=x.device), attacked_img)
    return attacked_img

  def gaussian_blur(self, x, kernel_size=5, sigma=2):
    '''
        Fixed 2 in training phase
        0.0001 -> 2 in testing phase
    '''
    attacked_img = TF.gaussian_blur(x, [kernel_size, kernel_size], [sigma, sigma])
    return attacked_img

  def median_blur(self, x, kernel_size=7):
    '''
       Fixed window: 7x7 in training phase
       3x3, 5x5, 7x7 window in testing phase
    '''
    pad = kernel_size // 2
    x_padded = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    batch, channels, height, width = x.shape
    patches = F.unfold(x_padded, kernel_size=kernel_size) # (batch, channels * Kh * Kw, L)

    patches = patches.view(batch, channels, kernel_size * kernel_size, height, width)

    attacked_img, _ = torch.median(patches, dim=2)

    return attacked_img

  def cropout(self, x, original_img, ratio=0.4):
    '''
      Ratio 0.4 during training phase
      Ratio [0.1, 0.5] during testing phase
    FIX: was replacing cropped region with original_img — leaks original image!
      Now fills with mean fill value (neutral) to avoid leaking cover image.
    '''
    batch, channels, height, width = x.shape
    crop_h = int(height * ratio ** 0.5)
    crop_w = int(width * ratio ** 0.5)

    random_top = torch.randint(0, height - crop_h + 1, (1,)).item()
    random_left = torch.randint(0, width - crop_w + 1, (1,)).item()

    fill_value = x.mean(dim=(2, 3), keepdim=True)  # (batch, channels, 1, 1)
    fill = fill_value.expand(-1, -1, crop_h, crop_w)  # (batch, channels, crop_h, crop_w)

    x = x.clone()
    x[:, :, random_top:random_top + crop_h, random_left:random_left + crop_w] = fill
    return x

  def dropout(self, x, original_img, ratio=0.4):
    '''
      Ratio 0.4 during training phase
      Ratio [0.2, 0.6] during testing phase
    FIX: was leaking original image into dropped pixels — bad for blind extraction.
      Now replaced with mean fill to avoid leaking cover signal.
    '''
    mask = (torch.rand_like(x) > ratio).float()
    fill_value = x.mean(dim=(2, 3), keepdim=True)
    fill = fill_value.expand_as(x)
    return x * mask + fill * (1 - mask)

  def rotation(self, x, angle=None):
    '''
      Angle range: [-30, 30] for both phase
    '''
    if angle is None:
      angle = random.uniform(-30.0, 30.0)

    angle_tensor = torch.tensor([angle] * x.shape[0], device=x.device)

    attacked_img = kornia.geometry.transform.rotate(x, angle_tensor)
    return attacked_img

  def scaling(self, x, ratio=None):
    '''
      Ratio [0.7, 1.5] during training phase
      Ratio [0.6, 2] during testing phase
    '''
    if ratio is None:
      ratio = torch.rand(1) * 0.8 + 0.7

    attacked_img = x.clone()
    batch, channels, height, width = x.shape

    new_h = int(height * ratio)
    new_w = int(width * ratio)


    scaled_img = F.interpolate(x, size=(new_h, new_w), mode='bilinear', align_corners=False)

    attacked_img = F.interpolate(scaled_img, size=(height, width), mode='bilinear', align_corners=False)

    return attacked_img

  def forward(self, x, origin_imgs=None, attack_type=None, rot_angle=None, scale=None, shear_angle=None):
    if attack_type is None:
      attack_type = np.random.choice(self.attack_types)
    if attack_type == 'none':
      return x
    elif attack_type == 'gaussian_noise':
      return self.gaussian_noise(x)
    elif attack_type == 'salt_and_pepper_noise':
      return self.salt_and_pepper_noise(x)
    elif attack_type == 'gaussian_blur':
      return self.gaussian_blur(x)
    elif attack_type == 'median_blur':
      return self.median_blur(x)
    elif attack_type == 'cropout':
      assert origin_imgs is not None; 'Must pass origin image'
      return self.cropout(x, origin_imgs)
    elif attack_type == 'dropout':
      assert origin_imgs is not None; 'Must pass origin image'
      return self.dropout(x, origin_imgs)
    elif attack_type == 'rotation':
      return self.rotation(x, rot_angle)
    elif attack_type == 'scaling':
      return self.scaling(x, scale)
    else:
      print("There is no type of that attack")
      return x