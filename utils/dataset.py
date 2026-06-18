import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class CustomDataset(Dataset):
  def __init__(self, img_dir, img_size=128, bit_length=64):
    self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    self.bit_length = bit_length
    self.img_transformer = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
  def __len__(self):
    return len(self.img_paths)
  def __getitem__(self, idx):
    image = Image.open(self.img_paths[idx]).convert('RGB')
    image = self.img_transformer(image)

    watermark = torch.randint(0, 2, (self.bit_length,)).float()
    return image, watermark