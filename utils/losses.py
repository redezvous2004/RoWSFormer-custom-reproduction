import torch
import torch.nn as nn

class WatermarkLoss(nn.Module):
  def __init__(self, lambda1=1.0, lambda2=1.0, lambda3=0.5):
    super().__init__()
    self.lambda1 = lambda1   # image quality
    self.lambda2 = lambda2   # watermark extraction
    self.lambda3 = lambda3   # pixel range constraint

    self.mse = nn.MSELoss()
    self.bce = nn.BCEWithLogitsLoss()

  def image_loss(self, origin_img, watermarked_img):
    return self.mse(origin_img, watermarked_img)
  def watermark_loss(self, origin_wm, extracted_wm):
    return self.bce(extracted_wm, origin_wm)
  def constraint_loss(self, watermarked_img):
    # Soft constraint
    return self.mse(watermarked_img, torch.clamp(watermarked_img, 0.0, 1.0))
  def forward(self, origin_img, watermarked_img, origin_wm, extracted_wm):
    lE = self.image_loss(origin_img, watermarked_img)
    lD = self.watermark_loss(origin_wm, extracted_wm)
    lC = self.constraint_loss(watermarked_img)

    total_loss = self.lambda1 * lE + self.lambda2 * lD + self.lambda3 * lC

    loss_dict = {
        'total': total_loss.item(),
        'image_loss': lE.item(),
        'watermark_loss': lD.item(),
        'constraint_loss': lC.item(),
    }
    return total_loss, loss_dict