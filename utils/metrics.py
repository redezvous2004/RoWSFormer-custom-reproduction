import torch

def PSNR(img1, img2):
  mse = torch.mean((img1 - img2) ** 2)
  if mse == 0:
    return float('inf')

  psnr = 10 * torch.log10(1 ** 2 / mse).item()
  return psnr

def bit_accuracy(origin_wm, extracted_wm, thresh_hold=0.5):
  extracted_probs = torch.sigmoid(extracted_wm)
  extracted_bins = (extracted_probs > thresh_hold).float()

  correct = (extracted_bins == origin_wm).float()
  accuracy = correct.mean().item() * 100

  return accuracy