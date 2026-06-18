import torch
import torch.nn as nn
from utils.metrics import bit_accuracy, PSNR
from tqdm import tqdm

def train_epoch(encoder, decoder, noise_layer, dataloader,
                criterion, optimizer, device, epoch, attack_type=None):
  encoder.train()
  decoder.train()

  total_losses = {
      'total': 0,
      'image_loss': 0,
      'watermark_loss': 0,
      'constraint_loss': 0,
  }
  num_batches = 0
  total_acc = 0
  total_psnr = 0

  pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

  for idx, (images, watermarks) in enumerate(pbar):
    images = images.to(device)
    watermarks = watermarks.to(device)

    optimizer.zero_grad()


    watermarked_imgs = encoder(images, watermarks)
    attacked_imgs = noise_layer(watermarked_imgs, origin_imgs=images, attack_type=attack_type)
    extracted_wms = decoder(attacked_imgs)

    loss, loss_dict = criterion(
        images, watermarked_imgs, watermarks, extracted_wms
    )
    loss.backward()

    if idx % 50 == 0:
      def grad_norm(params_iter):
        grads = [p.grad for p in params_iter if p.grad is not None]
        if not grads:
          return 0.0
        return torch.sqrt(sum(g.norm()**2 for g in grads)).item()

      print(f"\n  [Grad Norms @ batch {idx}]"
            f"  dec_fc={grad_norm(decoder.fc.parameters()):.6f}"
            f"  dec_extractor={grad_norm(decoder.extractor.parameters()):.6f}"
            f"  enc_output={grad_norm(encoder.output.parameters()):.6f}"
            f"  enc_wm_encoder={grad_norm(encoder.watermark_encoder.parameters()):.6f}")

    nn.utils.clip_grad_norm_(
        list(encoder.parameters()) + list(decoder.parameters()),
        max_norm=5.0
    )
    optimizer.step()

    with torch.no_grad():
      acc = bit_accuracy(watermarks, extracted_wms)
      psnr =  PSNR(images, watermarked_imgs)

    total_acc += acc
    total_psnr += psnr

    for key in total_losses:
      total_losses[key] += loss_dict[key]
    num_batches += 1

    pbar.set_postfix({
        'loss': f"{loss_dict['total']:.4f}",
        'img': f"{loss_dict['image_loss']:.4f}",
        'wm': f"{loss_dict['watermark_loss']:.4f}",
        'cst': f"{loss_dict['constraint_loss']:.4f}",
        'acc': f"{acc:.1f}%",
        'psnr': f"{psnr:.1f}dB"
    })
  avg_losses = {k: v / num_batches for k, v in total_losses.items()}
  avg_acc = total_acc / num_batches
  avg_psnr = total_psnr / num_batches
  return avg_losses, avg_acc, avg_psnr

def train(encoder, decoder, noise_layer, train_loader, criterion, optimizer, scheduler, device, num_epochs=10, patience=5, attack_type=None):
  best = 0
  patience_counter = 0
  for epoch in range(num_epochs):
    train_losses, train_acc, train_psnr = train_epoch(
        encoder, decoder, noise_layer, train_loader,
        criterion, optimizer, device, epoch, attack_type
    )
    scheduler.step()
    print(f"\nEpoch {epoch} Summary:")
    print(f"  Total Loss: {train_losses['total']:.6f}")
    print(f"  Image Loss: {train_losses['image_loss']:.6f}")
    print(f"  Watermark Loss: {train_losses['watermark_loss']:.6f}")
    print(f"  Constraint Loss: {train_losses['constraint_loss']:.6f}")
    print(f"  Bit Accuracy:    {train_acc:.2f}%")
    print(f"  PSNR:             {train_psnr:.2f} dB")
    print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

    if train_acc > best:
      best = train_acc
      patience_counter = 0
      torch.save({
          'epoch': epoch,
          'encoder_state': encoder.state_dict(),
          'decoder_state': decoder.state_dict(),
          'optimizer_state': optimizer.state_dict(),
          'loss': train_losses['total'],
          'bit_accuracy': train_acc,
          'psnr': train_psnr
      }, 'best_model.pth')
      print(f"Saved best model with acc = {best:.2f}, psnr = {train_psnr:.2f}%")
    else:
      patience_counter += 1
      if patience_counter >= patience:
        print("Early stopping at epoch {epoch}")
        break