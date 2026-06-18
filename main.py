import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.dataset import CustomDataset
from utils.losses import WatermarkLoss
from models.blocks.encoder import Encoder
from models.blocks.decoder import Decoder
from models.blocks.noise_layer import NoiseLayer
from train import train

import math
import numpy as np
import mathplotlib.pyplot as plt

if __name__ == "__main__":
    train_dataset = CustomDataset(img_dir='/content/drive/MyDrive/Colab Notebooks/dataset/rowsformer/DIV2K_train_HR', img_size=128, bit_length=64)
    valid_dataset = CustomDataset(img_dir='/content/drive/MyDrive/Colab Notebooks/dataset/rowsformer/DIV2K_valid_HR', img_size=128, bit_length=64)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Custom attack_list for training
    attack_list = [
    'none',
    'salt_and_pepper_noise',
    'rotation'
    ]
    for attack_name in attack_list:
        encoder = Encoder(in_channels=3, dim=64, num_stages=3, num_heads=4, window_size=8, watermark_length=64).to(device)
        decoder = Decoder(in_channels=3, dim=64, num_stages=3, num_heads=4, window_size=8, watermark_length=64).to(device)
        noise_layer = NoiseLayer()
        criterion = WatermarkLoss(lambda1=0.0, lambda2=1.0, lambda3=0.5)

        params = list(encoder.parameters()) + list(decoder.parameters())
        optimizer = torch.optim.AdamW(params, lr=2e-4, weight_decay=0.001)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40, eta_min=1e-6)

        # Phase 1
        print(f"\nPhase 1: Watermark-only, 10 epochs")
        criterion.lambda1 = 0.0
        criterion.lambda2 = 1.0
        train(encoder=encoder, decoder=decoder, noise_layer=noise_layer,
            train_loader=train_loader, criterion=criterion,
            optimizer=optimizer, scheduler=scheduler, device=device,
            num_epochs=10, patience=10, attack_type='none')

        # Phase 2
        print(f"\nPhase 2: Balanced, 10 epochs")
        criterion.lambda1 = 1.0
        criterion.lambda2 = 1.0
        train(encoder=encoder, decoder=decoder, noise_layer=noise_layer,
            train_loader=train_loader, criterion=criterion,
            optimizer=optimizer, scheduler=scheduler, device=device,
            num_epochs=10, patience=10, attack_type='none')

        # Phase 3:
        print(f"\nPhase 3: ({attack_name}), 15 epochs")
        criterion.lambda1 = 5.0
        criterion.lambda2 = 1.0
        train(encoder=encoder, decoder=decoder, noise_layer=noise_layer,
            train_loader=train_loader, criterion=criterion,
            optimizer=optimizer, scheduler=scheduler, device=device,
            num_epochs=15, patience=10, attack_type=attack_name)

        checkpoint = torch.load('best_model.pth', map_location=device)
        save_path = f'best_model_{attack_name}.pth'
        torch.save(checkpoint, save_path)
        print(f"\nSaved model for [{attack_name}] to {save_path}")
        
    # Visualization of watermark extraction for each attack
    noise_layer_test = NoiseLayer()
    for attack_name in attack_list:
        encoder = Encoder(in_channels=3, dim=64, num_stages=3, num_heads=4, window_size=8, watermark_length=64).to(device)
        decoder = Decoder(in_channels=3, dim=64, num_stages=3, num_heads=4, window_size=8, watermark_length=64).to(device)

        path = f'best_model_{attack_name}.pth'
        checkpoint = torch.load(path, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state'])
        decoder.load_state_dict(checkpoint['decoder_state'])
        encoder.eval()
        decoder.eval()

        with torch.no_grad():
            img, wm = next(iter(valid_loader))
            img = img[:1].to(device)
            wm = wm[:1].to(device)

            watermarked = encoder(img, wm)
            attacked = noise_layer_test(watermarked, attack_type=attack_name)
            extracted = decoder(attacked)
            pred = (torch.sigmoid(extracted) > 0.5).float()

            acc = (pred == wm).float().mean().item() * 100
            mse = F.mse_loss(watermarked, img).item()
            psnr = 10 * math.log10(1.0 / (mse + 1e-10))

        orig_bits = wm.squeeze().cpu().numpy().astype(int)
        extr_bits = pred.squeeze().cpu().numpy().astype(int)
        match = (orig_bits == extr_bits).astype(int)
        num_correct = match.sum()
        num_total = len(orig_bits)

        fig = plt.figure(figsize=(20, 8))
        gs = fig.add_gridspec(2, 3, height_ratios=[3, 1.5], hspace=0.35)

        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(img[0].cpu().permute(1,2,0).clamp(0,1))
        ax0.set_title('Original')
        ax0.axis('off')

        ax1 = fig.add_subplot(gs[0, 1])
        ax1.imshow(watermarked[0].cpu().permute(1,2,0).clamp(0,1))
        ax1.set_title(f'Watermarked\nPSNR={psnr:.1f}dB')
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[0, 2])
        ax2.imshow(attacked[0].cpu().permute(1,2,0).clamp(0,1))
        ax2.set_title(f'{attack_name}\nAcc={acc:.1f}%')
        ax2.axis('off')

        ax3 = fig.add_subplot(gs[1, :])
        colors = np.zeros((3, num_total, 3))
        for j in range(num_total):
            colors[0, j] = [0.2, 0.4, 0.9] if orig_bits[j] == 1 else [0.9, 0.9, 0.9]
            if match[j]:
                colors[1, j] = [0.2, 0.8, 0.3] if extr_bits[j] == 1 else [0.85, 0.95, 0.85]
            else:
                colors[1, j] = [0.9, 0.2, 0.2] if extr_bits[j] == 1 else [1.0, 0.8, 0.8]

            colors[2, j] = [0.2, 0.8, 0.3] if match[j] else [0.9, 0.2, 0.2]

        ax3.imshow(colors, aspect='auto', interpolation='nearest')
        ax3.set_yticks([0, 1, 2])
        ax3.set_yticklabels(['Original WM', 'Extracted WM', f'Match ({num_correct}/{num_total})'])
        ax3.set_xlabel('Bit Index (64 bits)')
        ax3.set_title(f'Watermark Comparison — Accuracy: {acc:.1f}%')
        ax3.set_xticks(range(0, num_total, 4))

        plt.suptitle(f'Model: best_model_{attack_name}.pth', fontweight='bold', fontsize=13)
        plt.tight_layout()
        plt.show()

        print(f"\n[{attack_name}] PSNR={psnr:.2f}dB | Acc={acc:.2f}%")
        print(f"  Original : {''.join(map(str, orig_bits))}")
        print(f"  Extracted: {''.join(map(str, extr_bits))}")

