import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import argparse
from tqdm import tqdm
import pydicom


def get_args():
    parser = argparse.ArgumentParser(description='Pre-train Perceptual Feature Extractor via Autoencoder Reconstruction')
    parser.add_argument('--input_dir', type=str, required=True, help='DICOM input root')
    parser.add_argument('--output_dir', type=str, default='', help='')
    parser.add_argument('--batch_size', type=int, default=2, help='')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lr_min', type=float, default=1e-6)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--val_split', type=float, default=0.15)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--in_nc', type=int, default=1, help='')
    parser.add_argument('--base_dim', type=int, default=32, help='')
    return parser.parse_args()


def load_dicom_series(series_folder):
    dcm_files = []
    for fname in os.listdir(series_folder):
        fpath = os.path.join(series_folder, fname)
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True)
            dcm_files.append((ds.SliceLocation, fpath))
        except Exception:
            continue

    dcm_files.sort(key=lambda x: x[0])
    slices = []
    for _, path in dcm_files:
        ds = pydicom.dcmread(path)
        img = ds.pixel_array.astype(np.float32)
        slices.append(img)
    vol = np.stack(slices, axis=0)  # (N, H, W)
    return vol


class DicomAutoencoderDataset(Dataset):
    def __init__(self, input_root):
        self.input_root = input_root
        self.series_ids = sorted([
            d for d in os.listdir(input_root)
            if os.path.isdir(os.path.join(input_root, d))
        ])

    def __len__(self):
        return len(self.series_ids)

    def norm(self, img):
        min_val = np.min(img)
        max_val = np.max(img)
        img = (img - min_val) / (max_val - min_val + 1e-8)
        return img.astype(np.float32)

    def __getitem__(self, idx):
        sid = self.series_ids[idx]
        series_path = os.path.join(self.input_root, sid)
        vol = load_dicom_series(series_path)
        vol = self.norm(vol)
        tensor = torch.from_numpy(vol).unsqueeze(1)  # (N, 1, H, W)
        return tensor, tensor


class PerceptualEncoder(nn.Module):
    def __init__(self, in_channels=1, base_dim=32):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_dim, base_dim, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(base_dim, base_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_dim * 2, base_dim * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_dim * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(base_dim * 2, base_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_dim * 4, base_dim * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_dim * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        feat1 = self.stage1(x)
        feat2 = self.stage2(feat1)
        feat3 = self.stage3(feat2)
        return [feat1, feat2, feat3]

class ReconstructionDecoder(nn.Module):
    def __init__(self, out_channels=1, base_dim=32):
        super().__init__()
        self.stage3_up = nn.Sequential(
            nn.ConvTranspose2d(base_dim * 4, base_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_dim * 2, base_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_dim * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.stage2_up = nn.Sequential(
            nn.ConvTranspose2d(base_dim * 2, base_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_dim, base_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.stage1_up = nn.Sequential(
            nn.ConvTranspose2d(base_dim, base_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_dim, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, feats):
        feat1, feat2, feat3 = feats
        x = self.stage3_up(feat3)
        x = self.stage2_up(x + feat2)
        x = self.stage1_up(x + feat1)
        return x


class PerceptAutoencoder(nn.Module):
    def __init__(self, in_channels=1, base_dim=32):
        super().__init__()
        self.encoder = PerceptualEncoder(in_channels, base_dim)
        self.decoder = ReconstructionDecoder(in_channels, base_dim)

    def forward(self, x):
        feats = self.encoder(x)
        recon = self.decoder(feats)
        return recon


def train_one_epoch(loader, model, optimizer, scaler, args, device):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Train")

    for input_vol, target_vol in pbar:
        B, N, C, H, W = input_vol.shape
        input_vol = input_vol.to(device)
        target_vol = target_vol.to(device)

        input_2d = input_vol.reshape(B * N, C, H, W)
        target_2d = target_vol.reshape(B * N, C, H, W)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=args.use_amp):
            recon_2d = model(input_2d)
            loss = nn.functional.mse_loss(recon_2d, target_2d)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix({"recon_loss": f"{loss.item():.6f}"})

    return total_loss / len(loader)


@torch.no_grad()
def val_one_epoch(loader, model, args, device):
    model.eval()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Val  ")

    for input_vol, target_vol in pbar:
        B, N, C, H, W = input_vol.shape
        input_vol = input_vol.to(device)
        target_vol = target_vol.to(device)

        input_2d = input_vol.reshape(B * N, C, H, W)
        target_2d = target_vol.reshape(B * N, C, H, W)

        with torch.cuda.amp.autocast(enabled=args.use_amp):
            recon_2d = model(input_2d)
            loss = nn.functional.mse_loss(recon_2d, target_2d)

        total_loss += loss.item()
        pbar.set_postfix({"recon_loss": f"{loss.item():.6f}"})

    return total_loss / len(loader)


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    full_dataset = DicomAutoencoderDataset(args.input_dir)
    n_total = len(full_dataset)
    n_val = max(1, int(n_total * args.val_split))
    indices = list(range(n_total))
    random.shuffle(indices)

    train_dataset = Subset(full_dataset, indices[n_val:])
    val_dataset = Subset(full_dataset, indices[:n_val])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"Total series: {n_total}, Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    model = PerceptAutoencoder(in_channels=args.in_nc, base_dim=args.base_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        print(f"\n=== Epoch {epoch+1}/{args.epochs} ===")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6e}")

        train_loss = train_one_epoch(train_loader, model, optimizer, scaler, args, device)
        val_loss = val_one_epoch(val_loader, model, args, device)
        scheduler.step()

        print(f"Train Recon Loss: {train_loss:.6f}")
        print(f"Val   Recon Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            encoder_path = os.path.join(args.output_dir, "best_percept_encoder.pth")
            torch.save(model.encoder.state_dict(), encoder_path)
            print(f"New best encoder saved to {encoder_path}")

        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(args.output_dir, f"autoencoder_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Full autoencoder checkpoint saved to {ckpt_path}")

    print("\n saved in:", os.path.join(args.output_dir, "best_percept_encoder.pth"))


if __name__ == '__main__':
    main()
