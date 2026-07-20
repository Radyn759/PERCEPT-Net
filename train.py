import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
import pydicom
from perceptnet.network_unet import UNetRes

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='DICOM_input')
    parser.add_argument('--gt_dir', type=str, required=True, help='GT DICOM')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='save path')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda')
    
    parser.add_argument('--in_nc', type=int, default=1)
    parser.add_argument('--out_nc', type=int, default=1)
    parser.add_argument('--nc', type=int, nargs='+', default=[64, 128, 256, 512])
    parser.add_argument('--nb', type=int, default=4)
    parser.add_argument('--act_mode', type=str, default='R')
    parser.add_argument('--downsample_mode', type=str, default='strideconv')
    parser.add_argument('--upsample_mode', type=str, default='convtranspose')

    parser.add_argument('--lambda_gan', type=float, default=1.0)
    parser.add_argument('--lambda_pixel', type=float, default=100.0)
    parser.add_argument('--lambda_percept', type=float, default=0.5, help='perceptual loss weight')
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


class DicomDataset(Dataset):
    def __init__(self, input_root, gt_root):
        self.input_root = input_root
        self.gt_root = gt_root

        self.series_ids = sorted([
            d for d in os.listdir(input_root)
            if os.path.isdir(os.path.join(input_root, d))
        ])

        for sid in self.series_ids:
            gt_series_path = os.path.join(gt_root, sid)
            assert os.path.isdir(gt_series_path), f"missing: {sid}"

    def __len__(self):
        return len(self.series_ids)

    def norm(self, img):
        min_val = np.min(img)
        max_val = np.max(img)
        img = (img - min_val) / (max_val - min_val + 1e-8)
        return img.astype(np.float32)

    def __getitem__(self, idx):
        sid = self.series_ids[idx]
        input_series_path = os.path.join(self.input_root, sid)
        gt_series_path = os.path.join(self.gt_root, sid)

        input_vol = load_dicom_series(input_series_path)
        gt_vol = load_dicom_series(gt_series_path)

        input_vol = self.norm(input_vol)
        gt_vol = self.norm(gt_vol)

        input_tensor = torch.from_numpy(input_vol).unsqueeze(1)
        gt_tensor = torch.from_numpy(gt_vol).unsqueeze(1)

        return input_tensor, gt_tensor

class PatchGAN2D(nn.Module):
    def __init__(self, input_channels=2, ndf=64, n_layers=3):
        super().__init__()
        layers = []

        layers.append(nn.Conv2d(input_channels, ndf, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, True))

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers.append(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(ndf * nf_mult))
            layers.append(nn.LeakyReLU(0.2, True))


        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers.append(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1))
        layers.append(nn.BatchNorm2d(ndf * nf_mult))
        layers.append(nn.LeakyReLU(0.2, True))


        layers.append(nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img, cond):
        combined = torch.cat([img, cond], dim=1)
        return self.model(combined)

class PerceptualFeatureExtractor(nn.Module):
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

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        feat1 = self.stage1(x)
        feat2 = self.stage2(feat1)
        feat3 = self.stage3(feat2)
        return [feat1, feat2, feat3]

def calc_perceptual_loss(feat_extractor, pred_img, gt_img):
    pred_feats = feat_extractor(pred_img)
    gt_feats = feat_extractor(gt_img)
    loss = 0.0
    for pred_feat, gt_feat in zip(pred_feats, gt_feats):
        loss += nn.functional.l1_loss(pred_feat, gt_feat)
    return loss / len(pred_feats)
    
def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device

    dataset = DicomDataset(args.input_dir, args.gt_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    generator = UNetRes(
        in_nc=args.in_nc, out_nc=args.out_nc, nc=args.nc, nb=args.nb,
        act_mode=args.act_mode, downsample_mode=args.downsample_mode, upsample_mode=args.upsample_mode
    ).to(device)

    discriminator = PatchGAN2D(input_channels=2, ndf=64, n_layers=3).to(device)

    percept_extractor = PerceptualFeatureExtractor(in_channels=args.in_nc, base_dim=32).to(device)
    percept_extractor.eval()

    opt_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_pixel = nn.L1Loss()

    for epoch in range(args.epochs):
        generator.train()
        discriminator.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for input_vol, gt_vol in pbar:
            B, N, C, H, W = input_vol.shape
            input_vol = input_vol.to(device)
            gt_vol = gt_vol.to(device)

            input_2d = input_vol.reshape(B * N, C, H, W)
            gt_2d = gt_vol.reshape(B * N, C, H, W)

            dummy_out = discriminator(gt_2d, input_2d)
            _, _, ph, pw = dummy_out.shape
            real_label = torch.ones(B * N, 1, ph, pw).to(device)
            fake_label = torch.zeros(B * N, 1, ph, pw).to(device)

            opt_G.zero_grad()
            fake_2d = generator(input_2d)  # (B*N, 1, H, W)

            pred_fake = discriminator(fake_2d, input_2d)
            loss_G_gan = criterion_gan(pred_fake, real_label)
            loss_G_pixel = criterion_pixel(fake_2d, gt_2d)
            loss_G_percept = calc_perceptual_loss(percept_extractor, fake_2d, gt_2d)
            
            loss_G = args.lambda_gan * loss_G_gan + args.lambda_pixel * loss_G_pixel
            loss_G.backward()
            opt_G.step()

            opt_D.zero_grad()
            pred_real = discriminator(gt_2d, input_2d)
            loss_D_real = criterion_gan(pred_real, real_label)

            pred_fake_detach = discriminator(fake_2d.detach(), input_2d)
            loss_D_fake = criterion_gan(pred_fake_detach, fake_label)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            opt_D.step()

            pbar.set_postfix({
                "G_loss": loss_G.item(),
                "D_loss": loss_D.item(),
                "percept_loss": loss_G_percept.item()
            })

        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), os.path.join(args.output_dir, f"G_epoch{epoch+1}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(args.output_dir, f"D_epoch{epoch+1}.pth"))
            print(f"Save checkpoint epoch {epoch+1} to {args.output_dir}")


if __name__ == '__main__':
    main()
