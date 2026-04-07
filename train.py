import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import argparse
from tqdm import tqdm
from perceptnet.network_unet import UNetRes

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='nii input')
    parser.add_argument('--gt_dir', type=str, required=True, help='nii GT')
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
    return parser.parse_args()

class NiiDataset(Dataset):
    def __init__(self, input_dir, gt_dir):
        self.input_paths = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
        self.gt_paths = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
        assert len(self.input_paths) == len(self.gt_paths)

    def __len__(self):
        return len(self.input_paths)

    def norm(self, img):
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img.astype(np.float32)

    def __getitem__(self, idx):

        input_nii = nib.load(self.input_paths[idx])
        input_img = input_nii.get_fdata()
        input_img = self.norm(input_img)
      
        gt_nii = nib.load(self.gt_paths[idx])
        gt_img = gt_nii.get_fdata()
        gt_img = self.norm(gt_img)

        input_img = torch.from_numpy(input_img).unsqueeze(0)
        gt_img = torch.from_numpy(gt_img).unsqueeze(0)
        return input_img, gt_img

class Discriminator(nn.Module):
    def __init__(self, in_nc=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_nc*2, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv3d(64, 128, 4, 2, 1), nn.BatchNorm3d(128), nn.LeakyReLU(0.2),
            nn.Conv3d(128, 256, 4, 2, 1), nn.BatchNorm3d(256), nn.LeakyReLU(0.2),
            nn.Conv3d(256, 1, 4, 1, 1)
        )

    def forward(self, x, y):
        inp = torch.cat([x, y], dim=1)
        return self.net(inp)

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = args.device
  
    dataset = NiiDataset(args.input_dir, args.gt_dir)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    generator = UNetRes(
        in_nc=args.in_nc, out_nc=args.out_nc, nc=args.nc, nb=args.nb,
        act_mode=args.act_mode, downsample_mode=args.downsample_mode, upsample_mode=args.upsample_mode
    ).to(device)
    
    opt_G = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))

    criterion_gan = nn.BCEWithLogitsLoss()
    criterion_pixel = nn.L1Loss()

    for epoch in range(args.epochs):
        generator.train()
        discriminator.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for input_img, gt_img in pbar:
            input_img = input_img.to(device)
            gt_img = gt_img.to(device)
            bs = input_img.size(0)
          
            real_label = torch.ones(bs, 1, 1, 1, 1).to(device)
            fake_label = torch.zeros(bs, 1, 1, 1, 1).to(device)
=
            opt_G.zero_grad()
            fake_img = generator(input_img)  # 核心：output = model(input)

            pred_fake = discriminator(fake_img, input_img)
            loss_G_gan = criterion_gan(pred_fake, real_label)
            
            loss_G_pixel = criterion_pixel(fake_img, gt_img)
            
            loss_G = args.lambda_gan * loss_G_gan + args.lambda_pixel * loss_G_pixel
            loss_G.backward()
            opt_G.step()

            opt_D.zero_grad()
            
            pred_real = discriminator(gt_img, input_img)
            loss_D_real = criterion_gan(pred_real, real_label)

            pred_fake = discriminator(fake_img.detach(), input_img)
            loss_D_fake = criterion_gan(pred_fake, fake_label)
            
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            opt_D.step()

            pbar.set_postfix({"G": loss_G.item(), "D": loss_D.item()})

        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(), os.path.join(args.output_dir, f"G_epoch{epoch+1}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(args.output_dir, f"D_epoch{epoch+1}.pth"))

if __name__ == '__main__':
    main()
