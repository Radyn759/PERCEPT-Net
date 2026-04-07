import numpy as np
import torch
import nibabel as nib
import argparse
from perceptnet.network_unet import UNetRes

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='input nii file')
    parser.add_argument('--output', type=str, required=True, help='output nii file')
    parser.add_argument('--model', type=str, required=True, help='checkpoint.pth')

    parser.add_argument('--in_nc', type=int, default=1)
    parser.add_argument('--out_nc', type=int, default=1)
    parser.add_argument('--nc', type=int, nargs='+', default=[64, 128, 256, 512])
    parser.add_argument('--nb', type=int, default=4)
    parser.add_argument('--act_mode', type=str, default='R')
    parser.add_argument('--downsample_mode', type=str, default='strideconv')
    parser.add_argument('--upsample_mode', type=str, default='convtranspose')

    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--no_norm', action='store_true')
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device('cpu' if args.no_cuda else 'cuda' if torch.cuda.is_available() else 'cpu')

    model = UNetRes(
        in_nc=args.in_nc,
        out_nc=args.out_nc,
        nc=args.nc,
        nb=args.nb,
        act_mode=args.act_mode,
        downsample_mode=args.downsample_mode,
        upsample_mode=args.upsample_mode
    ).to(device)

    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    nii = nib.load(args.input)
    data = nii.get_fdata()
    affine = nii.affine
    header = nii.header

    if not args.no_norm:
        data = (data - data.min()) / (data.max() - data.min() + 1e-8)

    x = torch.from_numpy(data).float().unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(x)

    out_data = output.squeeze().cpu().numpy()

    nib.save(nib.Nifti1Image(out_data, affine, header), args.output)

if __name__ == '__main__':
    main()
