import os
import numpy as np
import torch
import argparse
import pydicom
from pydicom.data import get_testdata_file
from perceptnet.network_unet import UNetRes

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='input path')
    parser.add_argument('--output_dir', type=str, required=True, help='output path')
    parser.add_argument('--model', type=str, required=True, help='Generator checkpoint G_epochxx.pth')

    parser.add_argument('--in_nc', type=int, default=1)
    parser.add_argument('--out_nc', type=int, default=1)
    parser.add_argument('--nc', type=int, nargs='+', default=[64, 128, 256, 512])
    parser.add_argument('--nb', type=int, default=4)
    parser.add_argument('--act_mode', type=str, default='R')
    parser.add_argument('--downsample_mode', type=str, default='strideconv')
    parser.add_argument('--upsample_mode', type=str, default='convtranspose')

    parser.add_argument('--no_cuda', action='store_true', help='cpu')
    parser.add_argument('--no_norm', action='store_true', help='skip normalization')
    return parser.parse_args()


def load_dicom_series(series_folder):
    dcm_items = []
    for fname in os.listdir(series_folder):
        fpath = os.path.join(series_folder, fname)
        try:
            ds = pydicom.dcmread(fpath)
            sloc = float(ds.SliceLocation) if hasattr(ds, "SliceLocation") else 0.0
            dcm_items.append((sloc, ds))
        except Exception:
            continue

    dcm_items.sort(key=lambda x: x[0])
    slice_arrays = []
    dcm_origin_list = []
    for _, ds in dcm_items:
        img = ds.pixel_array.astype(np.float32)
        slice_arrays.append(img)
        dcm_origin_list.append(ds)
    vol = np.stack(slice_arrays, axis=0)  # (N, H, W)
    return vol, dcm_origin_list


def save_pred_dicom(pred_volume, origin_dcm_list, out_folder):
    os.makedirs(out_folder, exist_ok=True)
    N = pred_volume.shape[0]
    assert N == len(origin_dcm_list), "miss match"

    for idx in range(N):
        pred_slice = pred_volume[idx]
        ori_ds = origin_dcm_list[idx]
        new_ds = ori_ds.copy()
        min_ori = ori_ds.pixel_array.min()
        max_ori = ori_ds.pixel_array.max()
        pred_rescaled = pred_slice * (max_ori - min_ori) + min_ori
        pred_rescaled = np.clip(pred_rescaled, min_ori, max_ori)
        new_ds.PixelData = pred_rescaled.astype(ori_ds.pixel_array.dtype).tobytes()
        save_path = os.path.join(out_folder, f"pred_{idx:04d}.dcm")
        new_ds.save_as(save_path)


def main():
    args = get_args()
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    print(f"Using device: {device}")

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
    print(f"Loaded model checkpoint: {args.model}")

    input_vol, ori_dcm_list = load_dicom_series(args.input_dir)
    N, H, W = input_vol.shape
    print(f"Loaded DICOM series: {N} slices, H={H}, W={W}")

    if not args.no_norm:
        input_vol = (input_vol - input_vol.min()) / (input_vol.max() - input_vol.min() + 1e-8)

    x = torch.from_numpy(input_vol).float().unsqueeze(1).to(device)

    with torch.no_grad():
        pred_out = model(x)  # (N,1,H,W)

    pred_vol = pred_out.squeeze(1).cpu().numpy()
    save_pred_dicom(pred_vol, ori_dcm_list, args.output_dir)
    print(f"Prediction DICOM saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
