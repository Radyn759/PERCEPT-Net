import numpy as np
import os
import csv
import argparse
import pydicom
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def bootstrap_confidence_interval(data, n_bootstrap=1000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    data = np.asarray(data)
    n = len(data)

    # 有放回重抽样
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        boot_means.append(np.mean(sample))

    boot_means = np.sort(boot_means)
    alpha = 1 - ci
    ci_low = boot_means[int(alpha / 2 * n_bootstrap)]
    ci_high = boot_means[int((1 - alpha / 2) * n_bootstrap)]

    return {
        "mean": np.mean(data),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "std": np.std(data, ddof=1)
    }


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
    return np.stack(slices, axis=0)


def calculate_metrics(pred, gt, data_range=1.0):
    pred = np.clip(pred, 0, data_range)
    gt = np.clip(gt, 0, data_range)

    psnr_val = psnr(gt, pred, data_range=data_range)
    ssim_val = ssim(gt, pred, data_range=data_range)
    mae_val = np.mean(np.abs(pred - gt))
    rmse_val = np.sqrt(np.mean((pred - gt) ** 2))

    return {
        "PSNR": psnr_val,
        "SSIM": ssim_val,
        "MAE": mae_val,
        "RMSE": rmse_val
    }


def bootstrap_confidence_interval(data, n_bootstrap=1000, ci=0.95, seed=42):
    rng = np.random.default_rng(seed)
    data = np.asarray(data)
    n = len(data)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.sort(boot_means)
    alpha = 1 - ci
    return np.mean(data), boot_means[int(alpha / 2 * n_bootstrap)], boot_means[int((1 - alpha / 2) * n_bootstrap)]


def main():
    parser = argparse.ArgumentParser(description='Endpoint metrics & statistical analysis')
    parser.add_argument('--pred_dir', type=str, required=True, help='')
    parser.add_argument('--gt_dir', type=str, required=True, help='')
    parser.add_argument('--output_csv', type=str, default='metrics_statistics.csv')
    parser.add_argument('--seed', type=int, default=42, help='')
    args = parser.parse_args()

    series_ids = sorted([d for d in os.listdir(args.pred_dir)
                         if os.path.isdir(os.path.join(args.pred_dir, d))])

    all_metrics = {"PSNR": [], "SSIM": [], "MAE": [], "RMSE": []}
    slice_details = []

    for sid in tqdm(series_ids, desc="Evaluating"):
        pred_vol = load_dicom_series(os.path.join(args.pred_dir, sid))
        gt_vol = load_dicom_series(os.path.join(args.gt_dir, sid))

        # 归一化到[0,1]
        pred_vol = (pred_vol - pred_vol.min()) / (pred_vol.max() - pred_vol.min() + 1e-8)
        gt_vol = (gt_vol - gt_vol.min()) / (gt_vol.max() - gt_vol.min() + 1e-8)

        for slice_idx in range(pred_vol.shape[0]):
            pred_slice = pred_vol[slice_idx]
            gt_slice = gt_vol[slice_idx]
            metrics = calculate_metrics(pred_slice, gt_slice)

            slice_details.append({
                "series_id": sid,
                "slice_idx": slice_idx,
                **metrics
            })

            for k in all_metrics:
                all_metrics[k].append(metrics[k])

    summary = []
    for metric_name, values in all_metrics.items():
        mean, ci_low, ci_high = bootstrap_confidence_interval(values, seed=args.seed)
        summary.append({
            "metric": metric_name,
            "mean": f"{mean:.4f}",
            "95%CI_low": f"{ci_low:.4f}",
            "95%CI_high": f"{ci_high:.4f}",
            "std": f"{np.std(values, ddof=1):.4f}"
        })

    # 保存汇总统计表
    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "mean", "95%CI_low", "95%CI_high", "std"])
        writer.writeheader()
        writer.writerows(summary)

    # 保存逐切片明细表
    detail_csv = args.output_csv.replace(".csv", "_detail.csv")
    with open(detail_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["series_id", "slice_idx", "PSNR", "SSIM", "MAE", "RMSE"])
        writer.writeheader()
        writer.writerows(slice_details)

    print(f"result: {args.output_csv}")
    print(f"detail: {detail_csv}")
    print("\n95% metric:")
    for row in summary:
        print(f"  {row['metric']}: {row['mean']} [{row['95%CI_low']}, {row['95%CI_high']}]")


if __name__ == '__main__':
    main()