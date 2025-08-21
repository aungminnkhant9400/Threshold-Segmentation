#!/usr/bin/env python3
"""
Otsu's method for tumor segmentation INSIDE YOUR ROI on SPECT.

Inputs (defaults are your files):
  --spect  /mnt/data/p_20_scan_2_pet_resample_crop_whole_body_Resample_Fregistered_to_TP_2_Resample_4.42.nii.gz
  --gt     /mnt/data/p20_tumor_1.nii.gz
  --roi    /mnt/data/roi_tumor_zone.nii.gz

Outputs (in outputs_otsu/):
  - pred_otsu_roi.nii.gz       (binary prediction mask)
  - otsu_roi_histogram.png     (histogram with chosen threshold)
Console prints:
  - Otsu threshold (intensity)
  - Dice, GT volume (mL), Pred volume (mL)
"""

import argparse
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndi
import matplotlib.pyplot as plt


# ----------------------------- utilities -----------------------------
def resample_like(moving_img, fixed_img, is_label=True):
    interp = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    return sitk.Resample(
        moving_img, fixed_img, sitk.Transform(), interp,
        fixed_img.GetOrigin(), fixed_img.GetSpacing(), fixed_img.GetDirection(),
        0, moving_img.GetPixelID()
    )

def to_np(img):
    return sitk.GetArrayFromImage(img)

def save_like(ref_img, arr_np, out_path):
    out = sitk.GetImageFromArray(arr_np.astype(np.uint8))
    out.CopyInformation(ref_img)
    sitk.WriteImage(out, str(out_path))

def dice(a, b):
    a = (a > 0).astype(np.uint8); b = (b > 0).astype(np.uint8)
    inter = int((a & b).sum()); s = int(a.sum() + b.sum())
    return 1.0 if s == 0 else 2.0 * inter / s

def otsu_threshold_1d(values, nbins=256):
    """
    Manual Otsu on 1D intensities (so we can restrict to ROI values).
    Returns threshold as the bin-center maximizing between-class variance.
    """
    values = np.asarray(values, dtype=np.float64)
    hist, edges = np.histogram(values, bins=nbins)
    p = hist.astype(np.float64) / max(hist.sum(), 1.0)
    centers = 0.5 * (edges[:-1] + edges[1:])

    omega0 = np.cumsum(p)
    mu     = np.cumsum(p * centers)
    muT    = mu[-1] if p.sum() > 0 else 0.0

    eps = 1e-12
    omega1 = 1.0 - omega0
    mu0 = np.where(omega0 > eps, mu / (omega0 + eps), 0.0)
    mu1 = np.where(omega1 > eps, (muT - mu) / (omega1 + eps), 0.0)

    sigma_b2 = omega0 * omega1 * (mu0 - mu1) ** 2
    k = int(np.argmax(sigma_b2))
    return float(centers[k]), (hist, edges, k)


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Otsu ROI segmentation on SPECT")
    ap.add_argument("--spect", default="P_20/p_20_scan_2_pet_resample_crop_whole_body_Resample_Fregistered_to_TP_2_Resample_4.42.nii.gz")
    ap.add_argument("--gt",    default="P_20/tumor/p20_tumor_1.nii.gz")
    ap.add_argument("--roi",   default="P_20/roi_tumor_zone.nii.gz")
    ap.add_argument("--outdir", default="P_20_outputs_otsu")
    ap.add_argument("--nbins", type=int, default=256)
    ap.add_argument("--clip",  type=float, nargs=2, default=[0.5, 99.5],
                    help="Percentile clipping inside ROI (e.g., 0.5 99.5). Use 0 100 to disable.")
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="Optional Gaussian smoothing sigma in voxels (e.g., 1.0).")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(exist_ok=True, parents=True)

    # Load
    spect_img = sitk.ReadImage(args.spect)
    gt_img    = sitk.ReadImage(args.gt)
    roi_img   = sitk.ReadImage(args.roi)

    # Align ROI/GT to SPECT geometry if needed (nearest neighbor for labels)
    for name, img in [("gt", gt_img), ("roi", roi_img)]:
        if (img.GetSize() != spect_img.GetSize() or
            img.GetSpacing() != spect_img.GetSpacing() or
            img.GetDirection() != spect_img.GetDirection() or
            img.GetOrigin()   != spect_img.GetOrigin()):
            print(f"[info] Resampling {name} to SPECT space...")
            img_res = resample_like(img, spect_img, is_label=True)
            if name == "gt":   gt_img = img_res
            if name == "roi":  roi_img = img_res

    spect = to_np(spect_img).astype(np.float32)
    gt    = (to_np(gt_img)  > 0).astype(np.uint8)
    roi   = (to_np(roi_img) > 0).astype(np.uint8)

    # Optional smoothing (apply to a copy; we will still use original to find the hotspot)
    sm = spect.copy()
    if args.smooth and args.smooth > 0:
        print(f"[info] Gaussian smoothing (sigma={args.smooth} voxels)...")
        sm = ndi.gaussian_filter(sm, sigma=args.smooth)

    # ROI intensities
    finite_roi = np.isfinite(sm) & (roi > 0)
    if finite_roi.sum() == 0:
        raise ValueError("ROI has no finite voxels; check ROI placement/registration.")
    vals = sm[finite_roi]

    # Percentile clipping to reduce long tails (helps SPECT stability)
    p_lo, p_hi = args.clip
    if (p_lo, p_hi) != (0.0, 100.0):
        lo, hi = np.percentile(vals, [p_lo, p_hi])
        vals = np.clip(vals, lo, hi)

    # Otsu on ROI values
    thr, (H, edges, k_idx) = otsu_threshold_1d(vals, nbins=args.nbins)
    print(f"[result] Otsu threshold (ROI) = {thr:.6g}")

    # Build prediction inside ROI (on smoothed image if used)
    pred = ((sm >= thr) & (roi > 0)).astype(np.uint8)

    # Keep the component connected to the hottest voxel in ROI (on ORIGINAL spect for hotspot)
    seed_idx = np.unravel_index(np.argmax((spect * (roi > 0)).astype(np.float32)), spect.shape)
    labeled, n = ndi.label(pred)
    if n > 0:
        seed_label = labeled[seed_idx]
        if seed_label != 0:
            pred = (labeled == seed_label).astype(np.uint8)
        else:
            # fallback: largest component
            sizes = ndi.sum(pred, labeled, index=np.arange(1, n+1))
            keep_label = int(np.argmax(sizes)) + 1
            pred = (labeled == keep_label).astype(np.uint8)

    # Save predicted mask
    pred_path = outdir / "pred_otsu_roi.nii.gz"
    save_like(spect_img, pred, pred_path)
    print(f"[saved] Predicted mask: {pred_path}")

    # Histogram figure (based on ROI values used for Otsu)
    centers = 0.5 * (edges[:-1] + edges[1:])
    plt.figure(figsize=(8,5))
    plt.bar(centers, H, width=(edges[1]-edges[0]))
    plt.axvline(thr, linestyle='--', linewidth=2)
    plt.xlabel("Voxel intensity (ROI)")
    plt.ylabel("Count")
    plt.title("Otsu inside ROI: histogram & chosen threshold")
    plt.tight_layout()
    hist_path = outdir / "otsu_roi_histogram.png"
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"[saved] Histogram figure: {hist_path}")

    # Metrics
    sx, sy, sz = spect_img.GetSpacing()
    vox_ml = (sx * sy * sz) / 1000.0
    vol_gt   = float(gt.sum() * vox_ml)
    vol_pred = float(pred.sum() * vox_ml)
    dsc      = float(dice(pred, gt))

    print(f"[metrics] GT volume      : {vol_gt:.2f} mL")
    print(f"[metrics] Otsu ROI vol.  : {vol_pred:.2f} mL")
    print(f"[metrics] Dice (Otsu ROI): {dsc:.3f}")


if __name__ == "__main__":
    main()
