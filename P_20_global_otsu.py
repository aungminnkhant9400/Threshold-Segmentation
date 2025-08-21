#!/usr/bin/env python3
"""
Global Otsu thresholding for whole-body SPECT (no ROI).
Inputs are hard-coded to your files.
"""

import numpy as np
import SimpleITK as sitk
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------- file paths -----------------------------
spect_path = "P_20/p_20_scan_2_pet_resample_crop_whole_body_Resample_Fregistered_to_TP_2_Resample_4.42.nii.gz"
gt_path    = "P_20/tumor/p20_tumor_1.nii.gz"
outdir     = Path("P_20_outputs_otsu_global"); outdir.mkdir(exist_ok=True, parents=True)

# ----------------------------- utils -----------------------------
def to_np(img): return sitk.GetArrayFromImage(img)

def save_like(ref_img, arr_np, out_path):
    out = sitk.GetImageFromArray(arr_np.astype(np.uint8))
    out.CopyInformation(ref_img)
    sitk.WriteImage(out, str(out_path))

def dice(a, b):
    a = (a > 0).astype(np.uint8); b = (b > 0).astype(np.uint8)
    inter = int((a & b).sum()); s = int(a.sum() + b.sum())
    return 1.0 if s == 0 else 2.0 * inter / s

def otsu_threshold_1d(values, nbins=256):
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
# Load images
spect_img = sitk.ReadImage(spect_path)
gt_img    = sitk.ReadImage(gt_path)

spect = to_np(spect_img).astype(np.float32)
gt    = (to_np(gt_img) > 0).astype(np.uint8)

# Collect intensities (finite only)
finite = np.isfinite(spect)
vals   = spect[finite]
# Clip tails to stabilize (optional)
lo, hi = np.percentile(vals, [0.5, 99.5])
vals = np.clip(vals, lo, hi)

# Otsu on whole volume
thr, (H, edges, k_idx) = otsu_threshold_1d(vals)
print(f"[result] Global Otsu threshold = {thr:.6g}")

# RAW mask
pred_raw = (spect >= thr).astype(np.uint8)
pred_raw[~finite] = 0

# Remove tiny specks
labeled, n = ndi.label(pred_raw)
if n > 0:
    sizes = ndi.sum(pred_raw, labeled, index=np.arange(1, n+1))
    keep = np.where(sizes >= 50)[0] + 1
    pred_raw = np.isin(labeled, keep).astype(np.uint8)

# Largest-component heuristic
labeled, n = ndi.label(pred_raw)
if n > 0:
    sizes = ndi.sum(pred_raw, labeled, index=np.arange(1, n+1))
    keep_label = int(np.argmax(sizes)) + 1
    pred_largest = (labeled == keep_label).astype(np.uint8)
else:
    pred_largest = pred_raw.copy()

# Save outputs
raw_path     = outdir / "pred_global_otsu_raw.nii.gz"
largest_path = outdir / "pred_global_otsu_largest.nii.gz"
save_like(spect_img, pred_raw, raw_path)
save_like(spect_img, pred_largest, largest_path)
print(f"[saved] RAW mask     : {raw_path}")
print(f"[saved] LARGEST mask : {largest_path}")

# Histogram figure
centers = 0.5 * (edges[:-1] + edges[1:])
plt.figure(figsize=(8,5))
plt.bar(centers, H, width=(edges[1]-edges[0]))
plt.axvline(thr, linestyle='--', linewidth=2)
plt.xlabel("Voxel intensity (global)")
plt.ylabel("Count")
plt.title("Global Otsu: histogram & threshold")
plt.tight_layout()
hist_path = outdir / "otsu_global_histogram.png"
plt.savefig(hist_path, dpi=150)
plt.close()
print(f"[saved] Histogram: {hist_path}")

# Metrics
sx, sy, sz = spect_img.GetSpacing()
vox_ml = (sx * sy * sz) / 1000.0

def report(name, pred_np):
    d = dice(pred_np, gt)
    vg = gt.sum() * vox_ml
    vp = pred_np.sum() * vox_ml
    print(f"[metrics:{name}] Dice={d:.3f} | GT Vol={vg:.2f} mL | Pred Vol={vp:.2f} mL")

report("RAW",     pred_raw)
report("LARGEST", pred_largest)
