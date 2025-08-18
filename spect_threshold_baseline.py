import SimpleITK as sitk
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from pathlib import Path

# --- paths ---
SPECT_PATH = "p_20_scan_2_pet_resample_crop_whole_body_Resample_Fregistered_to_TP_2_Resample_4.42.nii.gz"
MASK_PATH  = "p20_tumor_1.nii.gz"

# --- load ---
spect_img = sitk.ReadImage(SPECT_PATH)         # 3D image
mask_img  = sitk.ReadImage(MASK_PATH)          # 3D label image (should be 0/1+)

# helpers
def to_np(img):  # SITK to numpy array as (z,y,x)
    return sitk.GetArrayFromImage(img).astype(np.float32)

spect_np = to_np(spect_img)
mask_np  = to_np(mask_img)

print("SPECT shape (z,y,x):", spect_np.shape, "spacing (x,y,z):", spect_img.GetSpacing())
print("MASK  shape (z,y,x):", mask_np.shape,  "spacing (x,y,z):", mask_img.GetSpacing())

# Thresholding
finite = np.isfinite(spect_np)
if not finite.any():
    raise ValueError("SPECT has no finite voxels.")

Imax = float(spect_np[finite].max())
p = 0.40                                  # your first experiment
thr = p * Imax
print(f"Imax={Imax:.3f}, p={p:.2f}, threshold={thr:.3f}")

pred_np = (spect_np >= thr).astype(np.uint8)

#Keep the largest hot component
def keep_largest_component(bin_np, connectivity=1):
    labeled, n = ndi.label(bin_np, structure=ndi.generate_binary_structure(3, connectivity))
    if n == 0:
        return bin_np
    sizes = ndi.sum(bin_np, labeled, index=np.arange(1, n+1))
    keep_label = int(np.argmax(sizes)) + 1
    return (labeled == keep_label).astype(np.uint8)

# try both and see effect later
pred_np_largest = keep_largest_component(pred_np, connectivity=1)

#Save prediction as NIfTI(in Spect's geometry)
def save_like(ref_img, arr_np, out_path):
    out = sitk.GetImageFromArray(arr_np.astype(np.uint8))
    out.CopyInformation(ref_img)
    sitk.WriteImage(out, out_path)

OUT_DIR = Path("outputs"); OUT_DIR.mkdir(exist_ok=True)
save_like(spect_img, pred_np, OUT_DIR/"pred_p40_raw.nii.gz")
save_like(spect_img, pred_np_largest, OUT_DIR/"pred_p40_largest.nii.gz")
print("Saved:", OUT_DIR/"pred_p40_raw.nii.gz", "and", OUT_DIR/"pred_p40_largest.nii.gz")

#Dice Function
def dice(a, b):
    a = (a > 0).astype(np.uint8)
    b = (b > 0).astype(np.uint8)
    inter = int((a & b).sum())
    s = int(a.sum() + b.sum())
    return 1.0 if s == 0 else 2.0 * inter / s

#Volume(mL)
sx, sy, sz = spect_img.GetSpacing()   # (x,y,z) mm
vox_ml = (sx * sy * sz) / 1000.0
vol_pred_raw  = float(pred_np.sum() * vox_ml)
vol_pred_lc   = float(pred_np_largest.sum() * vox_ml)
vol_gt        = float((to_np(mask_img) > 0).sum() * vox_ml)

print(f"GT volume: {vol_gt:.2f} mL")
print(f"Pred (raw): {vol_pred_raw:.2f} mL")
print(f"Pred (largest): {vol_pred_lc:.2f} mL")
print("Dice (raw):", dice(pred_np, (mask_np>0)))
print("Dice (largest):", dice(pred_np_largest, (mask_np>0)))




