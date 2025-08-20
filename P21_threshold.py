import SimpleITK as sitk
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from pathlib import Path

# --- paths ---
SPECT_PATH = "P_21/p_21_scan_2_pet_resample_crop_whole_body_Resample_Fregistered_to_TP_2_Resample_4.42.nii.gz"
MASK_PATH  = "P_21/tumor/tumor_1.nii.gz"

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
p = 0.40                # your first experiment
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

OUT_DIR = Path("P_21_Outputs"); OUT_DIR.mkdir(exist_ok=True)
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

#ROI method
# 1) Load ROI and make sure it matches SPECT geometry
roi_img = sitk.ReadImage("P_21/roi_tumor_zone.nii.gz")

def resample_like(moving_img, fixed_img, is_label=True):
    interp = sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear
    return sitk.Resample(
        moving_img, fixed_img, sitk.Transform(), interp,
        fixed_img.GetOrigin(), fixed_img.GetSpacing(), fixed_img.GetDirection(),
        0, moving_img.GetPixelID()
    )

if (roi_img.GetSize()!=spect_img.GetSize() or
    roi_img.GetSpacing()!=spect_img.GetSpacing() or
    roi_img.GetDirection()!=spect_img.GetDirection() or
    roi_img.GetOrigin()!=spect_img.GetOrigin()):
    roi_img = resample_like(roi_img, spect_img, is_label=True)

roi_np = sitk.GetArrayFromImage(roi_img).astype(np.uint8)
roi = roi_np > 0

# --- Guards: ROI must contain voxels, and they must be finite in SPECT
if roi.sum() == 0:
    raise ValueError("ROI is empty (no voxels > 0). Did you export the brush as an image labelmap?")

finite_roi = np.isfinite(spect_np) & roi
if finite_roi.sum() == 0:
    raise ValueError("No finite SPECT voxels inside ROI. Check ROI location/registration.")

# 2) Compute Imax **inside ROI only** and threshold
p_roi = 0.50  # keep a separate variable name to avoid confusion with earlier p
Imax_roi = float(spect_np[finite_roi].max())
thr_roi = p_roi * Imax_roi

pred_roi_np = ((spect_np >= thr_roi) & roi).astype(np.uint8)

# 3) Keep the component connected to the hottest ROI voxel; fallback to largest component if needed
seed_idx = np.unravel_index(np.argmax((spect_np * roi).astype(np.float32)), spect_np.shape)

labeled, n = ndi.label(pred_roi_np)
seed_label = labeled[seed_idx]
if seed_label == 0 and n > 0:
    # seed not in any blob; fallback to largest component
    sizes = ndi.sum(pred_roi_np, labeled, index=np.arange(1, n+1))
    keep_label = int(np.argmax(sizes)) + 1
    pred_roi_np = (labeled == keep_label).astype(np.uint8)
elif seed_label != 0:
    pred_roi_np = (labeled == seed_label).astype(np.uint8)
# else: no components at all -> pred_roi_np stays empty

# 4) Save and evaluate
out_path = OUT_DIR / f"pred_p{int(p_roi*100):02d}_roi_seeded.nii.gz"
save_like(spect_img, pred_roi_np, out_path)
print("Saved:", out_path)

# --- Metrics for ROI prediction
def dice(a, b):
    a = (a > 0).astype(np.uint8)
    b = (b > 0).astype(np.uint8)
    inter = int((a & b).sum()); s = int(a.sum() + b.sum())
    return 1.0 if s == 0 else 2.0 * inter / s

sx, sy, sz = spect_img.GetSpacing()
vox_ml = (sx * sy * sz) / 1000.0

vol_gt   = (mask_np > 0).sum() * vox_ml
vol_pred = pred_roi_np.sum() * vox_ml
dice_roi = dice(pred_roi_np, (mask_np > 0))

print(f"GT volume:    {vol_gt:.2f} mL")
print(f"Pred (ROI):   {vol_pred:.2f} mL")
print(f"Dice (ROI):   {dice_roi:.3f}")
print(f"Threshold (ROI): p={p_roi:.2f}, Imax_roi={Imax_roi:.3f}, thr={thr_roi:.3f}")


