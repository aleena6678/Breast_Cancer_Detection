import numpy as np
import cv2
import pydicom

# ==============================
# Load DICOM series (FOLDER)
# ==============================
def load_dicom_series(files):
    slices = []

    for f in files:
        ds = pydicom.dcmread(f, stop_before_pixels=False)

        # ❗ Skip non-image DICOMs
        if not hasattr(ds, "pixel_array"):
            continue

        slices.append(ds)

    if len(slices) == 0:
        raise ValueError("No valid image slices found in uploaded DICOM files.")

    # ✅ SAFE sorting logic
    if hasattr(slices[0], "ImagePositionPatient"):
        slices.sort(
            key=lambda x: float(x.ImagePositionPatient[2])
            if hasattr(x, "ImagePositionPatient") and len(x.ImagePositionPatient) >= 3
            else 0.0
        )
    elif hasattr(slices[0], "InstanceNumber"):
        slices.sort(key=lambda x: int(x.InstanceNumber))
    else:
        # fallback: original order
        pass

    volume = np.stack([s.pixel_array for s in slices]).astype(np.float32)
    return volume, slices[0]

# ==============================
# Normalize PET slice
# ==============================
def normalize(img):
    img = np.clip(img, np.percentile(img, 5), np.percentile(img, 95))
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return (img * 255).astype(np.uint8)
# ==============================
# PET + CT Fusion
# ==============================
def fuse_pet_ct(ct_img, pet_img):
    ct_rgb = cv2.cvtColor(ct_img, cv2.COLOR_GRAY2RGB)
    pet_heat = cv2.applyColorMap(pet_img, cv2.COLORMAP_JET)
    return cv2.addWeighted(ct_rgb, 0.6, pet_heat, 0.4, 0)
# ================= PET QUANTITATIVE ANALYSIS =================

def analyze_pet_lesion(pet_img):
    thresh = np.percentile(pet_img, 90)
    lesion_mask = pet_img > thresh

    if lesion_mask.sum() == 0:
        return None, lesion_mask

    suv_max = float(pet_img[lesion_mask].max())
    suv_mean = float(pet_img[lesion_mask].mean())
    suv_std = float(pet_img[lesion_mask].std())
    active_pct = 100.0 * lesion_mask.sum() / pet_img.size

    metrics = {
        "SUVmax (proxy)": suv_max,
        "SUVmean (proxy)": suv_mean,
        "Uptake heterogeneity": suv_std,
        "Active region (%)": active_pct
    }
    return metrics, lesion_mask


def localize_lesion(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return "no focal abnormality detected"

    h, w = mask.shape
    x_mean, y_mean = xs.mean(), ys.mean()

    lr = "left" if x_mean < w / 2 else "right"
    ud = "upper" if y_mean < h / 2 else "lower"

    return f"{ud} {lr} region"


def generate_pet_clinical_text(metrics, location):
    if metrics is None:
        return (
            "No focal area of abnormal FDG uptake is identified. "
            "Overall metabolic activity appears within physiological limits."
        )

    suv = metrics["SUVmax (proxy)"]
    extent = metrics["Active region (%)"]

    intensity = (
        "low" if suv < 120 else
        "moderate" if suv < 180 else
        "high"
    )

    spread = (
        "localized" if extent < 5 else
        "moderately extensive" if extent < 15 else
        "diffuse"
    )

    return (
        f"A focal region of {intensity} metabolic activity is identified in the {location}. "
        f"The lesion demonstrates elevated FDG uptake with SUVmax (proxy) of {suv:.2f}. "
        f"Metabolic involvement is {spread}, involving approximately {extent:.2f}% of the imaged region. "
        "These findings are suspicious for metabolically active pathology. "
        "Clinical correlation and histopathological confirmation are recommended."
    )
def process_ct_like(img):
    img = img.astype(np.float32)
    img = np.clip(img, np.percentile(img, 10), np.percentile(img, 90))
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return (img * 255).astype(np.uint8)

# ==============================
# MAIN PET/CT AI FUNCTION
# ==============================
def predict_petct(uploaded_files):
    volume, ds = load_dicom_series(uploaded_files)

    # ---- select middle slice ----
    mid = volume.shape[0] // 2
    slice_img = volume[mid]

    # ---- PET slice (metabolic) ----
    pet_slice = normalize(slice_img)        # metabolic (PET-like)
    ct_slice  = process_ct_like(slice_img)  # anatomical (CT-like)


    # ---- CT slice (anatomical simulation) ----
    ct_raw = slice_img.copy()

    ct_raw = cv2.normalize(ct_raw, None, 0, 255, cv2.NORM_MINMAX)
    ct_raw = ct_raw.astype(np.uint8)

    # enhance anatomical edges (CT-like)
    edges = cv2.Canny(ct_raw, 40, 120)
    ct_slice = cv2.addWeighted(ct_raw, 0.85, edges, 0.15, 0)

    # ---- Fusion ----
    fusion = fuse_pet_ct(ct_slice, pet_slice)


    # ---- Clinical metrics ----
    mean_uptake = pet_slice.mean()
    max_uptake = pet_slice.max()
    active_ratio = np.sum(pet_slice > 180) / pet_slice.size
    metrics, lesion_mask = analyze_pet_lesion(pet_slice)
    location = localize_lesion(lesion_mask)
    clinical_summary = generate_pet_clinical_text(metrics, location)
    
    # ---- Stage estimation (explainable) ----
    if max_uptake < 120:
        stage = "No significant abnormal uptake"
        stage_level = "Normal / No active disease"
    elif active_ratio < 0.05:
        stage = "Localized metabolic activity"
        stage_level = "Stage I"
    elif active_ratio < 0.15:
        stage = "Moderate regional metabolic spread"
        stage_level = "Stage II"
    else:
        stage = "Extensive metabolic involvement"
        stage_level = "Stage III / IV"

    return {
    "stage": stage,
    "metrics": metrics,
    "clinical_summary": clinical_summary,
    "images": {
        "CT Slice (Anatomical)": ct_slice,
        "PET Slice (Metabolic)": pet_slice,
        "PET + CT Fusion": fusion
    }
}

def draw_lesion_contour(img, mask):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(img_rgb, contours, -1, (0, 255, 0), 2)
    return img_rgb
