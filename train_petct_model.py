import os
import numpy as np
import pydicom
import nibabel as nib
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ================= CONFIG =================

DATASET_DIR = r"D:\ACRIN_FLT_BREAST\ACRIN-FLT-Breast"  
MODEL_OUT = "petct_model.pkl"

# ================= SAFETY CHECK =================

print("🚀 PET/CT training started")
print("📂 Dataset path:", DATASET_DIR)

if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(f"❌ Dataset not found at {DATASET_DIR}")

# ================= DATA HOLDERS =================

X = []   # features
y = []   # labels

# ================= LOADERS =================

def load_volume(path):
    """Load DICOM or NIfTI volume"""
    try:
        if path.endswith(".dcm"):
            ds = pydicom.dcmread(path)
            return ds.pixel_array.astype(np.float32)

        elif path.endswith((".nii", ".nii.gz")):
            return nib.load(path).get_fdata()

    except Exception as e:
        print("⚠️ Failed to read:", path)
        return None

    return None

# ================= FEATURE EXTRACTION =================

def extract_features(volume):
    """Simple PET/CT radiomics-style features"""
    volume = volume[np.isfinite(volume)]  # remove NaNs

    return [
        np.mean(volume),
        np.std(volume),
        np.max(volume),
        np.percentile(volume, 95)
    ]

# ================= LABELING (CLINICALLY INSPIRED) =================

def assign_label(volume):
    """
    SUV-based heuristic labeling:
    0 = Normal
    1 = Benign
    2 = Malignant
    """
    mean_suv = np.mean(volume)

    if mean_suv < 2.5:
        return 0
    elif mean_suv < 5.0:
        return 1
    else:
        return 2

# ================= DATASET WALK =================

processed = 0

for root, _, files in os.walk(DATASET_DIR):
    for f in files:
        if not f.lower().endswith((".dcm", ".nii", ".nii.gz")):
            continue

        file_path = os.path.join(root, f)
        volume = load_volume(file_path)

        if volume is None or volume.size == 0:
            continue

        features = extract_features(volume)
        label = assign_label(volume)

        X.append(features)
        y.append(label)

        processed += 1
        if processed % 100 == 0:
            print(f"Processed {processed} scans")

# ================= FINAL DATA CHECK =================

X = np.array(X)
y = np.array(y)

print("📊 Total samples extracted:", X.shape)

if X.shape[0] < 10:
    raise ValueError("❌ Not enough PET/CT samples for training")

# ================= TRAIN / TEST SPLIT =================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ================= MODEL =================

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ================= EVALUATION =================

print("\n✅ Training complete\n")
print(classification_report(
    y_test,
    model.predict(X_test),
    target_names=["Normal", "Benign", "Malignant"]
))

# ================= SAVE MODEL =================

joblib.dump(model, MODEL_OUT)
print(f"\n💾 Model saved as: {MODEL_OUT}")
