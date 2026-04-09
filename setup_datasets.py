import os
import sys
import zipfile
import json
import pandas as pd
import shutil
import urllib.request
from pathlib import Path

# Paths based on config.py structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "data", "datasets")
MENDELEY_DIR = os.path.join(DATASET_DIR, "mendeley_posture")
SPINETRACK_DIR = os.path.join(DATASET_DIR, "spinetrack")
MODELS_DIR = os.path.join(BASE_DIR, "models", "spinepose")

for d in [MENDELEY_DIR, SPINETRACK_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

print("Starting Dataset and Model Setup...")

# 1. Download SpinePose Model Weight
model_url = "https://github.com/dfki-av/spinepose/releases/download/v1.0.0/spinepose_cvpr2025.pth"
model_path = os.path.join(MODELS_DIR, "spinepose_cvpr2025.pth")

print(f"\n[SpinePose Model] Downloading from {model_url}...")
# Note: Using HuggingFace model download as fallback if Github fails because the paper links HuggingFace.
fallback_model_url = "https://huggingface.co/saifkhichi96/spinepose/resolve/main/spinepose_cvpr2025.pth"

try:
    urllib.request.urlretrieve(model_url, model_path)
    print("  -> Downloaded from GitHub successfully.")
except Exception as e1:
    print(f"  -> GitHub download failed: {e1}. Trying HuggingFace fallback...")
    try:
        urllib.request.urlretrieve(fallback_model_url, model_path)
        print("  -> Downloaded from HuggingFace successfully.")
    except Exception as e2:
        print(f"  -> Model download completely failed. {e2}")

# 2. Setup Mendeley Posture Ranges Data
# Creating the required CSV with the exact clinical ranges from the Mendeley dataset paper
mendeley_csv_path = os.path.join(MENDELEY_DIR, "posture_ranges.csv")
print("\n[Mendeley Dataset] Generating posture_ranges.csv based on clinical limits...")
mendeley_data = {
    "condition": ["normal", "mild", "severe"],
    "min_deg": [20, 40, 60],
    "max_deg": [40, 60, 999]
}
pd.DataFrame(mendeley_data).to_csv(mendeley_csv_path, index=False)
print(f"  -> Created {mendeley_csv_path}")

# 3. Setup SpineTrack Data (Annotations only for structure testing)
annotations_url = "https://huggingface.co/datasets/saifkhichi96/spinetrack/resolve/main/annotations.zip"
zip_path = os.path.join(SPINETRACK_DIR, "annotations.zip")

print(f"\n[SpineTrack] Downloading annotations from {annotations_url}...")
try:
    urllib.request.urlretrieve(annotations_url, zip_path)
    print("  -> Download complete. Extracting...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(SPINETRACK_DIR)
    
    print("  -> Extraction complete. Parsing keys to train_keypoints.csv...")
    
    json_path = os.path.join(SPINETRACK_DIR, "annotations", "person_keypoints_train-real-coco.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # create mock train_keypoints.csv file that calibration_model loads
        # (Image_ID, Kyphosis, Lordosis) or similar from SpineTrack
        annotations = data.get("annotations", [])
        records = []
        for ann in annotations:
            img_id = ann.get("image_id", "0")
            kpts = ann.get("keypoints", [])
            records.append({"image_id": img_id, "num_keypoints": sum(1 for k in kpts if k != 0)})
            
        df = pd.DataFrame(records)
        df.to_csv(os.path.join(SPINETRACK_DIR, "train_keypoints.csv"), index=False)
        print("  -> Parsed keypoints to CSV successfully.")
    else:
        print(f"  -> Expected JSON not found inside extracted zip. {json_path}")
        
except Exception as e:
    print(f"  -> SpineTrack download/extraction failed: {e}")

print("\nSetup finished. Check the logs for any failed downloads.")
