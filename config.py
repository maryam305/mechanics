"""
config.py — Central Configuration for Dual-Method Spinal Curvature Analysis
============================================================================
Unified settings for SpinePose (Anatomical) + Geometric pipelines,
calibration model, dataset paths, and reproducibility parameters.
"""

import os

# ── Reproducibility ────────────────────────────────────────────────
REPRODUCIBILITY_NOTE = (
    "All random seeds fixed at 42. Results are deterministic given the same input."
)
RANDOM_SEED = 42

# ── Directory Structure ────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
RAW_VIDEO_DIR   = os.path.join(BASE_DIR, "data", "raw_videos")
OUTPUT_DIR      = os.path.join(BASE_DIR, "outputs")
FIGURE_DIR      = os.path.join(OUTPUT_DIR, "figures")
CSV_DIR         = os.path.join(OUTPUT_DIR, "csv")
MODEL_DIR       = os.path.join(BASE_DIR, "models", "calibration")
DATASET_DIR     = os.path.join(BASE_DIR, "data", "datasets")
OWN_DATASET_PATH = os.path.join(DATASET_DIR, "own_dataset")

for d in [RAW_VIDEO_DIR, OUTPUT_DIR, FIGURE_DIR, CSV_DIR, MODEL_DIR, DATASET_DIR, OWN_DATASET_PATH]:
    os.makedirs(d, exist_ok=True)

# ── Video Metadata ─────────────────────────────────────────────────
VIDEO_INFO = [
    {"display_name": "Normal Posture",   "filename": "normal_posture.mp4",  "label": "normal"},
    {"display_name": "Mild Kyphosis",    "filename": "mild_kyphosis.mp4",   "label": "mild"},
    {"display_name": "Severe Kyphosis",  "filename": "severe_kyphosis.mp4", "label": "severe"},
]

# ── SpinePose Model Configuration ─────────────────────────────────
SPINEPOSE_CONFIG = {
    "model_path":  os.path.join(BASE_DIR, "models", "spinepose", "spinepose_cvpr2025.pth"),
    "conf_thresh": 0.5,       # Minimum keypoint confidence to accept frame
    "input_size":  (256, 192) # Model input resolution (H, W)
}

# ── Contour / Geometric Configuration ────────────────────────────
CONTOUR_CONFIG = {
    "mask_area_min":    500,   # Minimum silhouette pixel area (px^2)
    "contour_pts_min":  20,    # Minimum back-contour points for fitting
    "poly_degree":      3,     # Polynomial degree for curve fitting (cubic)
    "smoothing_window": 9,     # Temporal smoothing window (frames)
    "iqr_multiplier":   1.5,   # IQR outlier rejection factor
}

# ── Reliability Thresholds ────────────────────────────────────────
RELIABILITY_CONF_THRESH = 0.5   # Warn if mean SpinePose confidence < this
REJECTION_RATE_WARN     = 40.0  # Warn if rejection rate (%) exceeds this

# ── Calibration Model Configuration ──────────────────────────────
CALIBRATION_CONFIG = {
    "n_estimators":  200,
    "random_state":  RANDOM_SEED,
    "test_size":     0.2,
    "features": [
        "spinepose_angle",
        "surface_curvature",
        "angle_diff",          # spinepose - surface
        "trunk_lean",
        "confidence",
        "curvature_variance",
    ],
    "target": "spinepose_angle_smoothed",
}

# ── Dataset Paths ─────────────────────────────────────────────────
# Mendeley Posture Dataset (used for clinical range validation)
MENDELEY_DATASET_PATH = os.path.join(DATASET_DIR, "mendeley_posture")
# SpineTrack Dataset (keypoint structure reference)
SPINETRACK_DATASET_PATH = os.path.join(DATASET_DIR, "spinetrack")
# Own collected dataset (training + evaluation)
OWN_DATASET_PATH = os.path.join(DATASET_DIR, "own_dataset")
OWN_LABELS_CSV   = os.path.join(OWN_DATASET_PATH, "labels.csv")  # cols: filename, label

# ── Clinical Reference Ranges (from Mendeley/Literature) ─────────
CLINICAL_RANGES = {
    "kyphosis": {
        "normal":   (20, 40),   # degrees
        "mild":     (40, 60),
        "severe":   (60, 999),
    },
    "geometric": {
        "normal":   (0, 20),
        "mild":     (20, 40),
        "severe":   (40, 999),
    }
}

# ── Plot Style ────────────────────────────────────────────────────
PLOT_COLORS = {
    "anatomical": "#1f77b4",
    "geometric":  "#ff7f0e",
    "calibrated": "#2ca02c",
    "reference":  "#d62728",
}
