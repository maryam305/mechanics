import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Global Research Transparency ──────────────────────────────────
# All parameters (thresholds, smoothing window, polynomial degree) are fixed 
# and documented to ensure reproducibility.
REPRODUCIBILITY_NOTE = (
    "All parameters (thresholds, smoothing window, polynomial degree) are fixed "
    "and documented to ensure reproducibility."
)
COORDINATE_CONVENTION = "Image coordinates (y-axis increases downward)"

METHOD_TYPE = {
    "spinepose": "anatomical",
    "silhouette": "geometric"
}

# ── Video classification ──────────────────────────────────────────
VIDEO_INFO = [
    {
        "filename": "ya rab.mp4",
        "display_name": "Subject1_Unloaded",
        "condition": "unloaded",
        "subject_id": "subject_1",
    },
    {
        "filename": "rab aded.mp4",
        "display_name": "Subject1_Loaded",
        "condition": "loaded",
        "subject_id": "subject_1",
    },
    {
        "filename": "Recording 2026-04-04 184226.mp4",
        "display_name": "Recording_Unloaded",
        "condition": "unloaded",
        "subject_id": "recording_1",
    },
    {
        "filename": "Video Project 3.mp4",
        "display_name": "VideoProject3_Unloaded",
        "condition": "unloaded",
        "subject_id": "project_3",
    },
]

# ── Directory Structure ───────────────────────────────────────────
RAW_VIDEO_DIR = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_DIR    = os.path.join(BASE_DIR, "data", "outputs")
VID_DIR       = os.path.join(OUTPUT_DIR, "videos")
PLOT_DIR      = os.path.join(OUTPUT_DIR, "plots")
REPORT_DIR    = os.path.join(OUTPUT_DIR, "reports")

# ── Pipeline Parameters ───────────────────────────────────────────
CONTOUR_CONFIG = {
    "iqr_multiplier": 1.5,
    "poly_degree": 3,              # 3 avoids overfitting (degree 4 introduces artificial curvature)
    "roi_pad_px": 30,              
    "mask_area_min": 500,          # Empirically chosen to reject incomplete segmentations
    "contour_pts_min": 20,
    "smoothing_window": 9,         # Temporal moving average size
}

# ── SpinePose ─────────────────────────────────────────────────────
SPINEPOSE_CONFIG = {
    "device": "cpu",
    "conf_thresh": 0.3,
}

# ── Normative values & Significance ──────────────────────────────
# Clinical Interpretations:
# - Differences > 5° between geometric and anatomical methods may be biomechanically significant.
# - Keypoint confidence < 0.5 indicates unreliable detection.
BIOMECHANICAL_SIGNIFICANCE_DEG = 5.0
RELIABILITY_CONF_THRESH = 0.5

NORMS = {
    "thoracic_kyphosis": {"mean": 47.3, "sd": 10.5, "ref": "Ohlendorf 2023 (surface)"},
    "lumbar_lordosis":   {"mean": 28.1, "sd": 9.3,  "ref": "Ohlendorf 2023 (surface)"},
    "lumbar_bending":    {"mean": 11.0, "sd": 5.0,  "ref": "Ohlendorf 2023"},
    "trunk_lean":        {"mean": 4.0,  "sd": 3.0,  "ref": "Lyu & LaBat 2016; Aslam 2025"},
}

# ── Aesthetics ────────────────────────────────────────────────────
COLORS = {
    "geometric": "#2196F3",  # Blue
    "anatomical": "#F44336", # Red
}

# ── Subject info ──────────────────────────────────────────────────
SUBJECT_HEIGHT_CM = 183.0
SUBJECT_SEX       = "male"
