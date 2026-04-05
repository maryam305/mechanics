import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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
    # Animation video commented out — SpinePose is designed for real humans,
    # not 3D animations, and takes extremely long on this video.
    # {
    #     "filename": "Male Walk Cycle Animation Reference _ Front & Side Views & Realtime playback speed.mp4",
    #     "display_name": "Animation_Reference",
    #     "condition": "unloaded",
    #     "subject_id": "animation_ref",
    # },
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

# ── Output directories ────────────────────────────────────────────
OUTPUT_DIR   = os.path.join(BASE_DIR, "output_analysis")
VID_DIR      = os.path.join(OUTPUT_DIR, "videos")
PLOT_DIR     = os.path.join(OUTPUT_DIR, "plots")
REPORT_DIR   = os.path.join(OUTPUT_DIR, "reports")

# ── Contour extraction ────────────────────────────────────────────
CONTOUR_CONFIG = {
    "iqr_multiplier": 1.5,
    "poly_degree": 3,              # 3 is better for sparse spine keypoints (4 overfits)
    "roi_pad_px": 30,              # padding around SpinePose bbox for ROI crop
}

# ── SpinePose ─────────────────────────────────────────────────────
SPINEPOSE_CONFIG = {
    "device": "cpu",
    "conf_thresh": 0.3,
}

# ── Normative values ─────────────────────────────────────────────
# Thoracic kyphosis & lumbar lordosis: Ohlendorf et al. (2023) Sci Rep
# 13:12395 used video rasterstereography (surface measurement).
# The multi-segment cumulative Cobb from 2D SpinePose landmarks
# typically gives ~15-25% lower values than surface rasterstereography
# due to fewer segments and 2D projection.  We keep the original
# reference but note the method difference in the plots.
NORMS = {
    "thoracic_kyphosis": {"mean": 47.3, "sd": 10.5, "ref": "Ohlendorf 2023 (surface)"},
    "lumbar_lordosis":   {"mean": 28.1, "sd": 9.3,  "ref": "Ohlendorf 2023 (surface)"},
    "lumbar_bending":    {"mean": 11.0, "sd": 5.0,  "ref": "Ohlendorf 2023"},
    "trunk_lean":        {"mean": 4.0,  "sd": 3.0,  "ref": "Lyu & LaBat 2016; Aslam 2025"},
}

# ── Subject info ──────────────────────────────────────────────────
SUBJECT_HEIGHT_CM = 183.0
SUBJECT_SEX       = "male"
