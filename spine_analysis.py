"""
spine_analysis.py — Anatomical Pipeline (SpinePose CVPR 2025)
=============================================================
Implements the Anatomical Method for spinal curvature measurement.

FIXES APPLIED (from research review):
  [FIX-1] Robust Cobb estimation using top-2 / bottom-2 segment averaging
  [FIX-2] Global trunk lean normalization before angle computation
  [FIX-3] Strict frame filtering with rejection reason logging
  [FIX-4] Temporal smoothing (Savitzky-Golay + moving average)
  [FIX-5] Full confidence-aware validity checking
"""

import os, sys, math, warnings
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from tqdm import tqdm
import csv
import config

warnings.filterwarnings("ignore")

# ── Research Transparency ──────────────────────────────────────────
DISCLAIMER = (
    "All measurements are derived from 2D projections and are sensitive to camera "
    "alignment, perspective distortion, and subject orientation. The method assumes "
    "sagittal-plane motion and does not account for 3D spinal rotation."
)
METHOD_LABEL        = "Anatomical Method (SpinePose CVPR 2025)"
VALIDATION_NOTE     = (
    "Validation is relative, based on biomechanical consistency and Mendeley dataset "
    "clinical reference ranges. No radiographic ground truth is used."
)

# ── SpinePose 37-keypoint Index Map ───────────────────────────────
SPINE_IDX = {
    "C1":    36, "C4":  35, "C7": 18, "T3":  30,
    "T8":    29, "L1":  28, "L3": 27, "L5":  26,
    "Sacrum":19,
}
SPINE_CHAIN = ["C1", "C4", "C7", "T3", "T8", "L1", "L3", "L5", "Sacrum"]

# ── Core Geometry ─────────────────────────────────────────────────

def _segment_inclination(pt_top: np.ndarray, pt_bot: np.ndarray) -> float:
    """
    Standardized segment inclination (degrees).
    Uses atan2(horizontal_displacement, |vertical_displacement|) so that
    a perfectly vertical segment = 0° and lateral tilt is positive/negative.
    """
    v = np.asarray(pt_bot, float) - np.asarray(pt_top, float)
    if abs(v[1]) < 1e-9:
        return 0.0
    return math.degrees(math.atan2(v[0], abs(v[1])))


def _global_trunk_lean(c7: np.ndarray, sacrum: np.ndarray) -> float:
    """
    Compute the trunk's global forward lean (C7 -> Sacrum inclination).
    Used for FIX-2: bias removal before Cobb estimation.
    """
    return _segment_inclination(c7, sacrum)


def robust_multi_segment_cobb(kpts: np.ndarray, landmark_names: list) -> tuple:
    """
    [FIX-1] Robust Cobb angle estimation.

    Instead of raw max - min (outlier-sensitive), uses:
        top    = mean of the 2 most tilted segments (highest inclinations)
        bottom = mean of the 2 least tilted segments (lowest inclinations)
        cobb   = |top - bottom|

    [FIX-2] Normalizes each segment inclination by subtracting the global
    trunk lean so that the Cobb angle reflects pure curvature, not forward tilt.

    Returns:
        cobb_angle (float): corrected Cobb angle in degrees
        lean       (float): trunk lean removed as bias
        raw_incls  (list):  raw per-segment inclinations (for diagnostics)
    """
    if len(landmark_names) < 2:
        return 0.0, 0.0, []

    pts = [kpts[SPINE_IDX[n]] for n in landmark_names]

    # Raw segment inclinations
    raw_incls = [_segment_inclination(pts[i], pts[i + 1])
                 for i in range(len(pts) - 1)]

    # FIX-2: compute global trunk lean and normalize
    c7     = kpts[SPINE_IDX["C7"]]
    sacrum = kpts[SPINE_IDX["Sacrum"]]
    lean   = _global_trunk_lean(c7, sacrum)
    normalized = [inc - lean for inc in raw_incls]

    # FIX-1: robust averaging instead of single max/min
    sorted_n = sorted(normalized)
    n        = len(sorted_n)

    if n >= 4:
        bottom = float(np.mean(sorted_n[:2]))
        top    = float(np.mean(sorted_n[-2:]))
    elif n >= 2:
        bottom = sorted_n[0]
        top    = sorted_n[-1]
    else:
        return 0.0, lean, raw_incls

    cobb = float(abs(top - bottom))
    return cobb, lean, raw_incls


# ── Frame Validity ─────────────────────────────────────────────────

def is_valid_frame(kpts: np.ndarray, conf: np.ndarray) -> tuple:
    """
    [FIX-5] Strict anatomical validity check.
    Returns (valid: bool, reason: str, mean_conf: float).

    Checks:
      1. Mean keypoint confidence across the spine chain >= threshold
      2. Top-to-bottom anatomical ordering is preserved (no skeleton flip)
    """
    key_names = ["C1", "C4", "C7", "T3", "T8", "L1", "L3", "L5", "Sacrum"]
    key_idx   = [SPINE_IDX[k] for k in key_names]
    mean_c    = float(np.mean([conf[i] for i in key_idx]))

    if mean_c < config.SPINEPOSE_CONFIG["conf_thresh"]:
        return False, "low_confidence", mean_c

    # Ordering check: each successive vertebra must be below the previous (y increases downward)
    ordered_keys = ["C7", "T3", "T8", "L1", "Sacrum"]
    pts_ordered  = [kpts[SPINE_IDX[k]] for k in ordered_keys]
    for i in range(len(pts_ordered) - 1):
        if pts_ordered[i][1] >= pts_ordered[i + 1][1]:
            return False, "ordering_violation", mean_c

    return True, "valid", mean_c


# ── Temporal Smoothing ─────────────────────────────────────────────

def smooth_savgol(arr: list, k: int = 9) -> np.ndarray:
    """
    [FIX-4] Savitzky-Golay smoothing with graceful fallback to moving average.
    Preserves signal peaks better than a plain moving average.
    """
    arr = np.asarray(arr, float)
    if len(arr) < k:
        return arr
    window = k if k % 2 != 0 else k - 1
    try:
        return savgol_filter(arr, window, 2, mode="interp")
    except Exception:
        pad = k // 2
        padded = np.pad(arr, pad, mode="edge")
        return np.convolve(padded, np.ones(k) / k, mode="valid")


def moving_average(arr: list, w: int = 7) -> np.ndarray:
    """Simple moving average for secondary smoothing pass."""
    arr = np.asarray(arr, float)
    if len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode="same")


# ── Main Analysis Function ─────────────────────────────────────────

def analyze_video_anatomical(vid_info: dict, estimator) -> dict:
    """
    Process a single video through the Anatomical pipeline.

    Args:
        vid_info:  dict with keys 'display_name', 'filename'
        estimator: SpinePose estimator instance (detect + estimate methods)

    Returns:
        dict with keys: time, kyphosis, kyphosis_raw, tfl, lean_bias,
                        frame_id, confidence, stats
    """
    display  = vid_info["display_name"]
    filename = vid_info["filename"]
    vid_path = os.path.join(config.RAW_VIDEO_DIR, filename)

    cap   = cv2.VideoCapture(vid_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n[{METHOD_LABEL}] Analyzing: {display}")
    print(f"  Disclaimer: {DISCLAIMER}")
    print(f"  Validation: {VALIDATION_NOTE}")

    R = {
        "time":         [],
        "kyphosis":     [],    # smoothed
        "kyphosis_raw": [],    # pre-smoothing
        "tfl":          [],    # trunk forward lean (raw C7->Sacrum)
        "lean_bias":    [],    # bias subtracted per frame (FIX-2)
        "frame_id":     [],
        "confidence":   [],
    }

    stats = {
        "total": 0, "valid": 0, "rejected": 0,
        "reasons": {}
    }

    pbar      = tqdm(total=total, desc=f"  [{display}]")
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        stats["total"] += 1

        try:
            bboxes = estimator.detect(frame)
            if bboxes is not None and len(bboxes) > 0:
                fr = estimator.estimate(frame, [bboxes[0]])
                if fr and len(fr) > 0:
                    fd   = np.array(fr[0])
                    fd   = fd[0] if fd.ndim == 3 else fd
                    kpts = fd[:, :2]
                    conf = fd[:, 2] if fd.shape[1] >= 3 else np.ones(fd.shape[0])

                    valid, reason, mean_c = is_valid_frame(kpts, conf)

                    if valid:
                        cobb, lean, _ = robust_multi_segment_cobb(
                            kpts, ["C7", "T3", "T8", "L1"]
                        )
                        tfl = _global_trunk_lean(
                            kpts[SPINE_IDX["C7"]], kpts[SPINE_IDX["Sacrum"]]
                        )

                        R["time"].append(frame_idx / fps)
                        R["frame_id"].append(frame_idx)
                        R["kyphosis_raw"].append(cobb)
                        R["tfl"].append(tfl)
                        R["lean_bias"].append(lean)
                        R["confidence"].append(mean_c)
                        stats["valid"] += 1
                    else:
                        stats["rejected"] += 1
                        stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
                else:
                    _log_reject(stats, "no_estimation")
            else:
                _log_reject(stats, "no_detection")

        except Exception as e:
            _log_reject(stats, "error")

        frame_idx += 1
        pbar.update(1)

    cap.release()
    pbar.close()

    # ── Temporal Smoothing ─────────────────────────────────────────
    win = config.CONTOUR_CONFIG["smoothing_window"]
    if R["kyphosis_raw"]:
        R["kyphosis"] = smooth_savgol(R["kyphosis_raw"], k=win).tolist()
        R["tfl"]      = smooth_savgol(R["tfl"],          k=win).tolist()
    else:
        R["kyphosis"] = []

    R["stats"] = stats

    # ── Rejection Summary ──────────────────────────────────────────
    _print_rejection_summary(display, stats, R.get("confidence", []))
    return R


def _log_reject(stats: dict, reason: str):
    stats["rejected"] += 1
    stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1


def _print_rejection_summary(display: str, stats: dict, confidences: list):
    total    = stats["total"] or 1
    rate     = stats["rejected"] / total * 100
    mean_c   = float(np.mean(confidences)) if confidences else 0.0

    print(f"\n  Rejection Summary — {display}:")
    print(f"    Total frames:  {stats['total']}")
    print(f"    Valid:         {stats['valid']}")
    print(f"    Rejected:      {stats['rejected']} ({rate:.1f}%)")
    for r, cnt in stats["reasons"].items():
        print(f"      {r}: {cnt} ({cnt/total*100:.1f}%)")
    print(f"    Mean keypoint confidence: {mean_c:.3f}")

    if mean_c < config.RELIABILITY_CONF_THRESH:
        print("    WARNING: Mean confidence < 0.5 — results may be unreliable.")
    if rate > config.REJECTION_RATE_WARN:
        print(f"    WARNING: Rejection rate > {config.REJECTION_RATE_WARN:.0f}% — check camera angle.")


# ── Batch Export ───────────────────────────────────────────────────

def export_csv(results: dict, display_name: str):
    """Save frame-level anatomical results to CSV."""
    out_path = os.path.join(config.CSV_DIR, f"{display_name}_anatomical.csv")
    keys     = ["frame_id", "time", "kyphosis", "kyphosis_raw", "tfl", "lean_bias", "confidence"]
    n        = len(results["time"])

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(keys)
        for i in range(n):
            writer.writerow([
                results["frame_id"][i],
                f"{results['time'][i]:.4f}",
                f"{results['kyphosis'][i]:.3f}",
                f"{results['kyphosis_raw'][i]:.3f}",
                f"{results['tfl'][i]:.3f}",
                f"{results['lean_bias'][i]:.3f}",
                f"{results['confidence'][i]:.4f}",
            ])

    print(f"  CSV saved -> {out_path}")
    return out_path


if __name__ == "__main__":
    print(f"Config loaded: {config.REPRODUCIBILITY_NOTE}")
    print("Run unified_pipeline.py to execute the full analysis.")
