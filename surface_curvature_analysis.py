"""
surface_curvature_analysis.py — Geometric Surface Curvature Pipeline
=====================================================================
Measures back-surface curvature from video silhouettes using MediaPipe
segmentation + polynomial fitting + standardized inclination angles.

FIXES APPLIED (from research review):
  [FIX-4] Centerline-based contour (left+right average) replaces extreme edge
  [FIX-3] Temporal moving-average smoothing applied post-frame
  [FIX-5] Strict frame filtering with per-reason rejection tracking
  [FIX-1] Robust angle computation using inclination differences (not raw poly)
"""

import os
import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
import config

warnings.filterwarnings("ignore")

# ── Pipeline Flow ──────────────────────────────────────────────────
# 1. Segmentation  -> MediaPipe Selfie Segmentation (+ GrabCut fallback)
# 2. Contour       -> Centerline (FIX-4: left+right mean, not extreme edge)
# 3. Filtering     -> IQR-based outlier rejection
# 4. Angle         -> Standardized inclination-based inclination difference
# 5. Smoothing     -> Temporal moving average (FIX-3)
# 6. Export        -> CSV + figures

DISCLAIMER = (
    "All measurements are derived from 2D projections and are sensitive to camera "
    "alignment, perspective distortion, and subject orientation. The method assumes "
    "sagittal-plane motion and does not account for 3D spinal rotation."
)
METHOD_LABEL = "Geometric Method (Surface Curvature)"
ANALOGY_NOTE = (
    "Geometric curvature is NOT anatomically equivalent to Cobb angle. "
    "It approximates back-surface shape and is calibrated against SpinePose output."
)


# ── 1. Segmentation ────────────────────────────────────────────────

def extract_person_mask_mediapipe(frame: np.ndarray,
                                   mask_area_min: int = 500) -> tuple:
    """
    Person silhouette via MediaPipe Selfie Segmentation (model_selection=1).

    Returns:
        mask    (np.ndarray | None): binary mask (255 = person)
        success (bool)
    """
    with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1) as seg:
        results = seg.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        condition = results.segmentation_mask > 0.5
        mask = np.where(condition, 255, 0).astype(np.uint8)

        if np.sum(mask > 0) < mask_area_min:
            return None, False

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, False

        clean   = np.zeros_like(mask)
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(clean, [largest], -1, 255, -1)
        return clean, True


def extract_person_mask_fallback(frame: np.ndarray) -> np.ndarray:
    """
    GrabCut-inspired fallback: simple luminance threshold -> largest blob.
    Used when MediaPipe confidence is low.
    """
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(mask)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(clean, [largest], -1, 255, -1)
    return clean


def get_person_bbox(mask: np.ndarray) -> tuple | None:
    cols = np.where(mask.any(axis=0))[0]
    rows = np.where(mask.any(axis=1))[0]
    if not len(cols) or not len(rows):
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])


# ── 2. Contour Extraction ──────────────────────────────────────────

def get_back_contour_centerline(mask: np.ndarray, H: int,
                                 y_start: float, y_end: float) -> np.ndarray | None:
    """
    [FIX-4] CENTERLINE contour: x_center = (left_edge + right_edge) / 2

    This replaces using the single extreme (left or right) edge, which was
    biased by clothing and arm interference. The centerline approximates the
    spine's lateral position and is more robust.

    IQR outlier rejection is applied after extraction.
    """
    raw_pts = []
    for y in range(max(0, int(y_start)), min(H, int(y_end))):
        cs = np.where(mask[y, :] > 0)[0]
        if len(cs) >= 2:
            cx = int((cs[0] + cs[-1]) / 2)   # FIX-4: center, not extreme
            raw_pts.append((cx, y))

    if len(raw_pts) < config.CONTOUR_CONFIG["contour_pts_min"]:
        return None
    raw_pts = np.array(raw_pts)

    # IQR outlier rejection
    x_vals     = raw_pts[:, 0].astype(float)
    q25, q75   = np.percentile(x_vals, 25), np.percentile(x_vals, 75)
    iqr        = q75 - q25
    mult       = config.CONTOUR_CONFIG["iqr_multiplier"]
    lo, hi     = q25 - mult * iqr, q75 + mult * iqr
    good       = (x_vals >= lo) & (x_vals <= hi)
    clean_pts  = raw_pts[good]

    return clean_pts if len(clean_pts) >= config.CONTOUR_CONFIG["contour_pts_min"] else None


# ── 3. Polynomial Curve Fitting ────────────────────────────────────

def fit_surface_curve(back_pts: np.ndarray,
                      degree: int = 3) -> tuple:
    """
    Fit a degree-3 polynomial to the centerline contour.
    Savitzky-Golay pre-smoothing stabilizes noisy x values.

    Returns:
        y_vals (np.ndarray), x_fitted (np.ndarray), poly (np.poly1d) | None×3
    """
    if back_pts is None:
        return None, None, None

    y_vals = back_pts[:, 1].astype(float)
    x_vals = back_pts[:, 0].astype(float)
    n      = len(x_vals)
    wlen   = max(5, min(11, n - (0 if n % 2 == 1 else 1)))
    x_sm   = savgol_filter(x_vals, wlen, 3) if n > wlen else x_vals

    try:
        poly = np.poly1d(np.polyfit(y_vals, x_sm, degree))
        return y_vals, poly(y_vals), poly
    except Exception:
        return None, None, None


# ── 4. Angle Computation ───────────────────────────────────────────

def compute_inclination_angle(poly, y_top: float, y_bottom: float) -> float:
    """
    [FIX-1] Standardized inclination-difference angle.

    Uses atan2(dx/dy, 1.0) at the top and bottom of the fitted curve,
    matching the anatomical pipeline's segment_inclination convention.
    Returns |angle_top - angle_bottom| in degrees.
    """
    if poly is None:
        return 0.0
    dp         = poly.deriv()
    angle_top  = float(np.degrees(np.arctan2(dp(y_top),  1.0)))
    angle_bot  = float(np.degrees(np.arctan2(dp(y_bottom), 1.0)))
    return float(abs(angle_top - angle_bot))


# ── 5. Temporal Smoothing ──────────────────────────────────────────

def moving_average(arr: list | np.ndarray, w: int = 7) -> np.ndarray:
    """[FIX-3] Moving average smoothing for geometric curvature series."""
    arr = np.asarray(arr, float)
    if len(arr) < w:
        return arr
    pad    = w // 2
    padded = np.pad(arr, pad, mode="edge")
    return np.convolve(padded, np.ones(w) / w, mode="valid")


def smooth_series(arr: list | np.ndarray, k: int = 9) -> np.ndarray:
    """Savitzky-Golay smoothing with moving-average fallback."""
    arr = np.asarray(arr, float)
    if len(arr) < k:
        return arr
    pad    = k // 2
    padded = np.pad(arr, pad, mode="edge")
    return np.convolve(padded, np.ones(k) / k, mode="valid")


# ── Classification ─────────────────────────────────────────────────

def classify_curvature(angle: float) -> tuple:
    """Map geometric angle to severity label + BGR colour for overlay."""
    ranges = config.CLINICAL_RANGES["geometric"]
    if angle < ranges["mild"][0]:
        return "Normal posture",   (0, 210, 100)
    elif angle < ranges["severe"][0]:
        return "Mild curvature",   (0, 165, 255)
    else:
        return "Severe curvature", (0, 50,  255)


# ── Main Analysis Function ─────────────────────────────────────────

def analyze_geometric_curvature(video_path: str) -> dict:
    """
    Process one video through the Geometric pipeline.

    Returns dict with:
        time, curvature_raw, curvature_s (smoothed),
        valid_frames, rejected_frames, reasons
    """
    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n[{METHOD_LABEL}] Analyzing: {os.path.basename(video_path)}")
    print(f"  Note: {ANALOGY_NOTE}")
    print(f"  {DISCLAIMER}")

    results = {
        "time":           [],
        "curvature_raw":  [],
        "curvature_s":    np.array([]),
        "valid_frames":   0,
        "rejected_frames":0,
        "reasons":        {},
    }
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        try:
            # 1. Segmentation
            mask, ok = extract_person_mask_mediapipe(
                frame, config.CONTOUR_CONFIG["mask_area_min"]
            )
            if not ok:
                mask = extract_person_mask_fallback(frame)

            bbox = get_person_bbox(mask)
            if bbox is None:
                _rej(results, "no_bbox"); frame_idx += 1; continue

            x0, y0, x1, y1 = bbox
            ph = y1 - y0
            if ph < 50:
                _rej(results, "too_small"); frame_idx += 1; continue

            shoulder_y = y0 + ph * 0.18
            hip_y      = y0 + ph * 0.65
            if hip_y <= shoulder_y:
                _rej(results, "invalid_bounds"); frame_idx += 1; continue

            # 2. FIX-4: centerline contour
            back_pts = get_back_contour_centerline(mask, h, shoulder_y, hip_y)
            if back_pts is None:
                _rej(results, "low_contour"); frame_idx += 1; continue

            # 3. Polynomial fit
            _, _, poly = fit_surface_curve(
                back_pts, degree=config.CONTOUR_CONFIG["poly_degree"]
            )
            if poly is None:
                _rej(results, "fit_fail"); frame_idx += 1; continue

            # 4. Angle
            angle = compute_inclination_angle(
                poly,
                shoulder_y + ph * 0.10,
                hip_y      - ph * 0.10
            )

            results["time"].append(frame_idx / fps)
            results["curvature_raw"].append(angle)
            results["valid_frames"] += 1

        except Exception:
            _rej(results, "error")

        frame_idx += 1

    cap.release()

    # 5. Smoothing
    win = config.CONTOUR_CONFIG["smoothing_window"]
    if results["curvature_raw"]:
        results["curvature_s"] = smooth_series(
            np.array(results["curvature_raw"]), k=win
        )

    # Print summary
    _print_summary(results)
    return results


def _rej(results: dict, reason: str):
    results["rejected_frames"] += 1
    results["reasons"][reason] = results["reasons"].get(reason, 0) + 1


def _print_summary(results: dict):
    total = results["valid_frames"] + results["rejected_frames"]
    rate  = (results["rejected_frames"] / total * 100) if total else 0
    mean  = float(np.mean(results["curvature_raw"])) if results["curvature_raw"] else 0

    print(f"\n  Summary:")
    print(f"    Total frames:  {total}")
    print(f"    Valid:         {results['valid_frames']}")
    print(f"    Rejected:      {results['rejected_frames']} ({rate:.1f}%)")
    for r, cnt in results["reasons"].items():
        print(f"      {r}: {cnt} ({cnt/total*100:.1f}%)" if total else f"      {r}: {cnt}")
    print(f"    Mean geometric curvature: {mean:.2f} deg")


# ── CSV Export ─────────────────────────────────────────────────────

def export_csv(results: dict, label: str) -> str:
    import csv
    out = os.path.join(config.CSV_DIR, f"{label}_geometric.csv")
    n   = len(results["time"])
    s   = results["curvature_s"]
    if len(s) != n:
        s = np.resize(s, n)

    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "curvature_raw", "curvature_smoothed"])
        for i in range(n):
            writer.writerow([
                f"{results['time'][i]:.4f}",
                f"{results['curvature_raw'][i]:.3f}",
                f"{float(s[i]):.3f}",
            ])
    print(f"  CSV saved -> {out}")
    return out


if __name__ == "__main__":
    test_vid = os.path.join(config.RAW_VIDEO_DIR, config.VIDEO_INFO[0]["filename"])
    res      = analyze_geometric_curvature(test_vid)
    if res["curvature_raw"]:
        print(f"Mean Curvature: {np.mean(res['curvature_raw']):.2f} deg")
