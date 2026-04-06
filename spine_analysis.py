"""
Unified Spinal Posture Analysis Pipeline (Anatomical Method)
===========================================================
Combines SpinePose (CVPR 2025) keypoint detection with multi-segment
Cobb angle estimation for scientifically accurate spinal curvature measurement.
"""
import os, sys, math, warnings
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.signal import savgol_filter
from tqdm import tqdm
import csv
import config

warnings.filterwarnings("ignore")

# ── Research Transparency ──────────────────────────────────────────
DISCLAIMER = (
    "All measurements are derived from 2D projections and are sensitive to camera alignment, "
    "perspective distortion, and subject orientation. The method assumes sagittal-plane motion "
    "and does not account for 3D spinal rotation."
)
METHOD_LABEL = "Anatomical Method (SpinePose)"
VALIDATION_PHILOSOPHY = (
    "Due to the absence of radiographic ground truth, validation is relative and based on "
    "biomechanical consistency and literature ranges."
)

# ── SpinePose 37-keypoint model constants ──────────────────────────
SPINE_IDX = {
    "C1":36, "C4":35, "C7":18, "T3":30, "T8":29,
    "L1":28, "L3":27, "L5":26, "Sacrum":19,
}
SPINE_CHAIN = ["C1","C4","C7","T3","T8","L1","L3","L5","Sacrum"]

def _segment_inclination(pt_top, pt_bot):
    """Standardized segment inclination: abs(atan2(dx, abs(dy)))."""
    v = np.array(pt_bot, dtype=float) - np.array(pt_top, dtype=float)
    if abs(v[1]) < 1e-9: return 0.0
    return abs(math.degrees(math.atan2(v[0], abs(v[1]))))

def multi_segment_cobb(kpts, landmark_names):
    """Multi-segment Cobb angle (max inclination - min inclination)."""
    if len(landmark_names) < 2: return 0.0
    pts = [kpts[SPINE_IDX[n]] for n in landmark_names]
    inclinations = [_segment_inclination(pts[i], pts[i+1]) for i in range(len(pts)-1)]
    if not inclinations: return 0.0
    return float(np.max(inclinations) - np.min(inclinations))

def trunk_forward_lean(c7, sacrum):
    return _segment_inclination(c7, sacrum)

def is_valid_frame(kpts, conf):
    """Anatomical validity check with rejection reason tracking."""
    key_names = ["C1","C4","C7","T3","T8","L1","L3","L5","Sacrum"]
    key_idx = [SPINE_IDX[k] for k in key_names]
    
    # Check 1: Confidence
    mean_c = np.mean([conf[i] for i in key_idx])
    if mean_c < config.SPINEPOSE_CONFIG["conf_thresh"]:
        return False, "low_confidence", mean_c
        
    # Check 2: Top-to-bottom ordering
    pts = [kpts[SPINE_IDX[k]] for k in key_names]
    for i in range(len(pts) - 1):
        if pts[i][1] >= pts[i + 1][1]:
            return False, "ordering_violation", mean_c
            
    return True, "valid", mean_c

def smooth(arr, k=9):
    arr = np.array(arr)
    if len(arr) < k: return arr
    window = k if k % 2 != 0 else k - 1
    try: return savgol_filter(arr, window, 2, mode="interp")
    except: return np.convolve(arr, np.ones(k)/k, mode='same')

def analyze_video_anatomical(vid_info, estimator):
    display, filename = vid_info["display_name"], vid_info["filename"]
    vid_path = os.path.join(config.RAW_VIDEO_DIR, filename)
    
    cap = cv2.VideoCapture(vid_path)
    fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\n[{METHOD_LABEL}] Analyzing: {display}")
    print(DISCLAIMER)
    
    R = {"time":[], "kyphosis":[], "tfl":[], "frame_id":[], "confidence":[]}
    stats = {"total": 0, "valid": 0, "rejected": 0, "reasons":{}}
    
    pbar = tqdm(total=total, desc=f"[{display}]")
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        stats["total"] += 1
        try:
            bboxes = estimator.detect(frame)
            if bboxes is not None and len(bboxes) > 0:
                fr = estimator.estimate(frame, [bboxes[0]])
                if fr and len(fr) > 0:
                    fd = np.array(fr[0])[0] if np.array(fr[0]).ndim == 3 else np.array(fr[0])
                    kpts = fd[:, :2]
                    conf = fd[:, 2]
                    
                    valid, reason, mean_c = is_valid_frame(kpts, conf)
                    if valid:
                        R["time"].append(frame_idx/fps)
                        R["frame_id"].append(frame_idx)
                        R["kyphosis"].append(multi_segment_cobb(kpts, ["C7","T3","T8","L1"]))
                        R["tfl"].append(trunk_forward_lean(kpts[SPINE_IDX["C7"]], kpts[SPINE_IDX["Sacrum"]]))
                        R["confidence"].append(mean_c)
                        stats["valid"] += 1
                    else:
                        stats["rejected"] += 1
                        stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
                else:
                    stats["rejected"] += 1
                    stats["reasons"]["no_estimation"] = stats["reasons"].get("no_estimation", 0) + 1
            else:
                stats["rejected"] += 1
                stats["reasons"]["no_detection"] = stats["reasons"].get("no_detection", 0) + 1
        except Exception as e:
            stats["rejected"] += 1
            stats["reasons"]["error"] = stats["reasons"].get("error", 0) + 1
            
        frame_idx += 1
        pbar.update(1)
        
    cap.release()
    pbar.close()
    
    # ── Rejection Analytics ────────────────────────────────────────
    rate = (stats["rejected"] / stats["total"] * 100) if stats["total"] > 0 else 0
    print(f"\nRejection Summary for {display}:")
    print(f"  Total Frames: {stats['total']}")
    print(f"  Valid:        {stats['valid']}")
    print(f"  Rejected:     {stats['rejected']} ({rate:.1f}%)")
    for r, count in stats["reasons"].items():
        print(f"    - {r}: {count} ({count/stats['total']*100:.1f}%)")
        
    # ── Confidence Awareness ───────────────────────────────────────
    mean_conf = np.mean(R["confidence"]) if R["confidence"] else 0
    print(f"  Average keypoint confidence: {mean_conf:.3f}")
    if mean_conf < config.RELIABILITY_CONF_THRESH:
        print("  WARNING: Mean confidence below 0.5 may indicate unreliable detection.")
        
    return R

if __name__ == "__main__":
    print(f"Reproducibility Note: {config.REPRODUCIBILITY_NOTE}")
