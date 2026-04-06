import os
import cv2
import numpy as np
import mediapipe as mp
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
import config

warnings.filterwarnings('ignore')

# ── Pipeline Flow ──────────────────────────────────────────────────
# 1. Segmentation (MediaPipe Selfie Segmentation with GrabCut fallback)
# 2. Keypoint/Contour Extraction (Shoulder to Hip level)
# 3. Filtering (IQR-based outlier rejection)
# 4. Angle Computation (Standardized inclination-based approach)
# 5. Smoothing (Temporal moving average)
# 6. Comparison (Anatomical vs Geometric - handled in unified pipeline)

# ── Research Transparency ──────────────────────────────────────────
DISCLAIMER = (
    "All measurements are derived from 2D projections and are sensitive to camera alignment, "
    "perspective distortion, and subject orientation. The method assumes sagittal-plane motion "
    "and does not account for 3D spinal rotation."
)
METHOD_LABEL = "Geometric Method"
ANALOGY_NOTE = "Geometric curvature (not anatomically equivalent to Cobb angle)"

def extract_person_mask_mediapipe(frame, mask_area_min=500):
    """
    Extracts person silhouette using MediaPipe Selfie Segmentation.
    Returns:
        mask (np.ndarray): binary mask (255=person)
        success (bool): whether MediaPipe gave a high-quality mask
    """
    with mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1) as selfie:
        results = selfie.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        condition = results.segmentation_mask > 0.5
        mask = np.where(condition, 255, 0).astype(np.uint8)
        
        # Validation: check for minimum area and connectivity
        area = np.sum(mask > 0)
        if area < mask_area_min:
            return None, False
            
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, False
            
        # Refine mask to largest connected region
        clean = np.zeros_like(mask)
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(clean, [largest], -1, 255, -1)
        return clean, True

def extract_person_mask_fallback(frame):
    """Fallback to simple thresholding/GrabCut if MediaPipe fails."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY) # dark bg threshold
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean = np.zeros_like(mask)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(clean, [largest], -1, 255, -1)
    return clean

def get_person_bbox(mask):
    cols = np.where(mask.any(axis=0))[0]
    rows = np.where(mask.any(axis=1))[0]
    if not len(cols) or not len(rows):
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])

def get_back_contour(mask, H, W, y_start, y_end):
    """Extracts the posterior (back) edge of the person silhouette."""
    upper_rows = range(max(0, int(y_start)), min(H, int(y_start + (y_end - y_start) * 0.4)))
    upper_x = []
    for r in upper_rows:
        cs = np.where(mask[r, :] > 0)[0]
        if len(cs): upper_x.extend(cs.tolist())
    
    centroid_x = float(np.mean(upper_x)) if upper_x else W / 2
    use_left_edge = centroid_x < W * 0.5

    raw_pts = []
    for y in range(max(0, int(y_start)), min(H, int(y_end))):
        cs = np.where(mask[y, :] > 0)[0]
        if len(cs) >= 2:
            bx = int(cs[0]) if use_left_edge else int(cs[-1])
            raw_pts.append((bx, y))

    if len(raw_pts) < config.CONTOUR_CONFIG["contour_pts_min"]: return None
    raw_pts = np.array(raw_pts)

    # IQR Outlier rejection
    x_vals = raw_pts[:, 0].astype(float)
    q25, q75 = np.percentile(x_vals, 25), np.percentile(x_vals, 75)
    iqr = q75 - q25
    lo, hi = q25 - 1.5 * iqr, q75 + 1.5 * iqr
    good = (x_vals >= lo) & (x_vals <= hi)
    clean_pts = raw_pts[good]

    return clean_pts if len(clean_pts) >= config.CONTOUR_CONFIG["contour_pts_min"] else None

def fit_surface_curve(back_pts, degree=3):
    """Fits a polynomial to the back contour (degree 3 avoids overfitting)."""
    if back_pts is None: return None, None, None
    y_vals, x_vals = back_pts[:, 1].astype(float), back_pts[:, 0].astype(float)
    n = len(x_vals)
    wlen = max(5, min(11, n - (0 if n % 2 == 1 else 1)))
    x_sm = savgol_filter(x_vals, wlen, 3) if n > wlen else x_vals

    try:
        poly = np.poly1d(np.polyfit(y_vals, x_sm, degree))
        return y_vals, poly(y_vals), poly
    except: return None, None, None

def compute_inclination_angle(poly, y_top, y_bottom):
    """Standardized angle calculation: abs(atan2(dx, abs(dy)))."""
    if poly is None: return 0.0
    dp = poly.deriv()
    dy = 1.0 # step in y
    dx_top = float(dp(y_top))
    dx_bot = float(dp(y_bottom))
    
    # Standardized atan2 based inclination (analogous to vector)
    angle_t = np.degrees(np.arctan2(dx_top, abs(dy)))
    angle_b = np.degrees(np.arctan2(dx_bot, abs(dy)))
    return float(abs(angle_t - angle_b))

def smooth_series(arr, k=9):
    if len(arr) < k: return np.array(arr)
    return np.convolve(arr, np.ones(k)/k, mode='same')

def classify_curvature(angle):
    if angle < 20: return 'Normal posture', (0, 210, 100)
    elif angle < 40: return 'Mild curvature', (0, 165, 255)
    else: return 'Severe curvature', (0, 50, 255)

def analyze_geometric_curvature(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n[{METHOD_LABEL}] Analyzing: {os.path.basename(video_path)}")
    print(DISCLAIMER)
    
    results = {"time":[], "curvature_raw":[], "valid_frames": 0, "rejected_frames": 0, "reasons":{}}
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        try:
            # 1. Segmentation
            mask, success = extract_person_mask_mediapipe(frame, config.CONTOUR_CONFIG["mask_area_min"])
            if not success:
                mask = extract_person_mask_fallback(frame)
                
            bbox = get_person_bbox(mask)
            if bbox is None:
                results["rejected_frames"] += 1
                results["reasons"]["no_bbox"] = results["reasons"].get("no_bbox", 0) + 1
                frame_idx += 1
                continue
                
            x0, y0, x1, y1 = bbox
            ph = y1 - y0
            shoulder_y = y0 + ph * 0.18
            hip_y = y0 + ph * 0.65
            
            # 2. Extract back contour
            back_pts = get_back_contour(mask, h, w, shoulder_y, hip_y)
            if back_pts is None:
                results["rejected_frames"] += 1
                results["reasons"]["low_contour"] = results["reasons"].get("low_contour", 0) + 1
                frame_idx += 1
                continue
                
            # 3. Fit curve
            _, _, poly = fit_surface_curve(back_pts, degree=config.CONTOUR_CONFIG["poly_degree"])
            if poly is None:
                results["rejected_frames"] += 1
                results["reasons"]["fit_fail"] = results["reasons"].get("fit_fail", 0) + 1
                frame_idx += 1
                continue
                
            # 4. Compute angle
            angle = compute_inclination_angle(poly, shoulder_y+ph*0.1, hip_y-ph*0.1)
            results["time"].append(frame_idx/fps)
            results["curvature_raw"].append(angle)
            results["valid_frames"] += 1
            
        except Exception as e:
            results["rejected_frames"] += 1
            results["reasons"]["error"] = results["reasons"].get("error", 0) + 1
            
        frame_idx += 1
    
    cap.release()
    
    # Post-processing
    curv = np.array(results["curvature_raw"])
    results["curvature_s"] = smooth_series(curv, config.CONTOUR_CONFIG["smoothing_window"])
    
    # ── Final Rejection Stats ──────────────────────────────────────
    total_f = results["valid_frames"] + results["rejected_frames"]
    rate = (results["rejected_frames"] / total_f * 100) if total_f > 0 else 0
    print(f"Total frames: {total_f}")
    print(f"Valid: {results['valid_frames']}")
    print(f"Rejected: {results['rejected_frames']} ({rate:.1f}%)")
    for r, count in results["reasons"].items():
        print(f"  - {r}: {count} ({count/total_f*100:.1f}%)")
        
    return results

if __name__ == "__main__":
    test_vid = os.path.join(config.RAW_VIDEO_DIR, config.VIDEO_INFO[0]["filename"])
    res = analyze_geometric_curvature(test_vid)
    print(f"Mean Curvature: {np.mean(res['curvature_raw']):.2f} deg")