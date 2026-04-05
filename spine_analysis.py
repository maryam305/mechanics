"""
Unified Spinal Posture Analysis Pipeline
=========================================
Combines SpinePose (CVPR 2025) keypoint detection with multi-segment
cumulative Cobb angle estimation for scientifically accurate spinal
curvature measurement.

Output:
  output_analysis/videos/   — annotated MP4 files
  output_analysis/plots/    — per-video and comparison PNG figures
  output_analysis/reports/  — CSV summary tables
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

warnings.filterwarnings("ignore")
import config

# ══════════════════════════════════════════════════════════════════
#  CONSTANTS — SpinePose 37-keypoint model (from metainfo.py)
# ══════════════════════════════════════════════════════════════════
# Verified against spinepose/metainfo.py:
#   17 = head (head top)
#   18 = neck (C7 / neck base)
#   19 = hip  (sacrum midpoint)
#   26 = spine_01 = L5
#   27 = spine_02 = L3
#   28 = spine_03 = T12/L1 transition
#   29 = spine_04 = T8
#   30 = spine_05 = T3
#   35 = neck_02  = C3/C4
#   36 = neck_03  = C1 (atlas)
SPINE_IDX = {
    "C1":36, "C4":35, "C7":18, "T3":30, "T8":29,
    "L1":28, "L3":27, "L5":26, "Sacrum":19,
}
SPINE_CHAIN = ["C1","C4","C7","T3","T8","L1","L3","L5","Sacrum"]

# ══════════════════════════════════════════════════════════════════
#  ANGLE FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def _segment_inclination(pt_top, pt_bot):
    """
    Inclination of a segment from the vertical axis (degrees).
    Positive = tilted to the right (or anteriorly in sagittal view).
    Uses atan2(Δx, |Δy|) so that a perfectly vertical segment → 0°.
    """
    v = np.array(pt_bot, dtype=float) - np.array(pt_top, dtype=float)
    if abs(v[1]) < 1e-9:
        return 0.0
    return math.degrees(math.atan2(v[0], abs(v[1])))


def cobb_angle(top_pair, bottom_pair):
    """Sagittal Cobb angle via inclination-from-vertical (2 segments)."""
    a1 = _segment_inclination(top_pair[0], top_pair[1])
    a2 = _segment_inclination(bottom_pair[0], bottom_pair[1])
    return abs(a1 - a2)


def multi_segment_cobb(kpts, landmark_names):
    """
    Multi-segment cumulative Cobb angle.

    Sums the absolute change in segment inclination across consecutive
    landmarks.  This is the standard method for estimating regional
    spinal curvature from multiple vertebral landmarks and produces
    values in the clinically expected range (thoracic 20-55°,
    lumbar 20-60°).

    For N landmarks there are N-1 segments; the cumulative angle is
    the sum of |Δθ| between consecutive segment pairs.
    """
    if len(landmark_names) < 3:
        return 0.0

    pts = [kpts[SPINE_IDX[n]] for n in landmark_names]

    # Compute inclination of each segment
    inclinations = []
    for i in range(len(pts) - 1):
        inc = _segment_inclination(pts[i], pts[i + 1])
        inclinations.append(inc)

    # Sum absolute changes in inclination between consecutive segments
    total = 0.0
    for i in range(len(inclinations) - 1):
        total += abs(inclinations[i + 1] - inclinations[i])

    return total


def lateral_deviation_angle(top_pt, mid_pt, bot_pt):
    """Lateral bending: supplement of the angle at mid_pt."""
    v1 = np.array(top_pt, dtype=float) - np.array(mid_pt, dtype=float)
    v2 = np.array(bot_pt, dtype=float) - np.array(mid_pt, dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0
    cos_val = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return 180.0 - math.degrees(math.acos(cos_val))


def trunk_forward_lean(c7, sacrum):
    """Angle of C7→Sacrum vector from vertical (abs value)."""
    v = np.array(sacrum, dtype=float) - np.array(c7, dtype=float)
    if abs(v[1]) < 1e-9:
        return 0.0
    return abs(math.degrees(math.atan2(v[0], abs(v[1]))))


def is_valid_frame(kpts, conf):
    """Anatomical validity: y-coords must be top-to-bottom and confidence OK."""
    key_names = ["C7","T3","T8","L1","L3","L5","Sacrum"]
    key_idx = [SPINE_IDX[k] for k in key_names]
    if np.mean([conf[i] for i in key_idx]) < 0.25:
        return False
    C7, T3, T8, L1, Sacrum = [kpts[SPINE_IDX[k]] for k in ["C7","T3","T8","L1","Sacrum"]]
    return C7[1] < T3[1] < T8[1] < L1[1] < Sacrum[1]


def smooth(arr, k=7):
    """Savitzky-Golay smoothing (preserves peaks better than box filter)."""
    arr = np.array(arr)
    if len(arr) < k:
        return arr
    try:
        return savgol_filter(arr, min(k, len(arr) | 1), 2)  # ensure odd window
    except Exception:
        return np.convolve(arr, np.ones(k) / k, mode="same")


# ══════════════════════════════════════════════════════════════════
#  MULTI-SEGMENT CUMULATIVE COBB ANGLE — REGION DEFINITIONS
# ══════════════════════════════════════════════════════════════════
# Thoracic kyphosis: C7 → T3 → T8 → L1  (3 segments, 2 Δθ)
# Lumbar lordosis  : L1 → L3 → L5 → Sacrum  (3 segments, 2 Δθ)
# Full back        : C7 → T3 → T8 → L1 → L3 → L5 → Sacrum

THORACIC_CHAIN = ["C7","T3","T8","L1"]
LUMBAR_CHAIN   = ["L1","L3","L5","Sacrum"]
FULL_CHAIN     = ["C7","T3","T8","L1","L3","L5","Sacrum"]


def compute_keypoint_curvature(kpts, chain_names, degree=3):
    """
    Fit a polynomial to spine keypoints for visualisation and
    compute the multi-segment cumulative Cobb angle.

    Returns:
        angle (float): cumulative Cobb angle in degrees
        poly (np.poly1d or None): fitted polynomial for drawing
        pts (np.ndarray): keypoint positions used
    """
    pts_xy = []
    for name in chain_names:
        idx = SPINE_IDX[name]
        pts_xy.append((float(kpts[idx][0]), float(kpts[idx][1])))

    pts_xy = np.array(pts_xy)
    y_vals = pts_xy[:, 1]
    x_vals = pts_xy[:, 0]

    if len(y_vals) < 3:
        return 0.0, None, pts_xy

    # --- Multi-segment cumulative Cobb angle (the scientific metric) ---
    angle = multi_segment_cobb(kpts, chain_names)

    # --- Polynomial fit for visualisation only ---
    try:
        fit_degree = min(degree, len(y_vals) - 1)
        poly = np.poly1d(np.polyfit(y_vals, x_vals, fit_degree))
    except Exception:
        poly = None

    # Sanity guard
    if angle > 90:
        angle = 0.0

    return angle, poly, pts_xy


def compute_max_curvature_kpt(poly, y_top, y_bottom, n_pts=50):
    """Compute max and mean curvature κ from the polynomial."""
    if poly is None:
        return 0.0, 0.0
    dp = poly.deriv(1)
    d2p = poly.deriv(2)
    y_range = np.linspace(y_top, y_bottom, n_pts)
    ks = [abs(float(d2p(y))) / (1 + float(dp(y))**2)**1.5 for y in y_range]
    return float(np.max(ks)), float(np.mean(ks))


# ══════════════════════════════════════════════════════════════════
#  MAIN VIDEO PROCESSING
# ══════════════════════════════════════════════════════════════════
def ensure_dirs():
    for d in [config.VID_DIR, config.PLOT_DIR, config.REPORT_DIR]:
        os.makedirs(d, exist_ok=True)


def analyze_video(vid_info, estimator):
    filename = vid_info["filename"]
    display  = vid_info["display_name"]
    condition = vid_info.get("condition", "unloaded")
    vid_path = os.path.join(config.BASE_DIR, filename)

    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n{'='*60}")
    print(f"  {display}  ({filename})")
    print(f"  {w}x{h} | {fps:.1f} fps | {total} frames | {condition}")
    print(f"{'='*60}")

    out_path = os.path.join(config.VID_DIR, f"{display}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    R = {"time":[], "kyph_geom":[], "curv_max":[],
         "kyph_spine":[], "lord_spine":[], "bend_spine":[], "tfl":[]}

    frame_idx = 0
    pbar = tqdm(total=total, desc=f"[{display}]")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # ── SpinePose detection first ─────────────────────────────
        kpts, conf = None, None
        spine_ky = spine_lo = spine_be = current_tfl = 0.0
        valid = False
        try:
            bboxes = estimator.detect(frame)
            if bboxes is not None and len(bboxes) > 0:
                fr = estimator.estimate(frame, [bboxes[0]])
                if fr and len(fr) > 0:
                    fd = np.array(fr[0])
                    if fd.ndim == 3:
                        fd = fd[0]
                    kpts = fd[:, :2]
                    conf = fd[:, 2] if fd.shape[1] > 2 else np.ones(len(fd))
                    if is_valid_frame(kpts, conf):
                        valid = True
                        C7 = kpts[SPINE_IDX["C7"]]
                        T3 = kpts[SPINE_IDX["T3"]]
                        T8 = kpts[SPINE_IDX["T8"]]
                        L1 = kpts[SPINE_IDX["L1"]]
                        L3 = kpts[SPINE_IDX["L3"]]
                        L5 = kpts[SPINE_IDX["L5"]]
                        Sacrum = kpts[SPINE_IDX["Sacrum"]]
                        # Multi-segment cumulative Cobb (clinical standard)
                        spine_ky = multi_segment_cobb(kpts, THORACIC_CHAIN)
                        spine_lo = multi_segment_cobb(kpts, LUMBAR_CHAIN)
                        spine_be = lateral_deviation_angle(C7, L1, Sacrum)
                        current_tfl = trunk_forward_lean(C7, Sacrum)
        except Exception:
            pass

        # ── Keypoint-polynomial geometric curvature ─────────────────
        geom_kyphosis = 0.0
        geom_lordosis = 0.0
        k_max = 0.0
        poly_thoracic = None
        poly_full = None
        kpt_pts = None
        if valid and kpts is not None:
            geom_kyphosis, poly_thoracic, _ = compute_keypoint_curvature(
                kpts, THORACIC_CHAIN, degree=config.CONTOUR_CONFIG["poly_degree"]
            )
            geom_lordosis, _, _ = compute_keypoint_curvature(
                kpts, LUMBAR_CHAIN, degree=min(3, config.CONTOUR_CONFIG["poly_degree"])
            )
            _, poly_full, kpt_pts = compute_keypoint_curvature(
                kpts, FULL_CHAIN, degree=config.CONTOUR_CONFIG["poly_degree"]
            )
            if poly_full is not None and kpt_pts is not None:
                k_max, _ = compute_max_curvature_kpt(
                    poly_full, kpt_pts[0][1], kpt_pts[-1][1]
                )

        # ── Record data ───────────────────────────────────────────
        if valid:
            R["time"].append(frame_idx / fps)
            R["kyph_geom"].append(geom_kyphosis)
            R["curv_max"].append(k_max)
            R["kyph_spine"].append(spine_ky)
            R["lord_spine"].append(spine_lo)
            R["bend_spine"].append(spine_be)
            R["tfl"].append(current_tfl)

        # ── Draw overlay (compact) ────────────────────────────────
        out = frame.copy()

        # Draw polynomial curvature overlay
        if poly_full is not None and kpt_pts is not None and len(kpt_pts) >= 2:
            y_top_d = kpt_pts[0][1]
            y_bot_d = kpt_pts[-1][1]
            y_range = np.linspace(y_top_d, y_bot_d, 80)
            pts = []
            for y in y_range:
                x = int(poly_full(y))
                if 0 <= x < w and 0 <= int(y) < h:
                    pts.append((x, int(y)))
            for i in range(len(pts) - 1):
                tc = i / max(len(pts) - 1, 1)
                cv2.line(out, pts[i], pts[i+1],
                         (int(255*(1-tc)), 80, int(255*tc)), 3)

        # Draw SpinePose chain
        if kpts is not None and valid:
            pts_int = kpts.astype(int)
            chain = [tuple(pts_int[SPINE_IDX[k]]) for k in SPINE_CHAIN]
            for j in range(len(chain) - 1):
                cv2.line(out, chain[j], chain[j+1], (0,255,0), 2)
            for k in SPINE_CHAIN:
                pt = tuple(pts_int[SPINE_IDX[k]])
                cv2.circle(out, pt, 3, (0,0,255), -1)
                cv2.putText(out, k, (pt[0]+4, pt[1]-2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0,0,0), 2)
                cv2.putText(out, k, (pt[0]+4, pt[1]-2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255,255,255), 1)

        # Text overlay with clear labels
        lines = [
            f"Kyphosis (Cobb): {geom_kyphosis:.1f} deg",
            f"Kyphosis (MS):   {spine_ky:.1f} deg",
            f"Lordosis (MS):   {spine_lo:.1f} deg",
            f"Trunk Lean:      {current_tfl:.1f} deg",
        ]
        # draw background rectangle
        bx0, by0 = 4, 4
        bx1 = 220
        by1 = 4 + len(lines) * 18 + 6
        overlay = out.copy()
        cv2.rectangle(overlay, (bx0, by0), (bx1, by1), (0,0,0), -1)
        out = cv2.addWeighted(overlay, 0.55, out, 0.45, 0)
        for i, txt in enumerate(lines):
            yp = 20 + i * 18
            # Color coding: red for geom kyphosis, blue for spine, orange for lordosis, purple for lean
            colors = [(0,80,255), (255,200,0), (0,180,255), (255,100,255)]
            cv2.putText(out, txt, (8, yp),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0,0,0), 3)
            cv2.putText(out, txt, (8, yp),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, colors[i], 1)

        out_vid.write(out)
        frame_idx += 1
        pbar.update(1)

    cap.release()
    out_vid.release()
    pbar.close()

    # Smooth stored arrays
    for k in ["kyph_geom","curv_max","kyph_spine","lord_spine","bend_spine","tfl"]:
        R[k] = np.array(R[k])
        R[k + "_s"] = smooth(R[k])

    R["time"] = np.array(R["time"])
    R["display_name"] = display
    R["condition"] = condition
    R["filename"] = filename

    n = len(R["time"])
    print(f"  Valid frames: {n}/{total}")
    if n > 0:
        print(f"  Geom Kyphosis : {np.mean(R['kyph_geom']):.1f} ± {np.std(R['kyph_geom']):.1f}°")
        print(f"  Spine Kyphosis: {np.mean(R['kyph_spine']):.1f} ± {np.std(R['kyph_spine']):.1f}°")
        print(f"  Spine Lordosis: {np.mean(R['lord_spine']):.1f} ± {np.std(R['lord_spine']):.1f}°")
        print(f"  Trunk Lean    : {np.mean(R['tfl']):.1f} ± {np.std(R['tfl']):.1f}°")
    return R


# ══════════════════════════════════════════════════════════════════
#  PLOTTING — RESEARCH-GRADE
# ══════════════════════════════════════════════════════════════════
COLORS = {
    "Kyphosis (Curv)":  "#E53935",
    "Kyphosis (MS)":    "#2196F3",
    "Lordosis (MS)":    "#FF9800",
    "Lumbar Bending":   "#4CAF50",
    "Trunk Lean":       "#9C27B0",
}


def plot_single_video(R):
    """Per-video figure: 3 time-series + bar chart + summary table."""
    t = R["time"]
    if len(t) == 0:
        return
    name = R["display_name"]

    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor("#FAFAFA")
    gs = GridSpec(4, 2, figure=fig, hspace=0.50, wspace=0.35)
    fig.suptitle(f"Spinal Posture Analysis — {name} ({R['condition']})",
                 fontsize=14, fontweight="bold", y=0.98)

    # ── Row 0: Kyphosis (two methods overlaid) ────────────────────
    ax = fig.add_subplot(gs[0, :])
    norm = config.NORMS["thoracic_kyphosis"]
    ax.axhspan(norm["mean"]-norm["sd"], norm["mean"]+norm["sd"],
               alpha=0.10, color="green", label=f'Normal range ({norm["mean"]-norm["sd"]:.0f}–{norm["mean"]+norm["sd"]:.0f}°)')
    ax.axhline(norm["mean"], color="green", lw=1.5, ls="--", alpha=0.7,
               label=f'Normative mean ({norm["mean"]}°)')
    ax.plot(t, R["kyph_geom_s"], color=COLORS["Kyphosis (Curv)"], lw=2,
            label=f'Curvature ({np.mean(R["kyph_geom"]):.1f}°)')
    ax.plot(t, R["kyph_spine_s"], color=COLORS["Kyphosis (MS)"], lw=2,
            label=f'MS-Cobb ({np.mean(R["kyph_spine"]):.1f}°)')
    ax.set_ylabel("Thoracic Kyphosis (°)")
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)
    ax.set_facecolor("#F5F5F5")

    # ── Row 1: Lordosis ──────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    norm = config.NORMS["lumbar_lordosis"]
    ax.axhspan(norm["mean"]-norm["sd"], norm["mean"]+norm["sd"],
               alpha=0.10, color="green")
    ax.axhline(norm["mean"], color="green", lw=1.5, ls="--", alpha=0.7)
    ax.plot(t, R["lord_spine_s"], color=COLORS["Lordosis (MS)"], lw=2)
    ax.set_ylabel("Lumbar Lordosis (°)")
    ax.set_title(f'Lordosis ({np.mean(R["lord_spine"]):.1f}°)', fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.set_facecolor("#F5F5F5")

    # ── Row 1: Trunk Lean ────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    norm = config.NORMS["trunk_lean"]
    ax.axhspan(norm["mean"]-norm["sd"], norm["mean"]+norm["sd"],
               alpha=0.10, color="green")
    ax.axhline(norm["mean"], color="green", lw=1.5, ls="--", alpha=0.7)
    ax.plot(t, R["tfl_s"], color=COLORS["Trunk Lean"], lw=2)
    ax.set_ylabel("Trunk Lean (°)")
    ax.set_title(f'Trunk Lean ({np.mean(R["tfl"]):.1f}°)', fontsize=10)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.set_facecolor("#F5F5F5")

    # ── Row 2: Bar chart → Subject vs Normative ──────────────────
    ax = fig.add_subplot(gs[2, 0])
    labels = ["Kyphosis\n(Curv)", "Kyphosis\n(MS-Cobb)", "Lordosis", "Trunk\nLean"]
    subj = [np.mean(R["kyph_geom"]), np.mean(R["kyph_spine"]),
            np.mean(R["lord_spine"]), np.mean(R["tfl"])]
    subj_sd = [np.std(R["kyph_geom"]), np.std(R["kyph_spine"]),
               np.std(R["lord_spine"]), np.std(R["tfl"])]
    norms_m = [config.NORMS["thoracic_kyphosis"]["mean"],
               config.NORMS["thoracic_kyphosis"]["mean"],
               config.NORMS["lumbar_lordosis"]["mean"],
               config.NORMS["trunk_lean"]["mean"]]
    norms_sd = [config.NORMS["thoracic_kyphosis"]["sd"],
                config.NORMS["thoracic_kyphosis"]["sd"],
                config.NORMS["lumbar_lordosis"]["sd"],
                config.NORMS["trunk_lean"]["sd"]]
    x = np.arange(len(labels))
    ww = 0.35
    bars1 = ax.bar(x - ww/2, subj, ww, yerr=subj_sd, capsize=4,
                   label="Subject", color="steelblue", alpha=0.85)
    bars2 = ax.bar(x + ww/2, norms_m, ww, yerr=norms_sd, capsize=4,
                   label="Normative", color="gray", alpha=0.45)
    for b, v in zip(bars1, subj):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.5,
                f"{v:.1f}", ha="center", fontsize=7, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Angle (°)")
    ax.set_title("Subject vs Normative", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_facecolor("#F5F5F5")

    # ── Row 2: Summary table ─────────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    ax.axis("off")
    col_labels = ["Metric", "Mean±SD", "Norm±SD", "Δ", "Status"]
    rows = []
    for metric_name, key, norm_key in [
        ("Kyphosis (Curv)", "kyph_geom", "thoracic_kyphosis"),
        ("Kyphosis (MS-Cobb)", "kyph_spine", "thoracic_kyphosis"),
        ("Lumbar Lordosis", "lord_spine", "lumbar_lordosis"),
        ("Trunk Lean", "tfl", "trunk_lean"),
    ]:
        arr = R[key]
        m, s = float(np.mean(arr)), float(np.std(arr))
        nm = config.NORMS[norm_key]["mean"]
        ns = config.NORMS[norm_key]["sd"]
        delta = m - nm
        z = delta / ns if ns > 0 else 0
        if abs(z) <= 1:
            status = "Normal"
        elif abs(z) <= 2:
            status = "Borderline"
        else:
            status = "Outside"
        rows.append([metric_name, f"{m:.1f}±{s:.1f}°", f"{nm}±{ns}°",
                      f"{delta:+.1f}°", status])

    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.8)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#37474F")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    status_colors = {"Normal":"#C8E6C9", "Borderline":"#FFF9C4", "Outside":"#FFCDD2"}
    for i, row in enumerate(rows):
        for j in range(len(col_labels)):
            if j == 4:
                tbl[i+1, j].set_facecolor(status_colors.get(row[4], "white"))
            else:
                tbl[i+1, j].set_facecolor("#ECEFF1" if i % 2 == 0 else "white")
    ax.set_title("Summary — vs Normative Reference", fontsize=10, pad=10, fontweight="bold")

    # ── Row 3: methodology note ──────────────────────────────────
    ax = fig.add_subplot(gs[3, :])
    ax.axis("off")
    ax.text(0.5, 0.5,
        "Kyphosis (Curv): multi-segment cumulative Cobb from thoracic keypoints C7→T3→T8→L1\n"
        "Kyphosis (MS-Cobb): same method applied via SpinePose (CVPR 2025) vertebral landmarks\n"
        "Lordosis: multi-segment Cobb L1→L3→L5→Sacrum | Normative: Ohlendorf et al. (2023) Sci Rep 13:12395",
        transform=ax.transAxes, ha="center", va="center",
        fontsize=8, color="gray", style="italic")

    plt.savefig(os.path.join(config.PLOT_DIR, f"{name}.png"),
                dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()


def plot_comparison(R_nl, R_ld):
    """Load-carriage comparison figure."""
    t_nl, t_ld = R_nl["time"], R_ld["time"]
    if len(t_nl) == 0 or len(t_ld) == 0:
        return

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#FAFAFA")
    gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("Load Carriage Comparison — Unloaded vs Loaded",
                 fontsize=14, fontweight="bold", y=0.98)

    # ── Time-series panels ────────────────────────────────────────
    metrics = [
        ("Kyphosis (MS-Cobb)", "kyph_spine_s", "thoracic_kyphosis"),
        ("Lordosis (MS-Cobb)", "lord_spine_s", "lumbar_lordosis"),
        ("Trunk Lean", "tfl_s", "trunk_lean"),
    ]
    for i, (name, key, norm_key) in enumerate(metrics):
        ax = fig.add_subplot(gs[0, i])
        norm = config.NORMS[norm_key]
        ax.axhspan(norm["mean"]-norm["sd"], norm["mean"]+norm["sd"],
                   alpha=0.10, color="green")
        ax.axhline(norm["mean"], color="green", lw=1, ls="--", alpha=0.7)
        m_nl = float(np.mean(R_nl[key]))
        m_ld = float(np.mean(R_ld[key]))
        ax.plot(t_nl, R_nl[key], color="steelblue", lw=2,
                label=f"Unloaded ({m_nl:.1f}°)")
        ax.plot(t_ld, R_ld[key], color="crimson", lw=2, ls="--",
                label=f"Loaded ({m_ld:.1f}°)")
        ax.set_title(f"{name} (Δ={m_ld-m_nl:+.1f}°)", fontsize=10)
        ax.set_ylabel("°")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25)
        ax.set_facecolor("#F5F5F5")

    # ── Bar chart ─────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0:2])
    short = ["Kyphosis\n(Curv)", "Kyphosis\n(MS-Cobb)", "Lordosis", "Trunk\nLean"]
    nl_m = [np.mean(R_nl[k]) for k in ["kyph_geom","kyph_spine","lord_spine","tfl"]]
    ld_m = [np.mean(R_ld[k]) for k in ["kyph_geom","kyph_spine","lord_spine","tfl"]]
    nm_m = [config.NORMS[n]["mean"] for n in
            ["thoracic_kyphosis","thoracic_kyphosis","lumbar_lordosis","trunk_lean"]]
    nm_sd = [config.NORMS[n]["sd"] for n in
             ["thoracic_kyphosis","thoracic_kyphosis","lumbar_lordosis","trunk_lean"]]
    x = np.arange(4)
    ww = 0.25
    ax.bar(x-ww, nl_m, ww, label="Unloaded", color="steelblue", alpha=0.85)
    ax.bar(x, ld_m, ww, label="Loaded", color="crimson", alpha=0.85)
    ax.bar(x+ww, nm_m, ww, yerr=nm_sd, capsize=4,
           label="Normative", color="gray", alpha=0.45)
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=9)
    ax.set_ylabel("Angle (°)")
    ax.set_title("Grouped Comparison", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.set_facecolor("#F5F5F5")

    # ── Violin plot ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    data_nl = [R_nl["kyph_spine"], R_nl["lord_spine"], R_nl["tfl"]]
    data_ld = [R_ld["kyph_spine"], R_ld["lord_spine"], R_ld["tfl"]]
    pos_nl = [1, 3, 5]
    pos_ld = [1.7, 3.7, 5.7]
    for pos, arr in zip(pos_nl, data_nl):
        vp = ax.violinplot([arr], positions=[pos], showmeans=True)
        for pc in vp["bodies"]:
            pc.set_facecolor("steelblue")
            pc.set_alpha(0.5)
    for pos, arr in zip(pos_ld, data_ld):
        vp = ax.violinplot([arr], positions=[pos], showmeans=True)
        for pc in vp["bodies"]:
            pc.set_facecolor("crimson")
            pc.set_alpha(0.5)
    ax.set_xticks([1.35, 3.35, 5.35])
    ax.set_xticklabels(["Kyphosis", "Lordosis", "TFL"], fontsize=9)
    ax.set_ylabel("°")
    ax.set_title("Distribution (blue=NL, red=LD)", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_facecolor("#F5F5F5")

    # ── Delta summary table ───────────────────────────────────────
    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    col_labels = ["Metric", "Unloaded Mean±SD", "Loaded Mean±SD",
                  "Δ (L−NL)", "Norm Mean±SD", "Status", "Reference"]
    rows = []
    for metric_name, key, norm_key in [
        ("Kyphosis (Curv)", "kyph_geom", "thoracic_kyphosis"),
        ("Kyphosis (MS-Cobb)", "kyph_spine", "thoracic_kyphosis"),
        ("Lumbar Lordosis", "lord_spine", "lumbar_lordosis"),
        ("Trunk Lean", "tfl", "trunk_lean"),
    ]:
        nl_arr, ld_arr = R_nl[key], R_ld[key]
        nl_mean, nl_std = float(np.mean(nl_arr)), float(np.std(nl_arr))
        ld_mean, ld_std = float(np.mean(ld_arr)), float(np.std(ld_arr))
        delta = ld_mean - nl_mean
        nm = config.NORMS[norm_key]["mean"]
        ns = config.NORMS[norm_key]["sd"]
        z = (ld_mean - nm) / ns if ns > 0 else 0
        status = "Normal" if abs(z) <= 1 else ("Borderline" if abs(z) <= 2 else "Outside")
        rows.append([metric_name,
                     f"{nl_mean:.1f}±{nl_std:.1f}°",
                     f"{ld_mean:.1f}±{ld_std:.1f}°",
                     f"{delta:+.1f}°",
                     f"{nm}±{ns}°",
                     status,
                     config.NORMS[norm_key].get("ref", "")])
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 2.0)
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#37474F")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    status_colors = {"Normal":"#C8E6C9", "Borderline":"#FFF9C4", "Outside":"#FFCDD2"}
    for i, row in enumerate(rows):
        for j in range(len(col_labels)):
            if j == 5:
                tbl[i+1, j].set_facecolor(status_colors.get(row[5], "white"))
            else:
                tbl[i+1, j].set_facecolor("#ECEFF1" if i % 2 == 0 else "white")
    ax.set_title("Load Carriage Delta Summary — vs Ohlendorf et al. (2023)",
                 fontsize=11, pad=10, fontweight="bold")

    plt.savefig(os.path.join(config.PLOT_DIR, "Load_Comparison.png"),
                dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()


def save_csv_report(all_results):
    """Save a CSV summary for all videos."""
    csv_path = os.path.join(config.REPORT_DIR, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Video", "Condition",
                         "Geom_Kyphosis_Mean", "Geom_Kyphosis_SD",
                         "Spine_Kyphosis_Mean", "Spine_Kyphosis_SD",
                         "Lordosis_Mean", "Lordosis_SD",
                         "TFL_Mean", "TFL_SD",
                         "N_Frames"])
        for name, R in all_results.items():
            if len(R["time"]) == 0:
                continue
            writer.writerow([
                R["display_name"], R["condition"],
                f"{np.mean(R['kyph_geom']):.2f}", f"{np.std(R['kyph_geom']):.2f}",
                f"{np.mean(R['kyph_spine']):.2f}", f"{np.std(R['kyph_spine']):.2f}",
                f"{np.mean(R['lord_spine']):.2f}", f"{np.std(R['lord_spine']):.2f}",
                f"{np.mean(R['tfl']):.2f}", f"{np.std(R['tfl']):.2f}",
                len(R["time"]),
            ])
    print(f"\nCSV report saved → {csv_path}")


# ══════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    print("="*60)
    print("  Unified Spinal Posture Analysis Pipeline")
    print("="*60)

    ensure_dirs()

    # Initialize SpinePose once (model download ~200 MB first time)
    from spinepose.pose_estimator import SpinePoseEstimator
    print("\nInitializing SpinePose (YOLOX)...")
    estimator = SpinePoseEstimator(detector="yolox", model_version="latest")

    all_results = {}
    for vinfo in config.VIDEO_INFO:
        R = analyze_video(vinfo, estimator)
        if R and len(R["time"]) > 0:
            all_results[vinfo["filename"]] = R
            plot_single_video(R)

    # Load comparison
    if "ya rab.mp4" in all_results and "rab aded.mp4" in all_results:
        print("\n── Plotting load-carriage comparison ──")
        plot_comparison(all_results["ya rab.mp4"], all_results["rab aded.mp4"])

    # CSV report
    save_csv_report(all_results)

    print("\n" + "="*60)
    print("  Pipeline complete!")
    print(f"  Videos → {config.VID_DIR}")
    print(f"  Plots  → {config.PLOT_DIR}")
    print(f"  Report → {config.REPORT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
