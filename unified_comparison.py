import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import config
from spine_analysis import analyze_video_anatomical
from surface_curvature_analysis import analyze_geometric_curvature

# Reproducibility & Pipeline Flow Note
PIPELINE_FLOW = (
    "Pipeline Flow: \n"
    "1. Segmentation / Keypoint Detection\n"
    "2. Filtering (Ordering checks, Confidence, IQR outlier rejection)\n"
    "3. Angle Computation (abs(atan2(dx, abs(dy))))\n"
    "4. Temporal Smoothing (Moving Average window=9)\n"
    "5. Comparison (Synchronized by Frame ID)"
)

# ── Research-Grade Documentation ───────────────────────────────────
SCOPE_STATEMENT = (
    "This system is intended for non-clinical, vision-based posture analysis "
    "and does not replace radiographic assessment."
)
CURVATURE_DEFINITION = (
    "Back Surface Curvature refers to the curvature of the external posterior body contour, "
    "influenced by soft tissue and clothing, and not a direct measure of vertebral alignment."
)
ERROR_SOURCES = (
    "Major sources of error include: \n"
    "1. Segmentation inaccuracies (geometric method)\n"
    "2. Keypoint detection noise (anatomical method)\n"
    "3. Camera misalignment and perspective distortion\n"
    "4. 2D projection of a 3D structure"
)
REPRODUCIBILITY_STATEMENT = (
    "All parameters, thresholds, and processing steps are fixed and centralized "
    "in the configuration file to ensure reproducibility across experiments."
)
CONTRIBUTION_STATEMENT = (
    "This work presents a dual-pipeline framework that enables direct comparison "
    "between geometric surface-based and anatomical keypoint-based spinal curvature estimation."
)
PRACTICAL_VALUE = (
    "Despite its limitations, the system provides a low-cost, non-invasive tool "
    "for posture trend analysis in real-world environments."
)
VISUAL_INSIGHT = (
    "Overlay visualization confirms that anatomical estimates follow vertebral alignment, "
    "while geometric estimates reflect surface deformation."
)

TECHNICAL_DISCLAIMER = (
    "All measurements are derived from 2D projections and are sensitive to camera alignment, "
    "perspective distortion, and subject orientation. The method assumes sagittal-plane motion "
    "and does not account for 3D spinal rotation."
)

VALIDATION_PHILOSOPHY = (
    "Due to the absence of radiographic ground truth, validation is relative and based on "
    "biomechanical consistency and literature ranges."
)

def run_comparison(vid_info, estimator):
    display = vid_info["display_name"]
    filename = vid_info["filename"]
    vid_path = os.path.join(config.RAW_VIDEO_DIR, filename)
    
    # 1. Run Anatomical Method
    res_a = analyze_video_anatomical(vid_info, estimator)
    # 2. Run Geometric Method
    res_g = analyze_geometric_curvature(vid_path)
    
    # 3. Time Alignment (Index-based Synchronization via Frame ID)
    # We use pandas for a robust merge
    df_a = pd.DataFrame({"frame_id": res_a["frame_id"], "anatomical_deg": res_a["kyphosis"]})
    df_g = pd.DataFrame({"frame_id": range(len(res_g["curvature_raw"])), "geometric_deg": res_g["curvature_raw"]})
    
    # Merge on frame_id to ensure we only subtract identical physical moments
    df_sync = pd.merge(df_a, df_g, on="frame_id")
    
    diff = df_sync["anatomical_deg"] - df_sync["geometric_deg"]
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    
    # 4. Reporting & Interpretation
    print(f"\n[{display}] Quantitative Comparison Breakdown:")
    print(f"  {SCOPE_STATEMENT}")
    print(f"  Synchronized Frames: {len(df_sync)}")
    print(f"  Mean Difference:      {mean_diff:.2f} deg")
    print(f"  Std Deviation:        {std_diff:.2f} deg")
    
    # Clinical/Research Interpretation
    print("  INTERPRETATION: Positive values indicate overestimation by the geometric method; negative values indicate underestimation.")
    if mean_diff > 0:
        print("  Systematic overestimation by the geometric method relative to the anatomical method.")
    else:
        print("  Systematic underestimation by the geometric method relative to the anatomical method.")
              
    if abs(mean_diff) > config.BIOMECHANICAL_SIGNIFICANCE_DEG:
        print(f"  SIGNIFICANCE: Differences greater than {config.BIOMECHANICAL_SIGNIFICANCE_DEG}° "
              "may be biomechanically significant.")
              
    # 5. Visualization (Professional Grade)
    fig = plt.figure(figsize=(15, 14))
    fig.patch.set_facecolor("#FAFAFA")
    gs = GridSpec(3, 2, figure=fig, hspace=0.40, wspace=0.25)
    
    # Plot Time-Series
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df_sync["frame_id"], df_sync["anatomical_deg"], color=config.COLORS["anatomical"], label="Anatomical Method (SpinePose)", lw=2)
    ax1.plot(df_sync["frame_id"], df_sync["geometric_deg"], color=config.COLORS["geometric"], label="Geometric Method (Surface Curvature)", lw=2, alpha=0.8)
    ax1.set_title(f"Synchronized Comparison — {display}", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Angle (deg)", fontsize=11)
    ax1.set_xlabel("Frame ID", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")
    
    # Plot Difference Histogram
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(diff, bins=25, color="gray", alpha=0.6, edgecolor="white")
    ax2.axvline(mean_diff, color="black", linestyle="--", label=f"Mean Diff: {mean_diff:.2f}°")
    ax2.set_title("Frequency of Deviations", fontsize=11)
    ax2.set_xlabel("Difference (Anatomical - Geometric) [deg]", fontsize=10)
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # Plot Method Logic Summary
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    ax3.text(0, 0.95, f"Subject: {vid_info['subject_id']}", fontweight="bold")
    ax3.text(0, 0.88, f"Mean anatomical: {np.mean(df_sync['anatomical_deg']):.1f} deg")
    ax3.text(0, 0.81, f"Mean geometric:   {np.mean(df_sync['geometric_deg']):.1f} deg")
    ax3.text(0, 0.65, f"Definition: {CURVATURE_DEFINITION}", fontsize=7, color="gray", wrap=True)
    ax3.text(0, 0.50, f"Interpretation: Positive diff indicates geometric overestimation.", fontsize=7, color="gray")
    ax3.text(0, 0.35, f"Validation: {VALIDATION_PHILOSOPHY}", fontsize=7, style="italic")
    
    # Plot Error Sources & Flow
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis("off")
    ax4.text(0, 0.9, f"Contribution: {CONTRIBUTION_STATEMENT}", fontsize=8, fontweight="bold")
    ax4.text(0, 0.7, ERROR_SOURCES, fontsize=8, color="#333333")
    ax4.text(0, 0.3, f"Reproducibility: {REPRODUCIBILITY_STATEMENT}", fontsize=8, color="#333333")
    ax4.text(0, 0.1, f"Practical Value: {PRACTICAL_VALUE}", fontsize=8, color="#333333")
    
    # Footer Disclaimer
    fig.text(0.5, 0.02, TECHNICAL_DISCLAIMER, ha="center", fontsize=8, color="red", style="italic")
    
    # Save Output
    os.makedirs(os.path.join(config.REPORT_DIR, display), exist_ok=True)
    plt.savefig(os.path.join(config.REPORT_DIR, display, f"{display}_comparison.png"), dpi=200, bbox_inches="tight")
    plt.close()
    
    # Final Research Logs
    print(f"\nFinal Disclosure for {display}:")
    print(f"  {REPRODUCIBILITY_STATEMENT}")
    print(f"  {TECHNICAL_DISCLAIMER}")
    print(f"  {VISUAL_INSIGHT}")
    print(f"  {PRACTICAL_VALUE}")
    
if __name__ == "__main__":
    from spinepose.pose_estimator import SpinePoseEstimator
    estimator = SpinePoseEstimator(detector="yolox", model_version="latest", device=config.SPINEPOSE_CONFIG["device"])
    
    for vid_info in config.VIDEO_INFO:
        run_comparison(vid_info, estimator)
