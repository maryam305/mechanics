"""
unified_pipeline.py — Master Orchestration Script
==================================================
Runs both pipelines, trains the calibration model, evaluates results,
and produces all figures, CSVs, and the comparison report.

Usage:
    python unified_pipeline.py                  # Full run
    python unified_pipeline.py --build-dataset  # Build feature CSV from own dataset
    python unified_pipeline.py --eval-only      # Load saved model, evaluate only

Outputs (in config.OUTPUT_DIR):
    figures/   — PNG plots per video + comparison
    csv/       — Per-frame results
    models/    — Trained calibration model
    report/    — Summary statistics table
"""

import os, sys, argparse, csv, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import config

from surface_curvature_analysis import (
    analyze_geometric_curvature, export_csv as export_geo_csv
)
from spine_analysis import analyze_video_anatomical, export_csv as export_anat_csv
from calibration_model import (
    RandomForestCalibrator, build_feature_matrix,
    load_mendeley_ranges, load_own_dataset_labels,
    validate_against_clinical_ranges
)

warnings.filterwarnings("ignore")

REPORT_DIR = os.path.join(config.OUTPUT_DIR, "report")
os.makedirs(REPORT_DIR, exist_ok=True)


# ── SpinePose Estimator Loader ─────────────────────────────────────

def load_spinepose_estimator():
    """
    Loads the SpinePose estimator. Wraps import so the pipeline gracefully
    handles environments where SpinePose is not installed.
    """
    try:
        from spinepose import SpinePoseEstimator   # Install per SpinePose README
        est = SpinePoseEstimator(detector="yolox", model_version="v1")
        print("[SpinePose] Estimator loaded.")
        return est
    except ImportError:
        print("[SpinePose] Package not installed. Using mock estimator.")
        return _MockEstimator()


class _MockEstimator:
    """
    Mock estimator for unit-testing the pipeline without GPU/SpinePose.
    Generates plausible synthetic keypoints for the 37-joint model.
    """
    def detect(self, frame):
        h, w = frame.shape[:2]
        return [np.array([0, 0, w, h], dtype=float)]

    def estimate(self, frame, bboxes):
        h = frame.shape[0]
        rng = np.random.default_rng(seed=42)
        kpts = np.zeros((37, 3))
        # Spine chain: C1(36)..Sacrum(19) — place vertically in frame center
        spine_map = {36: 0.1, 35: 0.18, 18: 0.25, 30: 0.35,
                     29: 0.48, 28: 0.58, 27: 0.65, 26: 0.72, 19: 0.82}
        for idx, frac in spine_map.items():
            kpts[idx, 0] = frame.shape[1] * 0.5 + rng.uniform(-5, 5)
            kpts[idx, 1] = h * frac
            kpts[idx, 2] = rng.uniform(0.6, 0.95)
        # Fill remaining with noise
        for i in range(37):
            if kpts[i, 2] == 0:
                kpts[i] = [frame.shape[1]*0.5, h*0.5, 0.3]
        return [kpts[np.newaxis, :, :]]


# ── Per-Video Analysis ─────────────────────────────────────────────

def run_single_video(vid_info: dict, estimator) -> dict:
    """
    Run both pipelines on a single video. Returns combined result dict.
    """
    vid_path = os.path.join(config.RAW_VIDEO_DIR, vid_info["filename"])
    label    = vid_info["display_name"]

    print(f"\n{'='*60}")
    print(f"  Video: {label}")
    print(f"{'='*60}")

    # Geometric pipeline
    geo = analyze_geometric_curvature(vid_path)

    # Anatomical pipeline
    anat = analyze_video_anatomical(vid_info, estimator)

    # CSV export
    export_geo_csv(geo, label)
    export_anat_csv(anat, label)

    return {"anat": anat, "geo": geo, "label": label, "vid_info": vid_info}


# ── Calibration ────────────────────────────────────────────────────

def run_calibration(all_results: list) -> tuple:
    """
    Build feature matrix from all videos, train RandomForest calibrator,
    add corrected_angle to each result dict.
    """
    print("\n" + "="*60)
    print("  CALIBRATION MODEL TRAINING")
    print("="*60)

    dfs = []
    for r in all_results:
        if r["anat"]["kyphosis"] and r["geo"]["curvature_raw"]:
            df = build_feature_matrix(r["anat"], r["geo"])
            df["video_label"] = r["label"]
            dfs.append(df)

    if not dfs:
        print("  WARNING: No valid frames across all videos. Skipping calibration.")
        return None, all_results

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(
        os.path.join(config.OWN_DATASET_PATH, "features.csv"), index=False
    )
    print(f"  Feature matrix: {combined_df.shape[0]} samples × {combined_df.shape[1]} features")

    calibrator = RandomForestCalibrator()
    calibrator.fit(combined_df)
    calibrator.save()

    # Feature importance
    fi = calibrator.feature_importance()
    print("\n  Feature Importance:")
    for feat, imp in sorted(fi.items(), key=lambda x: -x[1]):
        print(f"    {feat:<28} {imp:.4f}")

    # Add corrected angle to each result
    for r in all_results:
        df = build_feature_matrix(r["anat"], r["geo"])
        if len(df) > 0:
            r["corrected"] = calibrator.predict(df)
        else:
            r["corrected"] = np.array([])

    return calibrator, all_results


# ── Figures ────────────────────────────────────────────────────────

def plot_video_comparison(r: dict):
    """Per-video 3-panel figure: Geometric | Anatomical | Corrected."""
    anat      = r["anat"]
    geo       = r["geo"]
    corrected = r.get("corrected", np.array([]))
    label     = r["label"]
    colors    = config.PLOT_COLORS

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Spinal Analysis — {label}", fontsize=14, fontweight="bold")

    # Panel 1: Geometric
    ax = axes[0]
    ax.plot(geo["time"], geo["curvature_raw"], alpha=0.3,
            color=colors["geometric"], label="Raw")
    ax.plot(geo["time"],
            geo["curvature_s"][:len(geo["time"])] if len(geo["curvature_s"]) >= len(geo["time"])
            else np.resize(geo["curvature_s"], len(geo["time"])),
            color=colors["geometric"], label="Smoothed")
    ax.set_title("Geometric Surface Curvature")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (°)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Anatomical
    ax = axes[1]
    if anat["kyphosis_raw"]:
        ax.plot(anat["time"], anat["kyphosis_raw"], alpha=0.3,
                color=colors["anatomical"], label="Raw")
        ax.plot(anat["time"], anat["kyphosis"],
                color=colors["anatomical"], label="Smoothed")
    ax.set_title("Anatomical Cobb Angle (SpinePose)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (°)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Corrected
    ax = axes[2]
    if len(corrected) > 0:
        t = anat["time"][:len(corrected)]
        ax.plot(t, corrected, color=colors["calibrated"], label="AI Corrected")
        if anat["kyphosis"]:
            ax.plot(t, anat["kyphosis"][:len(corrected)],
                    "--", color=colors["anatomical"], alpha=0.5, label="SpinePose")
    else:
        ax.text(0.5, 0.5, "Calibration N/A", transform=ax.transAxes, ha="center")
    ax.set_title("AI-Corrected Angle")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (°)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(config.FIGURE_DIR, f"{label.replace(' ','_')}_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved -> {out}")


def plot_method_comparison_overview(all_results: list):
    """
    Summary bar chart: mean angle per method per video.
    Reproduces the comparison table from the roadmap.
    """
    labels   = [r["label"] for r in all_results]
    geo_means = [np.mean(r["geo"]["curvature_raw"]) if r["geo"]["curvature_raw"] else 0
                 for r in all_results]
    anat_means = [np.mean(r["anat"]["kyphosis"]) if r["anat"]["kyphosis"] else 0
                  for r in all_results]
    corr_means = [float(np.mean(r["corrected"])) if len(r.get("corrected", [])) > 0 else 0
                  for r in all_results]

    x    = np.arange(len(labels))
    w    = 0.25
    c    = config.PLOT_COLORS

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - w,    geo_means,  w, label="Geometric",  color=c["geometric"])
    ax.bar(x,        anat_means, w, label="Anatomical",  color=c["anatomical"])
    ax.bar(x + w,    corr_means, w, label="AI Corrected",color=c["calibrated"])

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("Mean Angle (°)")
    ax.set_title("Method Comparison — Mean Spinal Curvature per Video")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    out = os.path.join(config.FIGURE_DIR, "method_comparison_overview.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Overview figure -> {out}")


# ── Summary Report CSV ─────────────────────────────────────────────

def write_summary_report(all_results: list, calibrator):
    """Write per-video summary statistics to CSV."""
    rows = []
    for r in all_results:
        label     = r["label"]
        geo_raw   = r["geo"]["curvature_raw"]
        anat_raw  = r["anat"]["kyphosis_raw"] if r["anat"]["kyphosis_raw"] else []
        anat_sm   = r["anat"]["kyphosis"]
        corr      = r.get("corrected", [])

        row = {
            "video":               label,
            "valid_frames_geo":    r["geo"]["valid_frames"],
            "valid_frames_anat":   r["anat"]["stats"]["valid"],
            "geo_mean_deg":        f"{np.mean(geo_raw):.2f}" if geo_raw else "N/A",
            "geo_std_deg":         f"{np.std(geo_raw):.2f}"  if geo_raw else "N/A",
            "anat_mean_deg":       f"{np.mean(anat_sm):.2f}" if anat_sm else "N/A",
            "anat_std_deg":        f"{np.std(anat_sm):.2f}"  if anat_sm else "N/A",
            "corrected_mean_deg":  f"{float(np.mean(corr)):.2f}" if len(corr) > 0 else "N/A",
            "corrected_std_deg":   f"{float(np.std(corr)):.2f}"  if len(corr) > 0 else "N/A",
            "mean_confidence":     f"{np.mean(r['anat']['confidence']):.3f}"
                                    if r["anat"]["confidence"] else "N/A",
        }

        # Clinical validation against Mendeley ranges
        if anat_sm:
            print(f"\n  Clinical Validation ({label}) — Anatomical:")
            validate_against_clinical_ranges(
                np.array(anat_sm), method="anatomical"
            )

        rows.append(row)

    out = os.path.join(REPORT_DIR, "summary_statistics.csv")
    if rows:
        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"\n  Summary report -> {out}")

    # Print calibrator metrics if available
    if calibrator:
        m = calibrator.metrics
        print("\n  AI Calibration Model Performance:")
        print(f"    MAE:           {m['MAE']:.3f} deg")
        print(f"    STD residuals: {m['STD_residual']:.3f} deg")
        print(f"    Correlation R: {m['R']:.4f}")


# ── Main Entry Point ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Unified Spinal Analysis Pipeline")
    parser.add_argument("--build-dataset", action="store_true",
                        help="Build feature CSV from own dataset only")
    parser.add_argument("--eval-only", action="store_true",
                        help="Load saved model and evaluate without re-training")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  UNIFIED SPINAL POSTURE ANALYSIS PIPELINE")
    print(f"  {config.REPRODUCIBILITY_NOTE}")
    print("="*60)

    np.random.seed(config.RANDOM_SEED)

    # Load SpinePose estimator
    estimator = load_spinepose_estimator()

    # Validate Mendeley dataset availability
    mendeley = load_mendeley_ranges()
    print(f"\n  Mendeley clinical ranges loaded: {list(mendeley.keys())}")

    # Run all videos
    all_results = []
    for vid_info in config.VIDEO_INFO:
        vid_path = os.path.join(config.RAW_VIDEO_DIR, vid_info["filename"])
        if not os.path.exists(vid_path):
            print(f"  WARNING: Video not found: {vid_path} — skipping.")
            continue
        r = run_single_video(vid_info, estimator)
        all_results.append(r)

    if not all_results:
        print("\n  No valid videos processed. Check config.RAW_VIDEO_DIR.")
        sys.exit(1)

    # Calibration model
    if args.eval_only:
        calibrator = RandomForestCalibrator().load()
        for r in all_results:
            from calibration_model import build_feature_matrix
            df = build_feature_matrix(r["anat"], r["geo"])
            r["corrected"] = calibrator.predict(df) if len(df) > 0 else np.array([])
    else:
        calibrator, all_results = run_calibration(all_results)

    # Figures
    print("\n  Generating figures...")
    for r in all_results:
        plot_video_comparison(r)
    plot_method_comparison_overview(all_results)

    # Report
    write_summary_report(all_results, calibrator)

    print("\n" + "="*60)
    print("  PIPELINE COMPLETE")
    print(f"  Outputs in: {config.OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
