# Dual-Method Spinal Curvature Analysis & AI Calibration Layer
**Automated Clinical-Grade Posture Assessment Pipeline**

## 🚀 1. Project Overview
This project implements a robust, dual-method pipeline for measuring spinal curvature from 2D side-view (sagittal) video. It uses Computer Vision to replicate clinical radiographic standards (Cobb Angle) while simultaneously profiling back-surface geometry. An AI-driven calibration layer then reconciles the discrepancies between internal anatomical landmarks and external surface observations.

---

## 🏗️ 2. System Architecture
The system operates through five major stages:
1.  **Video Ingestion:** Native FPS frame extraction and adaptive resizing.
2.  **Dual-Detection Path:**
    *   **Anatomical Tracker:** SpinePose (CVPR 2025) detects 37 spinal keypoints.
    *   **Geometric Tracker:** MediaPipe Selfie Segmentation extracts the person's silhouette.
3.  **Algorithmic Analysis:**
    *   **Cobb Calculation:** Multi-segment inclination analysis with lean-bias removal.
    *   **Back Profiling:** Centerline extraction followed by degree-3 polynomial fitting.
4.  **Temporal Post-Processing:** Savitzky-Golay denoising and outlier rejection.
5.  **AI Calibration:** Features from both paths are fed into a RandomForest Regressor to predict the clinical Cobb angle with high precision.

---

## 📐 3. Technical Algorithms

### 3.1 Anatomical Cobb Angle (spine_analysis.py)
The Cobb angle is computed using a robust multi-segment estimator:
*   **Formula:** `Cobb = |Mean(Top-2 Inclinations) - Mean(Bottom-2 Inclinations)|`
*   **Inclination Definition:** `atan2(dx, |dy|)` relative to the vertical axis.
*   **Lean Bias (FIX-2):** The global trunk lean (C7 to Sacrum vector) is subtracted from every individual segment's inclination before the Cobb calculation. This ensures we measure **true curvature** rather than just forward tilt.
*   **Spinal Chain:** The model specifically tracks C1, C4, C7, T3, T8, L1, L3, L5, and the Sacrum.

### 3.2 Geometric Surface Curvature (surface_curvature_analysis.py)
*   **Segmentation:** Primarily uses MediaPipe, with a luminance-based GrabCut fallback for low-light frames.
*   **Centerline Extraction (FIX-4):** To avoid bias from loose clothing or arms, the x-coordinate is the mean of the left and right silhouette edges at every y-level (`cx = (x_left + x_right) / 2`).
*   **Curve Fitting:** A cubic polynomial ($x = ay^3 + by^2 + cy + d$) is fitted to the centerline.
*   **Surface Angle:** Determined by the mathematical difference in the first derivative (tangent) at the shoulder and hip levels.

---

## 🧠 4. AI Calibration Layer (calibration_model.py)
The system reconciles the two pipelines using a 6-feature matrix:
1.  `spinepose_angle`: The raw anatomical Cobb angle.
2.  `surface_curvature`: The raw geometric profile angle.
3.  `angle_diff`: The disagreement between the two trackers.
4.  `trunk_lean`: The removed forward tilt bias.
5.  `confidence`: The mean SpinePose detection confidence (used for error weighting).
6.  `curvature_variance`: Rolling 5-frame variance to detect temporal instability.

---

## 🛑 5. Troubleshooting & Windows Fixes
During development, several platform-specific issues were identified and resolved:
*   **Unicode Encoding Crash:** Windows `cmd` and `powershell` often fail when printing non-standard symbols like `→`. All logging symbols were replaced with standard `->` to prevent `UnicodeEncodeError`.
*   **MediaPipe Legacy Fix:** Recent MediaPipe versions often omit the `solutions` submodule. We forced installation of `mediapipe==0.10.11` and downgraded `protobuf<4` to restore functionality.
*   **Directory Management:** Automated creation of `data/`, `outputs/`, and `models/` folders ensures the pipeline runs out-of-the-box.

---

## 📊 6. Clinical Standards Comparison
The pipeline classifies posture into three clinical groups based on the Mendeley Posture Dataset:

| Group | Cobb Angle (Radiographic) | Surface Curvature (Geometric) |
| :--- | :--- | :--- |
| **Normal** | 20° – 40° | 0° – 20° |
| **Mild** | 40° – 60° | 20° – 40° |
| **Severe** | > 60° | > 40° |

---

## 🛠️ 7. Installation & Usage
### Installation
1.  **Environment:** Python 3.11+
2.  **Dependencies:** `pip install -r requirements.txt`
3.  **Setup:** `python setup_datasets.py` (Downloads weights, clinical ranges, and SpineTrack data).

### Execution
*   **Full Pipeline:** `python unified_pipeline.py`
*   **Build Features Only:** `python unified_pipeline.py --build-dataset`
*   **Evaluation Only:** `python unified_pipeline.py --eval-only`

---

## 📧 8. Research Credits
*   **Methodology:** Anatomical referencing based on SpinePose (CVPR 2025).
*   **Validation:** Biomechanical grounding derived from the Mendeley Posture Dataset.
