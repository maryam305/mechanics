# Clinical SpinePose Posture Analysis Pipeline

A research-grade Python pipeline for estimating and analyzing spinal posture and load-carriage kinematics from 2D video. By combining the state-of-the-art **SpinePose (CVPR 2025)** keypoint detection model with established biomechanical algorithms (Multi-Segment Cumulative Cobb angle), this tool bridges the gap between unconstrained 2D video and clinical-grade spinal metric analysis.

---

## 📑 Table of Contents
- [Features](#-features)
- [Scientific Methodology](#-scientific-methodology)
- [Workflow & Pipeline](#-workflow--pipeline)
- [Project Structure](#-project-structure)
- [Prerequisites & Installation](#-prerequisites--installation)
- [Usage & Configuration](#-usage--configuration)
- [Outputs & Interpretation](#-outputs--interpretation)

---

## ✨ Features

- **Robust Keypoint Detection**: Utilizes the 37-keypoint SpinePose model tailored specifically for clinical back-surface landmarks.
- **Scientific Kinematics**: Computes standard spinal parameters including Thoracic Kyphosis, Lumbar Lordosis, and Trunk Forward Lean.
- **Advanced Signal Processing**: Applies Savitzky-Golay filtering to preserve kinematic peaks while removing tracking noise.
- **Normative Benchmarking**: Automatically compares subject metrics against published scientific literature (e.g., Ohlendorf et al., 2023).
- **Automated Reporting**: Generates overlaid MP4 videos, comprehensive multi-panel analysis dashboards, and quantitative CSV reports.

---

## 🔬 Scientific Methodology

Traditional 2D geometric curvature derivations (like polynomial tangent-angle methods) often produce under-scaled values (5°–15°) that do not align with clinical expectations. This pipeline employs the **Multi-Segment Cumulative Cobb Angle** method, which is the standard clinical approach adapted for 2D landmark data.

### 1. Thoracic Kyphosis (Curvature & Cobb)
Calculated via the absolute change in segment inclinations across the thoracic region: `C7 → T3 → T8 → L1`. 
This produces physiologically accurate values (expected ~40°–60°).

### 2. Lumbar Lordosis
Calculated across the lumbar chain: `L1 → L3 → L5 → Sacrum`.

### 3. Trunk Forward Lean (TFL)
Computed dynamically frame-by-frame using the global inclination angle of the `C7 → Sacrum` vector relative to the true vertical axis.

---

## ⚙️ Workflow & Pipeline

The pipeline operates fully autonomously once configured. The step-by-step internal workflow is as follows:

1. **Video Ingestion & Pre-processing** (`config.py`):
   - Reads input MP4 files defined in the video dictionary.
   - Extracts subject meta-data and load-carriage conditions (e.g., unloaded vs. loaded).

2. **SpinePose Inference**:
   - Isolates the human bounding box.
   - Infers exactly 37 skeletal/spinal landmarks.
   
3. **Anatomical Validity & Confidence Gating**:
   - Ensures landmarks flow sequentially from top-to-bottom (Y-axis progression `C7 < T3 < T8 < L1 < Sacrum`).
   - Ignores frames that fall below the `0.25` overall confidence threshold to prevent gross outliers.

4. **Biomechanical Angle Computation**:
   - Calculates the raw multi-segment Cobb angles.
   - Calculates polynomial surface approximations (used strictly for visual overlays, not direct angle output).

5. **Temporal Smoothing**:
   - Passes raw angular metrics through a continuous Savitzky-Golay smoothing filter to stabilize tracking without destroying peak bending moments.

6. **Serialization & Asset Generation**:
   - Overlays calculated data, polynomial back-splines, and keypoint nodes directly onto a new copy of the video.
   - Generates the comprehensive PNG dashboard and logs data to `summary.csv`.

---

## 📂 Project Structure

```text
📦 project-root
 ┣ 📜 spine_analysis.py       # Main execution engine and calculation pipeline
 ┣ 📜 config.py               # Central configuration (inputs, normative data, ML parameters)
 ┣ 📜 requirements.txt        # Python dependency list
 ┣ 📂 output_analysis         # (Generated automatically)
 ┃ ┣ 📂 videos                # Rendered videos with data overlays
 ┃ ┣ 📂 plots                 # High-resolution time-series & bar chart PNGs
 ┃ ┗ 📂 reports               # CSV quantitative summaries
 ┗ 🎞️ [Original Video Files].mp4
```

---

## 💻 Prerequisites & Installation

### Environment Setup

It is highly recommended to isolate the environment to avoid CUDA/ONNX version conflicts.

```bash
# 1. Create and activate a Virtual Environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# 2. Install core dependencies
pip install -r requirements.txt

# 3. Ensure SpinePose is installed in the environment
# Install via GitHub if not available via pip directly:
pip install git+https://github.com/dfki-av/spinepose.git
```

*Note on Hardware Acceleration: The pipeline natively supports the ONNX Runtime. For GPU acceleration on Windows/NVIDIA, ensure `onnxruntime-gpu` and the corresponding CUDA/cuDNN toolkits route correctly.*

---

## 🚀 Usage & Configuration

### 1. Configure the Inputs
Open `config.py` and modify the `VIDEO_INFO` array to point to your target videos.

```python
VIDEO_INFO = [
    {
        "filename": "subject_walk.mp4",
        "display_name": "Subject1_Unloaded",
        "condition": "unloaded",
        "subject_id": "subject_1",
    }
]
```

*You can define normative scientific standards and polynomial fit parameters in `config.py` under the `NORMS` and `CONTOUR_CONFIG` dictionaries.*

### 2. Execute the Pipeline
Run the main script from your terminal:

```bash
python spine_analysis.py
```

*The script will process each video sequentially, displaying a progress bar, and will exit once all data is saved to `output_analysis/`.*

---

## 📈 Outputs & Interpretation

Once execution finishes, navigate to `output_analysis/`. You will find three distinct types of artifacts:

### 1. Annotated Videos (`videos/`)
The original video is re-rendered frame-by-frame with:
- Red nodes indicating detected spinal keypoints.
- A green spline representing the interpolated curvature model.
- A live, frame-by-frame text overlay displaying Cobb Kyphosis, Lordosis, and Trunk Lean degrees.

### 2. Clinical Dashboards (`plots/`)
Highly detailed, multi-dimensional matplotlib figures generated per-video and per-cohort, including:
- **Time-Series Tracking:** Visualizes dynamic kyphosis/lordosis over absolute time, overlaid with green bands representing "Normal" normative ranges.
- **Bar/Violin Comparisons:** Highlights the mean difference between *Loaded* vs. *Unloaded* conditions for the same subject.
- **Summary Tables:** A rigid clinical table outputting Mean ± SD natively colored to reflect status (`Normal`/`Borderline`/`Outside`).

### 3. Quantitative Reports (`reports/`)
A `summary.csv` file provides absolute numerical mean and standard deviation matrices across the entire run processing queue for programmatic access.
