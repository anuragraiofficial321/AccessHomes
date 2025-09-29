# AccessMate

Maps first-seen object detections from a video onto a floor plan using ARKit camera poses and depth estimates. It detects objects with YOLO, estimates depth with ZoeDepth, reprojects detections into world coordinates, enforces room containment, and saves a plotted floor plan with icons and per-detection annotated images.

---

# Table of contents

1. [Prerequisites](#prerequisites)
2. [Repository layout (visual)](#repository-layout-visual)
3. [Quick start — run pipeline](#quick-start---run-pipeline)
4. [Configuration (what to edit)](#configuration-what-to-edit)
5. [Detailed explanation of outputs](#detailed-explanation-of-outputs)
6. [How the pipeline works (step-by-step)](#how-the-pipeline-works-step-by-step)
7. [Troubleshooting & common issues](#troubleshooting--common-issues)
8. [Extending / customizing](#extending--customizing)
9. [Example commands and expected outputs](#example-commands-and-expected-outputs)
10. [Requirements / `requirements.txt`](#requirements--requirementstxt)

---

# Prerequisites

* Python 3.9+ (3.10 recommended). Virtualenv/venv strongly recommended.
* Enough disk space for models (YOLO weights and transformer models if used).
* Recommended CPU / GPU for performance:

  * CPU-only works but may be slow for Zoe and YOLO inference.
  * GPU recommended for Zoe (`torch` + CUDA) and Ultraytics YOLO if you have NVIDIA GPU.

Install system packages first (on Ubuntu example):

### Create & activate venv:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install Python dependencies (see `requirements.txt` bottom of this README):

```bash
pip install -r requirements.txt
```

**Optional** for ZoeDepth GPU acceleration:

* Install a CUDA-compatible PyTorch per your CUDA version (`pip` or `conda`) — visit PyTorch site for the right command.
* Then reinstall `transformers` and `accelerate` into the same venv if needed.

---

# Quick start — run pipeline

1. Put your inputs into `input_data/`:

   * `arkitData.json` — ARKit or iOS recording metadata containing camera transforms (required).
   * `floor_plan.json` — floor plan JSON (required).
   * `video.mp4` — video file used for detection (optional — if missing, detection is skipped).

2. Edit `config/config.py` if your input paths differ:

   * `DATA_DIR`, `ARKIT_PATH`, `FLOOR_PATH`, `VIDEO_PATH`
   * `YOLO_MODEL` path (if you use a custom model)
   * `TARGET_CLASSES` (list of classes to collect first-seen frames)

3. Run:

```bash
source venv/bin/activate
python main.py
```

Outputs appear in `output_data/`. See “Detailed explanation of outputs” below.

---

# Configuration (what to edit)

Primary config: `config/config.py`
Key variables and what they do:

* `DATA_DIR`: base input directory (default `input_data`).
* `ARKIT_PATH`: path to ARKit JSON (default `input_data/arkitData.json`).
* `FLOOR_PATH`: path to floor plan JSON.
* `VIDEO_PATH`: path to video file.
* `OUT_DATA_DIR`: directory where outputs will be saved.
* `YOLO_MODEL`: path to YOLO weights (ultralytics `.pt`).
* `YOLO_DEVICE`: `"cpu"` or `"cuda:0"` (if using GPU).
* `ZOE_MODEL_NAME`: Hugging Face model name for ZoeDepth.
* `ZOE_DEVICE`: `"cpu"` or `"cuda"`.
* `TARGET_CLASSES`: list of classes (strings) to collect first-seen images for.
* `CLASS_IMAGE_JSON`: path to `class_image.json` used by plotter to place PNG icons.

Important: don’t change function signatures or logic unless you know what you’re doing. The refactor preserved interfaces so `main.py` can call the modules.

---

# Detailed explanation of outputs

All outputs are saved in `output_data/`. Important files:

* `output_data/first_seen_detections.csv`
  CSV of all accepted (first-seen) detections. Columns include:

  * `class`: human class name (e.g., "sink")
  * `video_frame_index`: frame in the video where detected
  * `arkit_index`: matching ARKit frame index used for mapping
  * `cam_mapped_x`, `cam_mapped_y`: mapped camera floor coordinates
  * `object_mapped_x`, `object_mapped_y`: mapped object coordinates on floor plan
  * `object_yaw_deg`: estimated yaw angle (camera-relative)
  * `conf`: detection confidence
  * `distance_m`, `distance_ft`: depth measured by Zoe (meters / feet)

* `output_data/first_seen_detections_debug.csv`
  Per-detection debugging info (if available), including reprojection residuals.

* `output_data/reproject_debug.csv`
  Reprojection debug info saved if `DEBUG_REPROJECT` is enabled.

* `output_data/annotated_frames/`
  Per-detection full-frame images with bounding box + class label. Filenames:
  `vf{video_frame}_arkit{arkit}_{class}.jpg`

* `output_data/accepted_frames/`
  Visuals of accepted detections (bolder bbox + "ACCEPTED" banner). Filenames:
  `vf{video_frame}_arkit{arkit}_{class}_ACCEPT.jpg`
  The path to the accepted frame is stored for accepted detections in the `annotated_frame` field inside the detection info (and can be exported to CSV).

* `output_data/floor_plan_with_pngs.png`
  Floor plan plot with colored rooms and placed PNG icons for accepted detections.

---

# How the pipeline works (step-by-step)

1. **Load ARKit positions (`arkitData.json`)**

   * `main.py` -> `extract_positions_list()` parses camera transform matrices or x/y entries into `positions3d` and `meta`.

2. **Project 3D -> 2D**

   * `project_3d_to_2d()` converts `positions3d` to 2D (various projections supported).

3. **Load floor polygons**

   * `floor/floor_loader.py` reads `floor_plan.json` and extracts room polygons (`spaces`) and fixed furniture.

4. **Auto-map cameras onto floor**

   * Try multiple candidate projections and choose one that maximizes the number of camera positions falling inside rooms (heuristic).
   * Compute a 2D similarity (Umeyama) to align camera 2D track to floor coordinates.

5. **Initialize detectors**

   * YOLO via `detectors/yolo_detector.py` (requires `ultralytics`).
   * ZoeDepth via `detectors/zoe_depth.py` (optional; requires `transformers` + `torch`).

6. **Process video frames** (`process_video_first_per_class`)

   * For each frame, detect objects with YOLO.
   * For each detection, get depth from ZoeDepth (median patch of depth around bbox center).
   * Reproject object into world coordinates using camera pose and depth (`compute_object_world_and_mapped`) and map to floor coordinates.
   * **Room containment check:** ensure object and camera map into the same room polygon (with tolerance `room_margin`).

     * If allowed, nudge the mapped point inside room for better icon placement (`inside_push` distance).
   * If `save_detected=True`, save:

     * annotated full-frame image under `output_data/annotated_frames/`
     * if accepted (passes room check), save accepted-frame under `output_data/accepted_frames/` and attach path in detection info.

7. **Save CSVs & plot**

   * Save `first_seen_detections.csv` and plot floor plan via `floor/plotter.py`.

---

# Troubleshooting & common issues

* **`ultralytics.YOLO not available`**

  * Install via `pip install ultralytics`. Optionally ensure CUDA GPU support if desired.

* **ZoeDepth import errors (`transformers` / `torch`)**

  * Install `pip install torch transformers accelerate pillow`. If using GPU, install the correct `torch` wheel for your CUDA version.

* **`arkitData.json` missing or no camera transforms found**

  * Ensure `arkitData.json` contains camera transform matrices or `x`/`y` values. The extractor looks for `cameraTransform`, `transform`, `matrix`, or `x/y` pairs.

* **No detections saved**

  * Check `TARGET_CLASSES` in `config/config.py`. Make sure YOLO classes match expected names (you can print YOLO `model.names` to see class mapping).
  * Lower `YOLO_CONF` if too strict.

* **Annotated frames not saved**

  * Ensure `save_detected=True` is passed to `process_video_first_per_class` (default in `main.py` uses `True`) and program has write permission to `output_data/`.

* **Plot icons not visible or wrong scale**

  * Check `class_image.json` entries: each mapping should include `path` and ideally `width`/`height` in floor units. If missing, icons fall back to pixel size with a minimum.

* **ZoeDepth very slow**

  * Use GPU if available. Zoe model may be heavy; pre-infer or cache depth maps if re-running.

---

# Extending / customizing

* **Change classes to collect**: edit `TARGET_CLASSES` in `config/config.py`.
* **Change room margin or inside push**: when calling `process_video_first_per_class` in `main.py`, tune `room_margin` (units are same as floor plan) and `inside_push` (push mapped point inside polygon by X units).
* **Save crop images**: add a small snippet in `video/video_processing.py` where detection happens to crop `frame[y1:y2, x1:x2]` and save.
* **Use different YOLO weights**: place weights in `models/` and set `YOLO_MODEL` accordingly.
* **Export saved file paths to CSV**: add `annotated_frame` and file path columns when writing `first_seen_detections.csv` in `main.py`.

---

# Example commands and expected outputs

Run full pipeline (example):

```bash
# inside virtualenv
python main.py
```

Expected console output (abridged):

```
Loading and extracting ARKit positions...
Extracted 512 frames (3D positions).
No control points provided — using automatic mapping heuristic.
proj x,-z -> score 350 ...
Auto-chosen projection: x,-z rotation: 0.0 score: 350
Initializing YOLO detector (this may load weights and take a moment)...
Video opened: input_data/video.mp4, frames: 1800 size: 1280x720
DETECTED-ACCEPT: sink vf123 cam=(12.3,45.6) obj=(13.1,44.8) dist=2.345m ...
Saved reproject debug CSV: output_data/reproject_debug.csv
Saved per-detection debug CSV: output_data/first_seen_detections_debug.csv
Saved detections CSV: output_data/first_seen_detections.csv
Saved overlay image: output_data/floor_plan_with_pngs.png
Done.
```

Open this folder to view:

* `output_data/annotated_frames/` — many images (each detection)
* `output_data/accepted_frames/` — the accepted ones you care about
* `output_data/floor_plan_with_pngs.png` — final floorplan view

---

# Requirements (`requirements.txt`)

Save this as `requirements.txt` in project root and install with `pip install -r requirements.txt`.

```
numpy>=1.22
pandas>=1.3
matplotlib>=3.4
opencv-python>=4.5
ultralytics>=8.0   # optional; required for YOLO detection
transformers>=4.30 # optional; required for ZoeDepth
torch              # optional; install the correct wheel for your CUDA/CPU
accelerate>=0.20   # optional; for transformers acceleration
Pillow>=9.0
```

**Note:** Installing `torch` should typically be done following the official instructions for your platform (CUDA or CPU). Example for CPU-only:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

# Final tips & checklist

* Prepare `input_data/arkitData.json` and `input_data/floor_plan.json` first. The script cannot run mapping without floor polygons.
* If you don't have Zoe or YOLO available, the pipeline will print warnings and exit gracefully (it requires YOLO to produce detections in the current main pipeline).
* Use small sample video & model to test quickly before running a long video.
* Keep `class_image.json` icons and `models/` weights in the repo or point paths in `config/config.py`.

---