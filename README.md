## Overview — what this project does

This codebase takes ARKit tracking data + a floor plan + a video and produces:

* **Mapped floor-plan overlays (PNG)** with first-seen object detections plotted on the floor plan.

  * `floor_plan_with_emojis.png` — labels with emoji (if available).
  * `floor_plan_plain.png` — plain-text labels.
* **First-seen detections CSV** (rows for the first frame each requested class is detected):

  * `first_seen_detections.csv`
* **Debug CSVs** (optional, if enabled):

  * `first_seen_detections_debug.csv` — per-detection debug fields (mapping, depths, reprojections).
  * `reproject_debug.csv` — reprojection u/v and estimated u\_est/v\_est records.
* **Saved crops & annotated frames** (optional, controlled by config): one image crop per first-seen detection and an annotated video-frame image.
* **Console log output** describing progress and warnings.

It tries to compute where objects in the camera/video view lie on the 2D floorplan by:

1. extracting camera poses (3D) from ARKit JSON,
2. projecting camera positions into 2D,
3. automatically aligning those 2D points to the floor-plan coordinate system,
4. running an object detector (YOLO) on the video to find objects and the frame where each class is first seen,
5. estimating object depth (ZoeDepth deep model — or fallback),
6. raycasting from camera through the bounding-box center using depth to compute the 3D object point,
7. mapping that 3D point into floor-plan 2D via the previously computed similarity transform, and
8. plotting results and saving CSVs/images.

---

## Files & layout (what matters)

```
project/
│─ config/config.py          # paths, models, toggles (edit here)
│─ main.py                   # entrypoint (runs full pipeline)
│─ detectors/zoe_depth.py    # Zoe wrapper (depth estimation)
│─ detectors/yolo_detector.py# YOLO detection wrapper
│─ utils/*.py                # parsing, projection, intrinsics helpers
│─ video/video_processing.py # detection -> depth -> world mapping logic
│─ floor/floor_loader.py     # load polygons + fixed furniture
│─ floor/plotter.py          # plotting / PNG saving
```

You asked not to change algorithmic logic; the split keeps original behavior.

---

## Config you will edit

`config/config.py` contains these important values:

* `DATA_DIR`, `ARKIT_PATH`, `FLOOR_PATH`, `VIDEO_PATH` — input/output file paths.
* `YOLO_MODEL`, `YOLO_DEVICE`, `YOLO_CONF`, `YOLO_IOU` — YOLO detector config.
* `ZOE_MODEL_NAME`, `ZOE_DEVICE` — Zoe model id or local checkpoint and device.
* `TARGET_CLASSES` — classes to collect (first-seen). Example: `["couch","person","bed","chair"]`.
* `PROJECTION` — which axes to use for map projection (default `"x,-z"`).
* `CONTROL_POINTS` — optional exact (frame\_index, floor\_x, floor\_y) pairs for Umeyama alignment.
* Debug toggles: `DEBUG_REPROJECT`, `SAVE_DETECTED_CROPS`, `VERBOSE`.

Edit those paths and values to match your files.

---

## High-level pipeline (diagram)

Mermaid (if your viewer supports it):

```mermaid
flowchart LR
  A[arkitData.json] --> B[extract_positions_list()]
  B --> C[project_3d_to_2d]
  C --> D[auto_map_and_choose / Umeyama]
  D --> E[mapped_plot (2D cam positions)]
  F[floor_plan.json] --> G[load_floor_polygons()]
  G --> D
  H[video.mp4] --> I[YoloDetector.detect_frame()]
  I --> J[process_video_first_per_class]
  E --> J
  G --> J
  J --> K[zoe_depth.get_zoe_depth_map() or fallback]
  K --> L[compute_object_world_and_mapped()]
  L --> M[first_seen_detections.csv + debug CSVs]
  M --> N[plot_floorplan() -> PNGs]
```

ASCII alternative:

```
arkitData.json --(poses)--> projection/mapping ----\
                                                      \
                                                       -> process_video (YOLO + Zoe) -> world positions -> CSVs -> PNG
floor_plan.json --(polygons)--> mapping --------------/
video.mp4 --(YOLO)--> detection frames --(Zoe)--> depth -> reprojection -> mapping
```

---

## Practical step-by-step run

1. Prepare inputs:

   * `arkitData.json` (ARKit export).
   * `floor_plan.json` (floor plan GeoJSON-style used by the script).
   * `video.mp4` (recorded video aligned to ARKit frames).

2. Edit `config/config.py`:

   * Set `DATA_DIR` to the folder containing the three inputs (or adjust ARKIT\_PATH / FLOOR\_PATH / VIDEO\_PATH individually).
   * Ensure `YOLO_MODEL` path points to your weights (or leave — YOLO will print an error if missing).
   * Choose `ZOE_MODEL_NAME` or keep default. If using CPU, set `ZOE_DEVICE='cpu'`.

3. (Optional) Create a Python venv and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install numpy pandas matplotlib opencv-python ultralytics transformers torch torchvision pillow
# if GPU desired, install a torch matching your CUDA (see pytorch.org)
```

4. Run:

```bash
python3 main.py
```

5. Check outputs (in `DATA_DIR` or `input_data` depending on config):

   * `first_seen_detections.csv`
   * `first_seen_detections_debug.csv` (if debug saved)
   * `reproject_debug.csv` (if debug saved)
   * `floor_plan_with_emojis.png`, `floor_plan_plain.png`
   * saved crops like `first_person_vf12.jpg` and annotated frames `annotated_first_vf*.jpg`

---

## What each output field means (CSV)

`first_seen_detections.csv` (columns):

* `class`: detected class name from YOLO (e.g., `person`).
* `video_frame_index`: index in the video where class first seen.
* `arkit_index`: matched ARKit frame index used for pose.
* `cam_mapped_x`, `cam_mapped_y`: mapped camera 2D coordinates (on floorplan).
* `object_mapped_x`, `object_mapped_y`: mapped object position (on floorplan) — may be `null` if depth missing.
* `object_yaw_deg`: estimated object yaw (facing angle) in degrees (optional).
* `conf`: detector confidence.
* `crop_path`: file path to saved crop image (if saved).
* `annotated_frame`: saved annotated frame path.
* `depth_model_units`: raw depth units from Zoe (model-specific).
* `distance_m`: depth in meters (or `null` if missing).
* `distance_ft`: depth in feet (if conversion enabled).

`first_seen_detections_debug.csv` (verbose): per detection debug rows used to analyze reprojection errors, camera pose, mapped coords, orientation confidence.

`reproject_debug.csv`: records of actual image u,v and estimated u\_est,v\_est from reprojection for diagnostics.

---

## ZoeDepth issues & troubleshooting (common)

You experienced:

```
Attempting to cast a BatchFeature to type None. This is not supported.
```

**Why**: mismatch between the processor saved with the model (its config) and your `transformers` version or `AutoImageProcessor` expectations (list vs single-image; `use_fast` availability; device issues).

**Fixes** (we already provided a hardened wrapper):

1. Use the new `detectors/zoe_depth.py` (this repo contains it) — it:

   * tries `AutoImageProcessor(..., use_fast=True)` but falls back.
   * tries both `processor(image)` and `processor([image])`.
   * moves tensors to the selected device.
   * returns clear errors if Zoe cannot produce depths.

2. If Zoe still fails:

   * ensure `transformers` and `torch` versions are reasonably recent:

     ```bash
     pip install -U transformers torch
     ```

     (or pick a torch wheel compatible with your GPU driver)
   * set `ZOE_DEVICE = 'cpu'` in `config/config.py` to avoid GPU-driver mismatch during testing.
   * pass a local model checkpoint that matches your `transformers` version.

3. Pipeline fallback:

   * When Zoe fails for a frame, a simple heuristic (`estimate_depth_fallback`) computes an approximate depth from bounding-box height so the pipeline can still compute world coordinates. This lets you get mapped outputs even if Zoe is unavailable — less accurate, but better than nothing.

---

## YOLO & emoji notes

* YOLO (ultralytics) prints model summary on load. If YOLO fails to init, `main.py` returns early — install `ultralytics` and provide weights.
* Emoji labels in PNG are used only if an emoji-capable font is found. If not, the script prints:

  ```
  plot_floorplan: no emoji font found on system — falling back to plain labels
  ```

---

## Example logs and what they indicate

* `Extracted 2380 frames` — ARKit parsing succeeded; positions count = number of camera transforms.
* `proj x,z -> score 2082` — automatic mapping tried multiple projections and chose the highest score (points inside polygons).
* `Global orientation chosen by majority vote: use_transpose=False global_z_sign=-1.0` — script selected camera-rotation orientation direction and sign by voting over frames.
* `Zoe model loaded: Intel/zoedepth-nyu-kitti on cpu` — Zoe loaded and device set.
* `Warning: ZoeDepth failed for vf 1 -> ...` — Zoe failing for a frame; fallback used, or depth becomes None (check CSV).
* `DETECTED-START: person vf12 cam=(384.8,403.3) obj=(N/A,N/A) dist=N/A` — YOLO detected a class but object world could not be computed (depth missing). If fallback depths are used, `obj` and `dist` will be filled.

---

## If object mapped\_x/mapped\_y are `N/A` — check:

1. Was Zoe depth available and non-NaN for that frame? If not, `object_x/object_y` will be `null`.
2. Camera pose (ARKit) must include rotation (`rot`) or a 4x4/3x4 matrix for orientation; missing rotation can prevent raycast.
3. Intrinsics: if ARKit metadata includes camera intrinsics they will be used for reprojection; otherwise a fallback assumed intrinsics are used — results may be poorer.
4. If you want deterministic mapping improvements, provide `CONTROL_POINTS` (2 or more known correspondences). That triggers exact Umeyama similarity.

---

## Quick tips for reproducible results

* Ensure ARKit and video frames are roughly synchronized. `frameNumber` matching in ARKit -> video frame index mapping is attempted; if missing, nearest frame is used.
* Provide camera intrinsics (width/height + fx/fy/cx/cy) in ARKit metadata for better reprojection accuracy.
* For improved depth:

  * Install `torch` and a GPU driver compatible with your GPU + CUDA and pick `ZOE_DEVICE='cuda'`.
  * Or run Zoe on CPU (slower) but more likely to work without driver problems: set `ZOE_DEVICE='cpu'`.

---

## Where to tweak behavior (files)

* `config/config.py` — change paths, devices, classes, debug toggles.
* `detectors/zoe_depth.py` — change fallback heuristic parameters (`near_depth`, `far_depth`).
* `video/video_processing.py` — you can change how first-seen frames are chosen (mapping strategy, saved outputs).
* `floor/plotter.py` — change plotting style, marker sizes, label offsets.

---

