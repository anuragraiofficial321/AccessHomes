#!/usr/bin/env python3
"""
Updated script: detect_object_and_plot_full_updated.py

Changes from your patched version:
 - Adds per-detection debug CSV with p_world, proj2, mapped2, reprojection estimates (u_est,v_est).
 - Adds more verbose debug logging and optional FLAGS at top for easier toggling.
 - Keeps your automatic R vs R.T and z_sign detection.
 - Ensures intrinsics rescaling and units conversion are handled and logged.
 - Does NOT require CLI args (per your request) — edit CONFIG block and run.

Usage:
  python3 detect_object_and_plot_full_updated.py

Outputs (in DATA_DIR):
 - floor_plan_first_detections_only_with_distance.png (overlay image)
 - first_seen_detections.csv (summary)
 - first_seen_detections_debug.csv (detailed per-detection p_world/proj/mapped/reproj)
 - reproject_debug.csv (reprojection small table)

Make sure you have these files present as in your environment:
 - arkitData.json
 - floor_plan.json
 - video.mp4 (optional; script can skip detection if absent)

Requirements (approx): numpy, pandas, matplotlib, opencv-python, ultralytics, transformers, torch, pillow

"""

import json, math, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path as MplPath
import cv2

# optional YOLO helper; import only if available
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# --------------------- CONFIG ---------------------
DATA_DIR = Path("/home/anuragrai/Desktop/Client/AccessMate/output/cubicasa_data")
ARKIT_PATH = DATA_DIR / "arkitData.json"
FLOOR_PATH = DATA_DIR / "floor_plan.json"
VIDEO_PATH = DATA_DIR / "video.mp4"
OUT_PNG = DATA_DIR / "floor_plan_first_detections_only_with_distance.png"
OUT_CSV = DATA_DIR / "first_seen_detections.csv"
OUT_DEBUG_CSV = DATA_DIR / "first_seen_detections_debug.csv"
OUT_REPRO_CSV = DATA_DIR / "reproject_debug.csv"

# YOLO settings
YOLO_MODEL = "models/yolo11n.pt"
YOLO_DEVICE = "cpu"
YOLO_CONF = 0.80
YOLO_IOU = 0.45

# ZoeDepth settings (optional)
ZOE_MODEL_NAME = "Intel/zoedepth-nyu-kitti"
ZOE_DEVICE = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"

# Classes to collect first-seen frames
TARGET_CLASSES = ["person"]

# Projection and units
PROJECTION = "x,-z"
CONVERT_M_TO_FT = False
M_TO_FT = 3.280839895013123

# Control points if you want exact mapping: list of (frame_index, floor_x, floor_y)
CONTROL_POINTS = []

# Debug flags
DEBUG_REPROJECT = True
SAVE_DETECTED_CROPS = True
VERBOSE = True

# ----------------- Helper functions -----------------

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def find_dicts_with_key(obj, key):
    out=[]
    if isinstance(obj, dict):
        if key in obj: out.append(obj)
        for v in obj.values(): out += find_dicts_with_key(v, key)
    elif isinstance(obj, list):
        for it in obj: out += find_dicts_with_key(it, key)
    return out

def extract_candidates_from_raw(raw):
    candidates = []
    try:
        arr = np.array(raw, dtype=float)
    except Exception:
        return candidates
    if arr.ndim == 2 and arr.shape == (4,4):
        M = arr
        candidates.append({"pos": M[3,:3].astype(float), "rot": M[:3,:3].astype(float), "method": "lastrow_first3"})
        candidates.append({"pos": M[:3,3].astype(float), "rot": M[:3,:3].astype(float), "method": "lastcol_first3"})
        candidates.append({"pos": M.T[:3,3].astype(float), "rot": M.T[:3,:3].astype(float), "method": "transpose_lastcol"})
    if arr.ndim == 1 and arr.size == 16:
        flat = arr
        candidates.append({"pos": np.array([flat[3], flat[7], flat[11]], dtype=float), "rot": None, "method": "flat_rowmajor_3x4"})
        candidates.append({"pos": np.array([flat[12], flat[13], flat[14]], dtype=float), "rot": None, "method": "flat_columnmajor"})
    if arr.ndim == 1 and arr.size == 12:
        flat = arr
        candidates.append({"pos": np.array([flat[3], flat[7], flat[11]], dtype=float), "rot": None, "method": "flat_3x4_rowmajor"})
    return candidates

def extract_positions_list():
    ark = load_json(ARKIT_PATH)
    cand_dicts = find_dicts_with_key(ark, "cameraTransform")
    if not cand_dicts:
        cand_dicts = find_dicts_with_key(ark, "transform") + find_dicts_with_key(ark, "matrix") + find_dicts_with_key(ark, "camera_transform")
    entries = []
    for d in cand_dicts:
        rawmat = d.get("cameraTransform") or d.get("transform") or d.get("matrix") or None
        if rawmat is None:
            for k,v in d.items():
                if isinstance(v, list) and v and all(isinstance(x,(int,float)) for x in v):
                    if len(v) in (16,12):
                        rawmat = v; break
        if rawmat is None:
            continue
        cands = extract_candidates_from_raw(rawmat)
        if not cands:
            continue
        chosen = None
        for c in cands:
            if c["method"] == "lastrow_first3":
                chosen = c; break
        if chosen is None:
            chosen = cands[0]
        entries.append({"pos": chosen["pos"], "rot": chosen.get("rot"), "method": chosen["method"], "raw": d,
                        "frameNumber": d.get("frameNumber"), "frameTimestamp": d.get("frameTimestamp")})
    if not entries:
        def find_xy(obj):
            out=[]
            if isinstance(obj, dict):
                if 'x' in obj and 'y' in obj:
                    try:
                        out.append({"pos": np.array([float(obj['x']), 0.0, float(obj['y'])])})
                    except: pass
                for v in obj.values(): out += find_xy(v)
            elif isinstance(obj, list):
                for it in obj: out += find_xy(it)
            return out
        ark_xy = find_xy(ark)
        if ark_xy:
            entries = [{"pos": it["pos"], "rot": None, "method": "xy_direct", "raw": None, "frameNumber": None, "frameTimestamp": None} for it in ark_xy]
    if not entries:
        raise RuntimeError("No camera transforms or x/y data found in arkitData.json")
    positions = np.vstack([e["pos"] for e in entries])
    meta = [{"frameNumber": e.get("frameNumber"), "frameTimestamp": e.get("frameTimestamp"), "method": e.get("method"), "raw": e.get("raw"), "rot": e.get("rot")} for e in entries]
    return positions, meta

def project_3d_to_2d(positions3d, projection):
    positions3d = np.array(positions3d, dtype=float)
    if positions3d.ndim == 1:
        positions3d = positions3d.reshape(1,3)
    if projection == "x,-z":
        return np.column_stack([positions3d[:,0], -positions3d[:,2]])
    if projection == "x,z":
        return np.column_stack([positions3d[:,0], positions3d[:,2]])
    if projection == "-x,-z":
        return np.column_stack([-positions3d[:,0], -positions3d[:,2]])
    if projection == "-x,z":
        return np.column_stack([-positions3d[:,0], positions3d[:,2]])
    if projection == "y,-z":
        return np.column_stack([positions3d[:,1], -positions3d[:,2]])
    raise ValueError("Unknown projection: " + projection)

def umeyama_2d(src, dst, with_scaling=True):
    src = np.array(src, dtype=float)
    dst = np.array(dst, dtype=float)
    assert src.shape == dst.shape and src.shape[1] == 2
    N = src.shape[0]
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    cov = (dst_c.T @ src_c) / N
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1,1] = -1
    R = U @ S @ Vt
    if with_scaling:
        var_src = (src_c**2).sum() / N
        s = 1.0 / var_src * np.trace(np.diag(D) @ S)
    else:
        s = 1.0
    t = mu_dst - s * R @ mu_src
    return s, R, t

def apply_similarity_to_points(points2d, s, R, t):
    pts = np.array(points2d, dtype=float)
    return (s * (R @ pts.T)).T + t

def load_floor_polygons():
    floor = load_json(FLOOR_PATH)
    spaces = []
    if 'floors' in floor and len(floor['floors'])>0:
        for s in floor['floors'][0].get('spaces', []):
            coords = s.get('boundaryPolygon', {}).get('coordinates')
            if coords:
                poly = coords[0] if isinstance(coords[0][0], list) else coords
                spaces.append({"id": s.get('id'), "name": s.get('name') or s.get('class') or s.get('id'), "poly": np.array(poly)})
    if not spaces:
        raise RuntimeError("No spaces/polygons found in floor_plan.json")
    all_pts = np.vstack([sp["poly"] for sp in spaces])
    floor_min = all_pts.min(axis=0); floor_max = all_pts.max(axis=0)
    return spaces, floor_min, floor_max

def count_points_inside_polygons(points2d, spaces):
    cnt = 0
    for p in points2d:
        if np.isnan(p[0]) or np.isnan(p[1]): continue
        for sp in spaces:
            if MplPath(sp["poly"]).contains_point((p[0], p[1])):
                cnt += 1
                break
    return cnt

def auto_map_and_choose(proj2, spaces, floor_min, floor_max, compass=None):
    floor_center = (floor_min + floor_max) / 2.0
    valid_mask = ~np.isnan(proj2[:,0])
    if valid_mask.sum() == 0:
        raise RuntimeError("No valid projected points to map")
    vpts = proj2[valid_mask]
    cam_min = vpts.min(axis=0); cam_max = vpts.max(axis=0)
    cam_span = cam_max - cam_min
    cam_span[cam_span==0] = 1.0
    floor_size = floor_max - floor_min
    scale_x = (floor_size[0]*0.9) / cam_span[0]
    scale_y = (floor_size[1]*0.9) / cam_span[1]
    scale = (scale_x + scale_y) / 2.0
    mapped = (proj2 - (cam_min + cam_max)/2.0) * scale + floor_center
    rotations = [0.0]
    if compass is not None:
        rotations = [0.0, compass, -compass]
    best_score = -1; best_rot = 0.0; best_mapped=None
    for rot in rotations:
        theta = math.radians(rot)
        Rr = np.array([[math.cos(theta), -math.sin(theta)],[math.sin(theta), math.cos(theta)]])
        rel = mapped - floor_center
        mapped_rot = (rel @ Rr.T) + floor_center
        sc = count_points_inside_polygons(mapped_rot, spaces)
        if sc > best_score:
            best_score = sc; best_rot = rot; best_mapped = mapped_rot
    return best_mapped, scale, best_rot, best_score

def interp_missing(mapped):
    m = mapped.copy()
    n = len(m)
    for dim in (0,1):
        arr = m[:,dim]
        isn = np.isnan(arr)
        if isn.any():
            good_idx = np.where(~isn)[0]
            if good_idx.size>0:
                interp_all = np.interp(np.arange(n), good_idx, arr[good_idx])
                m[:,dim] = interp_all
    return m

# ----------------- YOLO wrapper -----------------
class YoloDetector:
    def __init__(self, model_path=YOLO_MODEL, device=YOLO_DEVICE, conf=YOLO_CONF, iou=YOLO_IOU):
        if YOLO is None:
            raise RuntimeError("ultralytics.YOLO not available: install ultralytics to use YOLO detection")
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.device = device
        try:
            self.model.fuse()
        except Exception:
            pass

    def detect_frame(self, frame_bgr):
        img = frame_bgr[..., ::-1]
        results = self.model.predict(source=[img], conf=self.conf, iou=self.iou, device=self.device, verbose=False)
        dets = []
        if not results:
            return dets
        r = results[0]
        if not hasattr(r, "boxes") or r.boxes is None:
            return dets
        for box in r.boxes:
            try:
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
            except Exception:
                try:
                    xyxy = box.xyxy.tolist()[0]
                    conf = float(box.conf.tolist()[0])
                    cls_id = int(box.cls.tolist()[0])
                except Exception:
                    continue
            cls_name = self.model.names.get(cls_id, str(cls_id))
            dets.append({'class_name': cls_name, 'class_id': cls_id, 'conf': conf, 'xyxy': xyxy})
        return dets

# ----------------- ZoeDepth integration -----------------
try:
    from transformers import ZoeDepthForDepthEstimation, AutoImageProcessor
    import torch
    from PIL import Image
    _zoe_processor = None
    _zoe_model = None
    _zoe_device = torch.device("cuda" if torch.cuda.is_available() and ZOE_DEVICE.startswith("cuda") else "cpu")
except Exception as e:
    _zoe_processor = None
    _zoe_model = None
    ZoeDepthForDepthEstimation = None
    AutoImageProcessor = None
    torch = None
    print("ZoeDepth import failed (transformers/torch may not be installed):", e)

def init_zoe(model_name=None, device=None):
    global _zoe_processor, _zoe_model, _zoe_device
    if ZoeDepthForDepthEstimation is None:
        print("ZoeDepth classes not available; cannot init Zoe.")
        return False
    model_name = model_name or ZOE_MODEL_NAME
    device = device or _zoe_device
    try:
        _zoe_processor = AutoImageProcessor.from_pretrained(model_name)
        _zoe_model = ZoeDepthForDepthEstimation.from_pretrained(model_name).to(device)
        _zoe_device = device
        print("Zoe model loaded:", model_name, "on", _zoe_device)
        return True
    except Exception as e:
        print("Failed to init Zoe:", e)
        _zoe_processor = None; _zoe_model = None
        return False

def get_zoe_depth_map(image_rgb):
    global _zoe_processor, _zoe_model, _zoe_device, torch
    if _zoe_processor is None or _zoe_model is None:
        raise RuntimeError("Zoe not initialized. Call init_zoe()")
    pil = Image.fromarray(image_rgb)
    inputs = _zoe_processor(images=pil, return_tensors="pt").to(_zoe_device)
    with torch.no_grad():
        outputs = _zoe_model(**inputs)
    depth_map = outputs.predicted_depth.squeeze().cpu().numpy().astype(np.float32)
    if depth_map.shape != image_rgb.shape[:2]:
        depth_map = cv2.resize(depth_map, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
    return depth_map

# ----------------- Frame mapping helpers -----------------
def build_arkit_frame_index_map(meta_list, num_video_frames=None):
    frame_nums = []
    for m in meta_list:
        fn = None
        if isinstance(m, dict):
            fn = m.get('frameNumber')
            if isinstance(fn, str):
                try:
                    fn = int(fn)
                except:
                    fn = None
        frame_nums.append(fn)
    valid = [fn for fn in frame_nums if fn is not None]
    if len(valid) >= 10:
        fn_to_idx = {fn: idx for idx, fn in enumerate(frame_nums) if fn is not None}
        keys_sorted = sorted(fn_to_idx.keys())
        def map_fn(vf):
            if vf in fn_to_idx:
                return fn_to_idx[vf]
            nearest = min(keys_sorted, key=lambda x: abs(x - vf))
            return fn_to_idx[nearest]
        return map_fn
    else:
        def map_fn(vf):
            return min(vf, len(meta_list)-1)
        return map_fn

def _clamp(v, a, b):
    return max(a, min(b, v))

def get_intrinsics_from_meta(meta_entry):
    if not isinstance(meta_entry, dict):
        return None

    def parse_K(K):
        try:
            arr = np.array(K, dtype=float)
            if arr.shape == (3,3):
                fx = float(arr[0,0]); fy = float(arr[1,1])
                cx = float(arr[0,2]); cy = float(arr[1,2])
                return fx, fy, cx, cy
            if arr.size >= 9:
                arr2 = arr.flatten()[:9].reshape(3,3)
                fx = float(arr2[0,0]); fy = float(arr2[1,1])
                cx = float(arr2[0,2]); cy = float(arr2[1,2])
                return fx, fy, cx, cy
        except Exception:
            return None
        return None

    K = meta_entry.get("cameraIntrinsics", None)
    raw = None
    if K is None:
        raw = meta_entry.get("raw") if isinstance(meta_entry.get("raw"), dict) else None
        if raw is not None:
            K = raw.get("cameraIntrinsics", None)

    parsed = None
    if K is not None:
        parsed = parse_K(K)

    if parsed is None:
        for k, v in meta_entry.items():
            if isinstance(v, list) and len(v) in (9, 12, 16):
                maybe = parse_K(v)
                if maybe is not None:
                    parsed = maybe
                    break
        if parsed is None and raw is not None:
            for k,v in raw.items():
                if isinstance(v, list) and len(v) in (9,12,16):
                    maybe = parse_K(v)
                    if maybe is not None:
                        parsed = maybe
                        break

    if parsed is None:
        return None

    fx, fy, cx, cy = parsed

    K_img_w = None; K_img_h = None
    def try_int(x):
        try:
            return int(np.asarray(x).item())
        except Exception:
            return None

    for key in ("imageWidth", "image_width", "width", "imageWidthPx", "imageWidthPixels"):
        if key in meta_entry:
            K_img_w = try_int(meta_entry[key]); break
    for key in ("imageHeight", "image_height", "height", "imageHeightPx", "imageHeightPixels"):
        if key in meta_entry:
            K_img_h = try_int(meta_entry[key]); break

    if raw is not None and (K_img_w is None or K_img_h is None):
        for key in ("imageWidth", "image_width", "width", "imageWidthPx"):
            if key in raw and K_img_w is None:
                K_img_w = try_int(raw[key]); break
        for key in ("imageHeight", "image_height", "height", "imageHeightPx"):
            if key in raw and K_img_h is None:
                K_img_h = try_int(raw[key]); break

    return (float(fx), float(fy), float(cx), float(cy), K_img_w, K_img_h)

def compute_object_world_and_mapped(cam_pos3d, cam_rot, intrinsics_tuple, bbox_xyxy, depth_val,
                                   chosen_proj, s_map, R_map, t_map, frame_size=None, z_sign=1.0):
    try:
        if cam_pos3d is None or cam_rot is None or depth_val is None:
            return None, None, None, None

        x1, y1, x2, y2 = bbox_xyxy
        u = (x1 + x2) * 0.5
        v = (y1 + y2) * 0.5

        if intrinsics_tuple is None:
            fw, fh = (frame_size if frame_size is not None else (640, 480))
            fx = fy = 0.8 * max(fw, fh)
            cx = fw / 2.0
            cy = fh / 2.0
        else:
            fx, fy, cx, cy, K_img_w, K_img_h = intrinsics_tuple
            if frame_size is not None and K_img_w is not None and K_img_h is not None:
                frame_w, frame_h = frame_size
                try:
                    K_w = int(K_img_w); K_h = int(K_img_h)
                except Exception:
                    K_w = None; K_h = None
                if K_w is not None and K_h is not None and (K_w != frame_w or K_h != frame_h):
                    sx = float(frame_w) / float(K_w)
                    sy = float(frame_h) / float(K_h)
                    fx = fx * sx
                    fy = fy * sy
                    cx = cx * sx
                    cy = cy * sy

        d_cam = np.array([(u - cx) / fx, (v - cy) / fy, float(z_sign)], dtype=float)
        n = np.linalg.norm(d_cam)
        d_cam_n = d_cam / n if n != 0 else d_cam

        R = np.array(cam_rot, dtype=float)
        d_world = R @ d_cam_n
        d_world_norm = np.linalg.norm(d_world)
        if d_world_norm == 0:
            return None, None, None, None
        d_world_dir = d_world / d_world_norm

        p_obj_world = np.array(cam_pos3d, dtype=float) + float(depth_val) * d_world_dir

        obj_proj2 = project_3d_to_2d(p_obj_world.reshape(1,3), chosen_proj)
        obj_mapped = apply_similarity_to_points(obj_proj2, s_map, R_map, t_map)
        obj_x = float(obj_mapped[0,0]); obj_y = float(obj_mapped[0,1])

        yaw_rad = math.atan2(d_world_dir[0], -d_world_dir[2])
        yaw_deg = math.degrees(yaw_rad)

        return obj_x, obj_y, yaw_deg, p_obj_world
    except Exception as e:
        print("compute_object_world_and_mapped error:", e)
        return None, None, None, None

# ----------------- Video processing (updated) -----------------
def process_video_first_per_class(video_path, detector, meta, mapped_plot, positions3d,
                                  s_map, R_map, t_map, chosen_proj,
                                  target_classes, save_detected=True, debug_reproject=False, reproject_csv=None):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    print(f"Video opened: {video_path}, frames: {total} size: {vid_w}x{vid_h}")
    map_fn = build_arkit_frame_index_map(meta, num_video_frames=total)

    target_set = None if target_classes is None else {c.lower() for c in target_classes}
    first_seen = {}
    vf_idx = 0
    saved_frame_idxs = set()
    depth_cache = {}
    repro_records = []
    debug_rows = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_h, frame_w = frame.shape[:2]
        dets = detector.detect_frame(frame)
        if not dets:
            vf_idx += 1
            continue

        for d in dets:
            cls = d['class_name']; cls_l = cls.lower()
            if target_set is not None and cls_l not in target_set:
                continue
            if cls_l in first_seen:
                continue

            a_idx = map_fn(vf_idx)
            a_idx = max(0, min(a_idx, mapped_plot.shape[0]-1))
            mx, my = float(mapped_plot[a_idx,0]), float(mapped_plot[a_idx,1])

            # robust cam_rot extraction & normalization
            cam_pos3d = positions3d[a_idx]
            cam_rot = None
            try:
                if isinstance(meta[a_idx], dict):
                    cam_rot = meta[a_idx].get("rot", None)
                if cam_rot is None:
                    raw = meta[a_idx].get("raw") if isinstance(meta[a_idx].get("raw"), dict) else None
                    if raw is not None:
                        rawmat = raw.get("cameraTransform") or raw.get("transform") or raw.get("matrix") or None
                        if rawmat is None:
                            for k,v in raw.items():
                                if isinstance(v, list) and len(v) in (12,16):
                                    rawmat = v; break
                        if rawmat is not None:
                            cands = extract_candidates_from_raw(rawmat)
                            if cands:
                                chosen_cand = None
                                for cc in cands:
                                    if cc.get("rot") is not None:
                                        chosen_cand = cc; break
                                if chosen_cand is None:
                                    chosen_cand = cands[0]
                                cam_rot = chosen_cand.get("rot")
                if cam_rot is not None:
                    cam_rot = np.array(cam_rot, dtype=float)
                    if cam_rot.shape == (4,4):
                        cam_rot = cam_rot[:3,:3]
                    if cam_rot.shape != (3,3):
                        if cam_rot.T.shape == (3,3):
                            cam_rot = cam_rot.T
                        else:
                            print("Warning: cam_rot found but has unexpected shape", cam_rot.shape, "-> ignoring")
                            cam_rot = None
            except Exception as e:
                print("Warning: error extracting cam_rot:", e)
                cam_rot = None

            # === NEW: auto-detect rotation orientation (R vs R.T) and camera forward sign (+1 or -1) ===
            z_sign = 1.0
            if cam_rot is not None:
                try:
                    neighbor_idx = None
                    if a_idx + 1 < len(positions3d):
                        neighbor_idx = a_idx + 1
                    elif a_idx - 1 >= 0:
                        neighbor_idx = a_idx - 1
                    if neighbor_idx is not None:
                        travel = positions3d[neighbor_idx] - cam_pos3d
                        norm_t = np.linalg.norm(travel)
                        if norm_t > 1e-6:
                            travel_n = travel / norm_t
                            best_score = -10.0
                            best_R = cam_rot.copy()
                            best_sign = 1.0
                            for Rtry in (cam_rot, cam_rot.T):
                                for sign_try in (1.0, -1.0):
                                    fw = Rtry @ np.array([0.0, 0.0, sign_try])
                                    fw_n = fw / (np.linalg.norm(fw) + 1e-12)
                                    score = float(np.dot(fw_n, travel_n))
                                    if score > best_score:
                                        best_score = score
                                        best_R = Rtry.copy()
                                        best_sign = sign_try
                            cam_rot = best_R
                            z_sign = best_sign
                except Exception as e:
                    print("Warning: auto orientation detect failed:", e)
                    z_sign = 1.0

            crop_path = None
            if save_detected and SAVE_DETECTED_CROPS:
                try:
                    DATA_DIR.mkdir(parents=True, exist_ok=True)
                    h_img, w_img = frame.shape[:2]
                    x1, y1, x2, y2 = [int(round(v)) for v in d['xyxy']]
                    x1 = _clamp(x1, 0, w_img-1); y1 = _clamp(y1, 0, h_img-1)
                    x2 = _clamp(x2, 0, w_img-1); y2 = _clamp(y2, 0, h_img-1)
                    if x2 > x1 and y2 > y1:
                        crop = frame[y1:y2, x1:x2]
                        crop_path = DATA_DIR / f"first_{cls_l}_vf{vf_idx}.jpg"
                        cv2.imwrite(str(crop_path), crop)
                        crop_path = str(crop_path)
                except Exception as e:
                    print("Warning: failed to save crop:", e)
                    crop_path = None

            annotated_frame = None
            if save_detected and vf_idx not in saved_frame_idxs:
                try:
                    frame_copy = frame.copy()
                    for dd in dets:
                        xx1, yy1, xx2, yy2 = [int(round(v)) for v in dd['xyxy']]
                        cv2.rectangle(frame_copy, (xx1, yy1), (xx2, yy2), (0,255,0), 2)
                        cv2.putText(frame_copy, f"{dd['class_name']} {dd['conf']:.2f}",
                                    (xx1, max(0, yy1-8)), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (255,255,255), 2)
                    annotated_frame = DATA_DIR / f"annotated_first_vf{vf_idx}.jpg"
                    cv2.imwrite(str(annotated_frame), frame_copy)
                    annotated_frame = str(annotated_frame)
                    saved_frame_idxs.add(vf_idx)
                except Exception as e:
                    print("Warning: failed to save annotated frame:", e)
                    annotated_frame = None

            # ZoeDepth depth
            depth_val, distance_m, distance_ft = None, None, None
            try:
                if vf_idx in depth_cache:
                    depth_map = depth_cache[vf_idx]
                else:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if _zoe_model is None:
                        ok = init_zoe(ZOE_MODEL_NAME, device=None)
                        if not ok:
                            raise RuntimeError("Failed to initialize ZoeDepth model.")
                    depth_map = get_zoe_depth_map(rgb)
                    depth_cache[vf_idx] = depth_map

                x1, y1, x2, y2 = [int(round(v)) for v in d['xyxy']]
                cx_bbox = int(round((x1 + x2) / 2))
                cy_bbox = int(round((y1 + y2) / 2))
                h_img, w_img = depth_map.shape[:2]
                cx_bbox = _clamp(cx_bbox, 0, w_img-1)
                cy_bbox = _clamp(cy_bbox, 0, h_img-1)
                patch = depth_map[max(0, cy_bbox-1):min(h_img-1, cy_bbox+1)+1,
                                  max(0, cx_bbox-1):min(w_img-1, cx_bbox+1)+1]
                depth_val = float(np.median(patch.astype(float))) if patch.size > 0 else float(depth_map[cy_bbox, cx_bbox])
                distance_m = depth_val
                distance_ft = distance_m * M_TO_FT
            except Exception as e:
                print("Warning: ZoeDepth failed for vf", vf_idx, "->", e)

            # compute object world position using pixel ray + intrinsics + rotation
            obj_x, obj_y, obj_yaw = None, None, None
            p_world = None
            try:
                if cam_rot is None:
                    print(f"DEBUG: No cam_rot for arkit index {a_idx}; meta entry keys: {list(meta[a_idx].keys()) if isinstance(meta[a_idx], dict) else type(meta[a_idx])}")
                if distance_m is None:
                    print(f"DEBUG: No depth for vf {vf_idx} (a_idx {a_idx})")

                intr = None
                try:
                    intr = get_intrinsics_from_meta(meta[a_idx])
                    if intr is None and isinstance(meta[a_idx].get("raw"), dict):
                        intr = get_intrinsics_from_meta(meta[a_idx].get("raw"))
                except Exception:
                    intr = None

                if cam_rot is not None and distance_m is not None:
                    frame_size = (frame_w, frame_h)
                    objx, objy, objyaw, pworld = compute_object_world_and_mapped(cam_pos3d, cam_rot, intr, d['xyxy'],
                                                                         distance_m, chosen_proj, s_map, R_map, t_map,
                                                                         frame_size=frame_size, z_sign=z_sign)
                    if objx is not None:
                        obj_x, obj_y, obj_yaw = objx, objy, objyaw
                        p_world = pworld

                        # optional debug reproject: verify p_obj_world projects back near bbox center
                        if debug_reproject:
                            x1, y1, x2, y2 = d['xyxy']
                            u = (x1 + x2) * 0.5; v = (y1 + y2) * 0.5
                            if intr is not None:
                                fx, fy, cx, cy, K_w, K_h = intr
                                if (K_w is not None and K_h is not None) and (frame_w != K_w or frame_h != K_h):
                                    sx = float(frame_w) / float(K_w); sy = float(frame_h) / float(K_h)
                                    fx = fx * sx; fy = fy * sy; cx = cx * sx; cy = cy * sy
                            else:
                                fx = fy = 0.8 * max(frame_w, frame_h)
                                cx = frame_w/2.0; cy = frame_h/2.0
                            d_cam = np.array([(u - cx) / fx, (v - cy) / fy, float(z_sign)], dtype=float)
                            d_cam_n = d_cam / np.linalg.norm(d_cam) if np.linalg.norm(d_cam)!=0 else d_cam
                            R = np.array(cam_rot, dtype=float)
                            d_world = R @ d_cam_n
                            p_obj_world = np.array(cam_pos3d, dtype=float) + float(distance_m) * (d_world / np.linalg.norm(d_world))
                            try:
                                p_cam_est = R.T @ (p_obj_world - np.array(cam_pos3d, dtype=float))
                                u_est = (p_cam_est[0] / p_cam_est[2]) * fx + cx
                                v_est = (p_cam_est[1] / p_cam_est[2]) * fy + cy
                            except Exception:
                                u_est, v_est = None, None
                            repro_records.append({
                                "vf_idx": vf_idx, "class": cls, "u": u, "v": v, "u_est": u_est, "v_est": v_est,
                                "depth": distance_m, "z_sign": z_sign
                            })

                            # save debug row for this detection
                            debug_rows.append({
                                "class": cls,
                                "video_frame_index": vf_idx,
                                "arkit_index": a_idx,
                                "cam_pos_x": float(cam_pos3d[0]), "cam_pos_y": float(cam_pos3d[1]), "cam_pos_z": float(cam_pos3d[2]),
                                "p_world_x": float(p_world[0]) if p_world is not None else None,
                                "p_world_y": float(p_world[1]) if p_world is not None else None,
                                "p_world_z": float(p_world[2]) if p_world is not None else None,
                                "proj2_x": float(p_world[0]) if p_world is not None else None,
                                "proj2_z": float(-p_world[2]) if p_world is not None else None,
                                "mapped_x": obj_x, "mapped_y": obj_y,
                                "u": u, "v": v, "u_est": u_est, "v_est": v_est,
                                "depth_m": distance_m
                            })

            except Exception as e:
                print("Warning: failed to compute object true position:", e)

            # Save info
            info = {
                'class_name': cls,
                'video_frame_index': vf_idx,
                'arkit_index': a_idx,
                'mapped_x': mx,
                'mapped_y': my,
                'object_x': obj_x,
                'object_y': obj_y,
                'object_yaw_deg': obj_yaw,
                'bbox': d['xyxy'],
                'conf': d['conf'],
                'crop_path': crop_path,
                'annotated_frame': annotated_frame,
                'depth_model_units': depth_val,
                'distance_m': distance_m,
                'distance_ft': distance_ft,
            }
            first_seen[cls_l] = info

            dist_print = f"{distance_m:.3f}m" if distance_m is not None else "N/A"
            print(f"DETECTED-START: {cls} vf{vf_idx} cam=({mx:.1f},{my:.1f}) obj=({obj_x if obj_x else 'N/A'},{obj_y if obj_y else 'N/A'}) dist={dist_print} yaw={obj_yaw if obj_yaw else 'N/A'} z_sign={z_sign}")

            if target_set is not None and set(first_seen.keys()) >= target_set:
                break

        if target_set is not None and set(first_seen.keys()) >= target_set:
            break

        vf_idx += 1

    cap.release()

    if debug_reproject and reproject_csv is not None and repro_records:
        try:
            pd.DataFrame(repro_records).to_csv(reproject_csv, index=False)
            print("Saved reprojection debug CSV:", reproject_csv)
        except Exception as e:
            print("Warning: failed to save reprojection CSV:", e)

    if debug_rows:
        try:
            pd.DataFrame(debug_rows).to_csv(OUT_DEBUG_CSV, index=False)
            print("Saved per-detection debug CSV:", OUT_DEBUG_CSV)
        except Exception as e:
            print("Warning: failed to save per-detection debug CSV:", e)

    return first_seen

# ----------------- Main pipeline -----------------
def main():
    print("Loading and extracting ARKit positions...")
    positions3d, meta = extract_positions_list()
    print(f"Extracted {len(positions3d)} frames (3D positions).")

    if CONVERT_M_TO_FT:
        positions3d = positions3d * M_TO_FT
        print("Converted ARKit positions meters -> feet using factor", M_TO_FT)

    proj2_all = project_3d_to_2d(positions3d, PROJECTION)
    print("Projected to 2D (projection=", PROJECTION, "), sample:", proj2_all[:6])

    spaces, floor_min, floor_max = load_floor_polygons()
    floor_center = (floor_min + floor_max) / 2.0
    print("Floor bounds:", floor_min, floor_max)

    if CONTROL_POINTS and len(CONTROL_POINTS) >= 2:
        print("Using CONTROL_POINTS for exact alignment (Umeyama similarity).")
        src = []; dst = []
        for (fi, fx, fy) in CONTROL_POINTS:
            if fi < 0 or fi >= len(proj2_all):
                raise ValueError(f"Control point frame index {fi} out of range (0..{len(proj2_all)-1})")
            src.append(proj2_all[fi]); dst.append([fx, fy])
        src = np.vstack(src); dst = np.vstack(dst)
        s_map, R_map, t_map = umeyama_2d(src, dst, with_scaling=True)
        mapped_all = apply_similarity_to_points(proj2_all, s_map, R_map, t_map)
        chosen_method = f"umeyama_{len(CONTROL_POINTS)}pts"
        rotation_used = 0.0
        chosen_score = count_points_inside_polygons(mapped_all, spaces)
        chosen_proj = PROJECTION
    else:
        print("No control points provided — using automatic mapping heuristic.")
        floor_json = load_json(FLOOR_PATH)
        compass = floor_json.get("compassHeading", None)
        proj_options = ["x,-z","x,z","-x,-z","-x,z","y,-z"]
        best_score = -1; best_choice=None
        for proj in proj_options:
            p2 = project_3d_to_2d(positions3d, proj)
            try:
                mapped, scale, rot_deg, score = auto_map_and_choose(p2, spaces, floor_min, floor_max, compass=compass)
            except Exception as e:
                print(f"proj {proj} failed: {e}")
                continue
            print(f"proj {proj} -> score {score} (rot {rot_deg:.1f}°, scale approx {scale:.2f})")
            if score > best_score:
                best_score = score; best_choice = {"proj": proj, "mapped": mapped, "scale": scale, "rot_deg": rot_deg, "p2": p2}
        if best_choice is None:
            raise RuntimeError("Automatic mapping failed to generate any candidate.")
        proj2_all = best_choice["p2"]
        mapped_all = best_choice["mapped"]
        chosen_method = f"auto_proj_{best_choice['proj']}"
        rotation_used = best_choice["rot_deg"]
        chosen_score = best_score
        chosen_proj = best_choice['proj']
        print("Auto-chosen projection:", best_choice["proj"], "rotation:", rotation_used, "score:", chosen_score)

        try:
            s_map, R_map, t_map = umeyama_2d(proj2_all, mapped_all, with_scaling=True)
        except Exception as e:
            print("Warning: failed to compute umeyama similarity for mapping; falling back. Error:", e)
            s_map, R_map, t_map = 1.0, np.eye(2), np.array([0.0,0.0])

    mapped_plot = interp_missing(mapped_all)
    print("mapped_plot sample:", mapped_plot[:5])

    print("Initializing YOLO detector (this may load weights and take a moment)...")
    detector = None
    try:
        detector = YoloDetector(model_path=YOLO_MODEL, device=YOLO_DEVICE, conf=YOLO_CONF, iou=YOLO_IOU)
    except Exception as e:
        print("Warning: YOLO detector not available or failed to initialize:", e)
        print("If you want detection, install ultralytics and provide YOLO_MODEL weights at", YOLO_MODEL)
        return

    try:
        if ZoeDepthForDepthEstimation is not None and _zoe_model is None:
            init_zoe(ZOE_MODEL_NAME, device=None)
    except Exception as e:
        print("Zoe init warning:", e)

    found_first = {}
    if VIDEO_PATH.exists():
        found_first = process_video_first_per_class(VIDEO_PATH, detector, meta, mapped_plot, positions3d,
                                                    s_map, R_map, t_map, chosen_proj,
                                                    TARGET_CLASSES, save_detected=True,
                                                    debug_reproject=DEBUG_REPROJECT, reproject_csv=OUT_REPRO_CSV)
        print(f"First-seen classes collected: {len(found_first)}")
    else:
        print(f"Video not found at {VIDEO_PATH} -> skipping detection")

    if found_first:
        rows = []
        for cls_l, info in found_first.items():
            rows.append({
                "class": info['class_name'],
                "video_frame_index": info['video_frame_index'],
                "arkit_index": info['arkit_index'],
                "cam_mapped_x": info['mapped_x'],
                "cam_mapped_y": info['mapped_y'],
                "object_mapped_x": info.get('object_x'),
                "object_mapped_y": info.get('object_y'),
                "object_yaw_deg": info.get('object_yaw_deg'),
                "conf": info['conf'],
                "crop_path": info['crop_path'],
                "annotated_frame": info['annotated_frame'],
                "depth_model_units": info.get("depth_model_units"),
                "distance_m": info.get('distance_m'),
                "distance_ft": info.get('distance_ft'),
            })
        df = pd.DataFrame(rows)
        df.to_csv(OUT_CSV, index=False)
        print("Saved detections CSV:", OUT_CSV)
    else:
        print("No first-seen detections to save to CSV.")

    fig, ax = plt.subplots(figsize=(12,10))
    for sp in spaces:
        poly = sp["poly"]
        patch = MplPolygon(poly, closed=True, fill=True, alpha=0.25, edgecolor='black')
        ax.add_patch(patch)
        c = poly.mean(axis=0)
        ax.text(c[0], c[1], sp["name"], fontsize=9, ha='center', va='center')

    markers = ['*','X','P','D','o','s','^','v']
    class_to_marker = {}
    for i, (cls_l, info) in enumerate(sorted(found_first.items())):
        cls = info['class_name']
        if cls not in class_to_marker:
            class_to_marker[cls] = markers[len(class_to_marker) % len(markers)]
        m = class_to_marker[cls]

        ax.scatter([info['mapped_x']], [info['mapped_y']], s=120, marker='x', zorder=20)
        ax.text(info['mapped_x']+4, info['mapped_y']+4, f"Cam vf{info['video_frame_index']}", fontsize=8)

        objx = info.get('object_x', None)
        objy = info.get('object_y', None)
        dist_label = ""
        if info.get("distance_ft") is not None:
            dist_label = f" ({info['distance_ft']:.1f}ft)"
        if objx is not None and objy is not None:
            ax.scatter([objx], [objy], s=220, marker='*', zorder=25)
            ax.text(objx+6, objy+6, f"{cls}{dist_label}", fontsize=9, fontweight='bold')
            dx = objx - info['mapped_x']
            dy = objy - info['mapped_y']
            ax.arrow(info['mapped_x'], info['mapped_y'], dx, dy, head_width=6, length_includes_head=True, fc='k', ec='k', zorder=18)

    ax.set_xlim(floor_min[0]-10, floor_max[0]+10)
    ax.set_ylim(floor_min[1]-10, floor_max[1]+10)
    ax.set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()
    plt.title(f"Floor plan — first-seen per-class with distance/arrow (found {len(found_first)} of {len(TARGET_CLASSES)}) method {chosen_method} rot {rotation_used:.1f}° score {chosen_score}")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)
    print("Saved overlay image:", OUT_PNG)
    print("Done.")

if __name__ == "__main__":
    main()