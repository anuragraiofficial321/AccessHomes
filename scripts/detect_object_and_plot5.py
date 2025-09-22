#!/usr/bin/env python3
"""
Updated script: detect_object_and_plot_full_updated.py

This version integrates a conservative fallback so arrows and object markers are drawn on the floor plan
even when exact reprojected object coordinates are missing.
Only the plotting stage was changed/added — detection and depth logic left as-is.
"""

import json, math, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon, Patch
from matplotlib.path import Path as MplPath
import cv2

# optional YOLO helper; import only if available
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# ---> ADD THIS: make Depth-Anything-V2 repo importable (adjust path if needed)
DA2_PATH = "/home/anuragrai/Desktop/Client/AccessMate/Depth-Anything-V2"
if os.path.isdir(DA2_PATH) and DA2_PATH not in sys.path:
    sys.path.insert(0, DA2_PATH)
# <--- end add

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
YOLO_CONF = 0.75
YOLO_IOU = 0.45

# Classes to collect first-seen frames
TARGET_CLASSES = ["couch","person"]

# Projection and units
PROJECTION = "x,-z"
CONVERT_M_TO_FT = True
M_TO_FT = 3.280839895013123

# Control points if you want exact mapping: list of (frame_index, floor_x, floor_y)
CONTROL_POINTS = []

# Debug flags
DEBUG_REPROJECT = True
SAVE_DETECTED_CROPS = True
VERBOSE = True

# ----------------- Fallback plotting params (new) -----------------
# How many "floor-plan units" per meter of depth to place estimated objects.
# You may need to tune this for your dataset. It's a heuristic only for visualization.
DEPTH_TO_PLANE_SCALE = 6.0
# Minimum visual offset so arrows are visible
MIN_EST_DISTANCE = 1.0

# ----------------- Helper functions -----------------

def load_fixed_furniture():
    """
    Load fixed furniture bounding polygons from floor_plan.json.

    Returns:
      list of dicts: [{'class': 'Closet', 'poly': np.array([...]), 'space': 'BEDROOM'}, ...]
    """
    floor = load_json(FLOOR_PATH)
    furn = []
    if 'floors' not in floor or not floor['floors']:
        return furn
    # pick first floor (your file format uses floors[0])
    floor0 = floor['floors'][0]
    for s in floor0.get('spaces', []):
        space_name = s.get('name') or s.get('class') or s.get('id')
        # common keys: fixedFurniture or fixed_furniture
        ff_list = s.get('fixedFurniture') or s.get('fixed_furniture') or []
        for f in ff_list:
            bp = f.get('boundingPolygon') or f.get('bounding_polygon') or None
            poly = None
            if bp:
                coords = bp.get('coordinates')
                if coords:
                    # coordinates sometimes stored as [[ [x,y], ... ]] or as [ [x,y], ... ]
                    try:
                        if isinstance(coords[0][0], list):
                            poly = np.array(coords[0], dtype=float)
                        else:
                            poly = np.array(coords, dtype=float)
                    except Exception:
                        try:
                            poly = np.array(coords, dtype=float)
                        except Exception:
                            poly = None
            if poly is None:
                # try alternative bounding polygon fields (some files have 'boundingPolygon' nested differently)
                continue
            furn.append({
                'class': f.get('class') or f.get('type') or 'fixedFurniture',
                'poly': poly,
                'space': space_name
            })
    return furn

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
    """
    Loads spaces and fixed furniture from floor_plan.json.

    Returns:
      spaces: list of {"id","name","poly": np.array([...])}
      floor_min, floor_max: 2D min/max across all space polygons
      fixed_furniture: list of {"class","poly": np.array([...]), "space_id": optional}
    """
    floor = load_json(FLOOR_PATH)
    spaces = []
    fixed_furniture = []
    floors_list = floor.get('floors', []) or []
    # take first floor by default
    if not floors_list:
        raise RuntimeError("No floors found in floor_plan.json")
    floor0 = floors_list[0]
    for s in floor0.get('spaces', []):
        coords = None
        bp = s.get('boundaryPolygon') or {}
        if bp:
            coords = bp.get('coordinates')
            # coords sometimes is a single polygon with list of points
            if coords:
                poly = coords[0] if isinstance(coords[0][0], list) else coords
                poly_arr = np.array(poly, dtype=float)
                spaces.append({"id": s.get('id'), "name": s.get('name') or s.get('class') or s.get('id'), "poly": poly_arr})
        # fixed furniture inside this space
        ff = s.get('fixedFurniture', []) or []
        for f in ff:
            bpoly = None
            b = f.get('boundingPolygon') or {}
            if b:
                coords_f = b.get('coordinates')
                if coords_f:
                    polyf = coords_f[0] if isinstance(coords_f[0][0], list) else coords_f
                    try:
                        polyf_arr = np.array(polyf, dtype=float)
                        fixed_furniture.append({"class": f.get('class') or f.get('type') or "furniture", "poly": polyf_arr, "space_id": s.get('id')})
                    except Exception:
                        continue
    # fallback: try top-level fixedFurniture if any (rare)
    top_ff = floor.get('fixedFurniture', []) or []
    for f in top_ff:
        b = f.get('boundingPolygon') or {}
        if b:
            coords_f = b.get('coordinates')
            if coords_f:
                polyf = coords_f[0] if isinstance(coords_f[0][0], list) else coords_f
                try:
                    polyf_arr = np.array(polyf, dtype=float)
                    fixed_furniture.append({"class": f.get('class') or f.get('type') or "furniture", "poly": polyf_arr, "space_id": None})
                except Exception:
                    continue

    if not spaces:
        raise RuntimeError("No spaces/polygons found in floor_plan.json")

    all_pts = np.vstack([sp["poly"] for sp in spaces])
    floor_min = all_pts.min(axis=0); floor_max = all_pts.max(axis=0)
    return spaces, floor_min, floor_max, fixed_furniture

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

# ----------------- Depth-Anything V2 integration (replacement for ZoeDepth) -----------------
DEPTH_ENCODER = "vitl"  # choose: 'vits', 'vitb', 'vitl', 'vitg' -> pick vitl for good quality/speed tradeoff
DEPTH_CHECKPOINT_FILE = "checkpoints/depth_anything_v2_vitl.pth"  # update to match downloaded checkpoint filename
# Force CPU to avoid CUDA driver mismatch warning
DEPTH_DEVICE = 'cpu'

_depth_model = None
_depth_device = None

try:
    # Try to import the model class from a local clone or installed package
    # If you cloned Depth-Anything-V2, python path should allow: depth_anything_v2.dpt.DepthAnythingV2
    from depth_anything_v2.dpt import DepthAnythingV2
    import torch
    import numpy as _np
    import cv2 as _cv2
    _depth_import_ok = True
except Exception as _e:
    DepthAnythingV2 = None
    torch = None
    _depth_import_ok = False
    print("Depth-Anything V2 import failed (depth_anything_v2 or torch not available):", _e)

# model configuration dicts used when creating DepthAnythingV2
_model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

def init_depth_anything(encoder=None, checkpoint_path=None, device=None):
    """
    Initialize Depth-Anything V2 model. Returns True on success.
    - encoder: one of 'vits','vitb','vitl','vitg'
    - checkpoint_path: local .pth file path
    - device: torch device string (e.g., 'cuda','cpu','mps')
    """
    global _depth_model, _depth_device
    if DepthAnythingV2 is None or torch is None:
        print("Depth-Anything V2 classes not available; cannot init depth model.")
        return False
    encoder = encoder or DEPTH_ENCODER
    ckpt = checkpoint_path or DEPTH_CHECKPOINT_FILE
    device = device or DEPTH_DEVICE
    if encoder not in _model_configs:
        print("Unknown encoder:", encoder)
        return False
    cfg = _model_configs[encoder]
    try:
        # build model and load checkpoint
        model = DepthAnythingV2(**cfg)
        # load checkpoint (map to cpu first then move to device)
        state = torch.load(ckpt, map_location='cpu')
        # if state is wrapped in dict['model'] or similar, attempt to find the state_dict
        sd = state
        if isinstance(state, dict):
            # common keys
            for k in ('state_dict','model','model_state_dict'):
                if k in state:
                    sd = state[k]; break
        if hasattr(model, "load_state_dict"):
            model.load_state_dict(sd)
        else:
            # fallback naive assign
            model.load_state_dict(sd)
        dev = torch.device(device)
        model = model.to(dev).eval()
        _depth_model = model
        _depth_device = dev
        print(f"Depth-Anything V2 model loaded: encoder={encoder}, ckpt={ckpt}, device={dev}")
        return True
    except Exception as e:
        print("Failed to init Depth-Anything V2:", e)
        _depth_model = None
        return False

def get_depthanything_depth_map(image_rgb):
    """
    image_rgb: HxWx3 uint8 numpy (RGB)
    Returns: depth_map as HxW numpy float32 in same pixel resolution as input (resized back if needed)
    """
    global _depth_model, _depth_device
    if _depth_model is None:
        raise RuntimeError("Depth-Anything model not initialized. Call init_depth_anything()")
    # DepthAnythingV2 typically expects numpy BGR or RGB depending on repo impl; the official snippet used raw_img as cv2 BGR
    # We will pass RGB as the original repo snippet did in your prompt.
    img = image_rgb
    # call the model's inference helper (the repo provides a method named 'infer_image' per snippet)
    try:
        raw_depth = _depth_model.infer_image(img)  # returns HxW numpy (float) in relative units
    except Exception as e:
        # sometimes the repo expects BGR; try BGR if RGB call fails
        try:
            alt = image_rgb[..., ::-1]
            raw_depth = _depth_model.infer_image(alt)
        except Exception as e2:
            raise RuntimeError(f"Depth-Anything inference failed (tried RGB and BGR): {e} / {e2}")
    # match output shape to input shape if needed
    if raw_depth is None:
        raise RuntimeError("Depth-Anything returned None depth map")
    # ensure numpy dtype float32
    if isinstance(raw_depth, (list,tuple)):
        raw_depth = _np.array(raw_depth, dtype=_np.float32)
    if raw_depth.shape != img.shape[:2]:
        # resize to exact input size (use linear)
        raw_depth = _cv2.resize(raw_depth.astype(_np.float32), (img.shape[1], img.shape[0]), interpolation=_cv2.INTER_LINEAR)
    return raw_depth.astype(_np.float32)


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

# ----------------- Video processing (kept mostly unchanged) -----------------
def process_video_first_per_class(video_path, detector, meta, mapped_plot, positions3d,
                                  s_map, R_map, t_map, chosen_proj,
                                  target_classes, save_detected=True, debug_reproject=False, reproject_csv=None,
                                  global_use_transpose=None, global_z_sign=None):
    """
    Processes the video and returns a dict of first-seen detections per class.

    Arguments mirror the main script. New args:
      global_use_transpose: if True/False, force using cam_rot.T (True) or cam_rot as-is (False).
      global_z_sign: if +1.0 or -1.0, force the camera forward sign. If None, per-frame auto-detect is used.

    Returns:
      first_seen: dict mapping lower-case class -> info dict (same shape used by main()).
    """
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

    # Local alias for convenience
    save_detected = save_detected and SAVE_DETECTED_CROPS

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_h, frame_w = frame.shape[:2]
        dets = []
        try:
            dets = detector.detect_frame(frame)
        except Exception as e:
            print("Warning: detector failed on frame", vf_idx, "->", e)
            dets = []

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

            # === GLOBAL override application: if provided, use the global transpose choice and z_sign ===
            z_sign = 1.0
            if cam_rot is not None and (global_use_transpose is not None or global_z_sign is not None):
                try:
                    # apply transpose override if asked
                    if global_use_transpose is True:
                        cam_rot = cam_rot.T if cam_rot is not None else cam_rot
                    elif global_use_transpose is False:
                        cam_rot = cam_rot
                    # apply global z sign if provided
                    if global_z_sign is not None:
                        z_sign = float(global_z_sign)
                    else:
                        z_sign = 1.0
                except Exception as e:
                    print("Warning: applying global orientation override failed:", e)
                    z_sign = 1.0
            else:
                # legacy per-frame neighbor-based auto-detect
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
            if save_detected:
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

            # Depth-Anything depth (replacement for Zoe block)
            depth_val, distance_m, distance_ft = None, None, None
            try:
                if vf_idx in depth_cache:
                    depth_map = depth_cache[vf_idx]
                else:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # init model lazily
                    if _depth_model is None:
                        ok = init_depth_anything(encoder=DEPTH_ENCODER, checkpoint_path=DEPTH_CHECKPOINT_FILE, device=None)
                        if not ok:
                            raise RuntimeError("Failed to initialize Depth-Anything V2 model.")
                    # get depth map (HxW float)
                    depth_map = get_depthanything_depth_map(rgb)
                    depth_cache[vf_idx] = depth_map

                # sample central bbox area similar to previous logic
                x1, y1, x2, y2 = [int(round(v)) for v in d['xyxy']]
                cx_bbox = int(round((x1 + x2) / 2))
                cy_bbox = int(round((y1 + y2) / 2))
                h_img, w_img = depth_map.shape[:2]
                cx_bbox = _clamp(cx_bbox, 0, w_img-1)
                cy_bbox = _clamp(cy_bbox, 0, h_img-1)
                patch = depth_map[max(0, cy_bbox-1):min(h_img-1, cy_bbox+1)+1,
                                max(0, cx_bbox-1):min(w_img-1, cx_bbox+1)+1]
                depth_val = float(np.median(patch.astype(float))) if patch.size > 0 else float(depth_map[cy_bbox, cx_bbox])
                # Depth-Anything produces **relative** depth units (same as Zoe previously). We treat it the same: 'distance_m = depth_val'
                distance_m = depth_val
                distance_ft = distance_m * M_TO_FT
            except Exception as e:
                print("Warning: Depth-Anything failed for vf", vf_idx, "->", e)


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


# ----------------- New: fallback estimator for plotting -----------------
def estimate_object_if_missing(info, floor_center):
    """
    Return (objx,objy) for this detection info. If object_x/object_y present, return them.
    Otherwise try simple fallbacks in order:
      1) If distance_m present and nonzero: place the object along the vector from camera mapped
         point to the floor_center scaled by DEPTH_TO_PLANE_SCALE*distance_m.
      2) If object_yaw_deg present: small offset in yaw direction.
      3) Default small offset downward so an arrow is visible.
    """
    if info.get('object_x') is not None and info.get('object_y') is not None:
        return float(info['object_x']), float(info['object_y'])

    camx = float(info.get('mapped_x', floor_center[0]))
    camy = float(info.get('mapped_y', floor_center[1]))

    # 1) depth-based fallback
    d_m = info.get('distance_m')
    if d_m is not None:
        try:
            d_m = float(d_m)
            offset = max(MIN_EST_DISTANCE, DEPTH_TO_PLANE_SCALE * d_m)
            dir_vec = np.array(floor_center) - np.array([camx, camy])
            n = np.linalg.norm(dir_vec)
            if n < 1e-6:
                dir_unit = np.array([0.0, -1.0])
            else:
                dir_unit = dir_vec / n
            obj = np.array([camx, camy]) + dir_unit * offset
            return float(obj[0]), float(obj[1])
        except Exception:
            pass

    # 2) yaw-based fallback
    yaw = info.get('object_yaw_deg')
    if yaw is not None:
        try:
            ang = np.deg2rad(float(yaw))
            offset = max(MIN_EST_DISTANCE, 2.0)
            dx = np.sin(ang) * offset
            dy = -np.cos(ang) * offset
            return camx + dx, camy + dy
        except Exception:
            pass

    # 3) default small offset downward
    return camx, camy - 2.0

# ----------------- Main pipeline (updated plotting to draw fixed furniture and fallback) -----------------
def main():
    print("Loading and extracting ARKit positions...")
    positions3d, meta = extract_positions_list()
    print(f"Extracted {len(positions3d)} frames (3D positions).")

    if CONVERT_M_TO_FT:
        positions3d = positions3d * M_TO_FT
        print("Converted ARKit positions meters -> feet using factor", M_TO_FT)

    proj2_all = project_3d_to_2d(positions3d, PROJECTION)
    print("Projected to 2D (projection=", PROJECTION, "), sample:", proj2_all[:6])

    # ----------------- FIX: load floor polygons BEFORE mapping attempt -----------------
    try:
        floor_load_result = load_floor_polygons()
    except Exception as e:
        raise RuntimeError(f"Failed to load floor_plan.json before mapping: {e}")

    if isinstance(floor_load_result, (list, tuple)):
        if len(floor_load_result) < 3:
            raise RuntimeError(f"load_floor_polygons() returned a sequence of length {len(floor_load_result)}; expected >=3")
        spaces = floor_load_result[0]
        floor_min = np.asarray(floor_load_result[1], dtype=float)
        floor_max = np.asarray(floor_load_result[2], dtype=float)
        # possible fourth element fixed_furniture
        fixed_furniture = floor_load_result[3] if len(floor_load_result) > 3 else []
    elif isinstance(floor_load_result, dict):
        if 'spaces' in floor_load_result and 'floor_min' in floor_load_result and 'floor_max' in floor_load_result:
            spaces = floor_load_result['spaces']
            floor_min = np.asarray(floor_load_result['floor_min'], dtype=float)
            floor_max = np.asarray(floor_load_result['floor_max'], dtype=float)
            fixed_furniture = floor_load_result.get('fixed_furniture', [])
        else:
            if 'spaces' in floor_load_result:
                spaces = floor_load_result['spaces']
                all_pts = np.vstack([sp['poly'] for sp in spaces])
                floor_min = all_pts.min(axis=0)
                floor_max = all_pts.max(axis=0)
            else:
                raise RuntimeError("load_floor_polygons() returned a dict but it lacks expected keys ('spaces' / 'floor_min' / 'floor_max').")
            fixed_furniture = []
    else:
        raise RuntimeError(f"Unexpected return type from load_floor_polygons(): {type(floor_load_result)}")

    floor_center = (floor_min + floor_max) / 2.0
    print("Floor bounds:", floor_min, floor_max)

    # ---- ADDED: load fixed furniture polygons to draw on the overlay (if function exists) ----
    try:
        # if load_fixed_furniture returns extra items, prefer that dataset
        more_fix = load_fixed_furniture()
        if more_fix:
            fixed_furniture = more_fix
    except Exception as e:
        print("Warning: load_fixed_furniture failed:", e)
        # keep previously extracted fixed_furniture if any

    print(f"Loaded {len(fixed_furniture)} fixed furniture items from {FLOOR_PATH}")

    # mapping as before...
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

    # -------------------- GLOBAL orientation voting --------------------
    print("Computing global orientation votes (R vs R.T and forward z sign) across frames...")
    votes = []
    for i in range(len(positions3d)):
        # extract cam_rot candidate for this meta entry
        cam_rot = None
        try:
            if isinstance(meta[i], dict):
                cam_rot = meta[i].get("rot", None)
            if cam_rot is None:
                raw = meta[i].get("raw") if isinstance(meta[i].get("raw"), dict) else None
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
                        cam_rot = None
        except Exception:
            cam_rot = None

        if cam_rot is None:
            continue

        # neighbor displacement -> travel dir
        neighbor_idx = None
        if i+1 < len(positions3d):
            neighbor_idx = i+1
        elif i-1 >= 0:
            neighbor_idx = i-1
        if neighbor_idx is None:
            continue
        travel = positions3d[neighbor_idx] - positions3d[i]
        norm_t = np.linalg.norm(travel)
        if norm_t < 1e-6:
            continue
        travel_n = travel / norm_t

        # evaluate 4 choices and pick best
        best_score = -10.0
        best_choice = (False, 1.0)  # (use_transpose, z_sign)
        for use_transpose in (False, True):
            Rtry = cam_rot.T if use_transpose else cam_rot
            for sign_try in (1.0, -1.0):
                fw = Rtry @ np.array([0.0, 0.0, sign_try])
                fw_n = fw / (np.linalg.norm(fw) + 1e-12)
                score = float(np.dot(fw_n, travel_n))
                if score > best_score:
                    best_score = score
                    best_choice = (use_transpose, sign_try)
        votes.append(best_choice)

    # majority vote
    if votes:
        from collections import Counter
        cnt = Counter(votes)
        global_use_transpose, global_z_sign = cnt.most_common(1)[0][0]
        print(f"Global orientation chosen by majority vote: use_transpose={global_use_transpose} global_z_sign={global_z_sign} (votes {len(votes)})")
    else:
        global_use_transpose, global_z_sign = None, None
        print("Global orientation vote: insufficient data -> leaving per-frame auto-detect active")

    # -------------------- Detector init --------------------
    print("Initializing YOLO detector (this may load weights and take a moment)...")
    detector = None
    try:
        detector = YoloDetector(model_path=YOLO_MODEL, device=YOLO_DEVICE, conf=YOLO_CONF, iou=YOLO_IOU)
    except Exception as e:
        print("Warning: YOLO detector not available or failed to initialize:", e)
        print("If you want detection, install ultralytics and provide YOLO_MODEL weights at", YOLO_MODEL)
        return

    # Initialize Depth-Anything lazily (we'll still init per-frame when needed)
    try:
        # try to init once so model ready before heavy YOLO loop (optional)
        if _depth_model is None:
            ok = init_depth_anything(encoder=DEPTH_ENCODER, checkpoint_path=DEPTH_CHECKPOINT_FILE, device='cpu')
            if not ok:
                print("Depth-Anything init returned False (check checkpoint path). Will attempt lazy init in per-frame loop.")
    except Exception as e:
        print("Depth-Anything init warning:", e)


    found_first = {}
    if VIDEO_PATH.exists():
        # Pass global choices into processing
        found_first = process_video_first_per_class(VIDEO_PATH, detector, meta, mapped_plot, positions3d,
                                                    s_map, R_map, t_map, chosen_proj,
                                                    TARGET_CLASSES, save_detected=True,
                                                    debug_reproject=DEBUG_REPROJECT, reproject_csv=OUT_REPRO_CSV,
                                                    global_use_transpose=global_use_transpose, global_z_sign=global_z_sign)
        print(f"First-seen classes collected: {len(found_first)}")
    else:
        print(f"Video not found at {VIDEO_PATH} -> skipping detection")

    # rest of main unchanged (saving CSV, plotting)...
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

    # ---- ADDED: draw fixed furniture (hatched polygons + label) ----
    for ff in fixed_furniture:
        try:
            fpoly = ff['poly']
            patch = MplPolygon(fpoly, closed=True, fill=True, facecolor='none',
                               edgecolor='saddlebrown', linewidth=1.2, hatch='////', zorder=22)
            ax.add_patch(patch)
            cent = fpoly.mean(axis=0)
            ax.text(cent[0], cent[1], ff['class'], fontsize=7, ha='center', va='center',
                    zorder=23, color='saddlebrown', bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5))
        except Exception as e:
            print("Warning: failed to draw fixed furniture entry:", e)
    try:
        fpatch = Patch(facecolor='none', edgecolor='saddlebrown', hatch='////', label='fixed furniture')
        ax.legend(handles=[fpatch], loc='upper right', fontsize=8)
    except Exception:
        pass

    markers = ['*','X','P','D','o','s','^','v']
    class_to_marker = {}

    # track which detections used fallback
    fallback_info = {}

    for i, (cls_l, info) in enumerate(sorted(found_first.items())):
        cls = info['class_name']
        if cls not in class_to_marker:
            class_to_marker[cls] = markers[len(class_to_marker) % len(markers)]
        m = class_to_marker[cls]

        camx = float(info['mapped_x'])
        camy = float(info['mapped_y'])
        ax.scatter([camx], [camy], s=120, marker='x', zorder=20)
        ax.text(camx+4, camy+4, f"Cam vf{info['video_frame_index']}", fontsize=8)

        objx = info.get('object_x', None)
        objy = info.get('object_y', None)
        used_fallback = False
        if objx is None or objy is None:
            # compute fallback and mark it for debugging
            try:
                estx, esty = estimate_object_if_missing(info, floor_center)
                objx, objy = estx, esty
                used_fallback = True
            except Exception as e:
                print("Warning: fallback estimator failed for", cls_l, "->", e)
                objx, objy = camx, camy - 2.0
                used_fallback = True

        dist_label = ""
        if info.get("distance_ft") is not None:
            try:
                dist_label = f" ({info['distance_ft']:.1f}ft)"
            except Exception:
                dist_label = ""

        # draw object marker and arrow
        ax.scatter([objx], [objy], s=220, marker='*', zorder=25)
        ax.text(objx+6, objy+6, f"{cls}{dist_label}", fontsize=9, fontweight='bold')
        dx = objx - camx
        dy = objy - camy
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            dy = -2.0
        ax.arrow(camx, camy, dx, dy, head_width=6, length_includes_head=True, fc='k', ec='k', zorder=18)

        if used_fallback:
            fallback_info[cls_l] = {"est_x": objx, "est_y": objy, "cam_x": camx, "cam_y": camy}

    # Optionally annotate which detections were estimated
    if fallback_info:
        y = floor_min[1] - 5.0
        ax.text(floor_min[0], y, f"Estimated positions used for: {', '.join(sorted(fallback_info.keys()))}", fontsize=8, color='red', va='top')

    ax.set_xlim(floor_min[0]-10, floor_max[0]+10)
    ax.set_ylim(floor_min[1]-10, floor_max[1]+10)
    ax.set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()
    plt.title(f"Floor plan — first-seen per-class with distance/arrow (found {len(found_first)} of {len(TARGET_CLASSES)}) method {chosen_method} rot {rotation_used:.1f}° score {chosen_score}")
    plt.tight_layout()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=200)
    print("Saved overlay image:", OUT_PNG)
    print("Done.")



if __name__=="__main__":
    main()
