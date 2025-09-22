import json, math, os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.path import Path as MplPath
import cv2
from ultralytics import YOLO

# --------------------- CONFIG ---------------------
DATA_DIR = Path("/home/anuragrai/Desktop/Client/AccessMate/output/cubicasa_data")
ARKIT_PATH = DATA_DIR / "arkitData.json"
FLOOR_PATH = DATA_DIR / "floor_plan.json"
VIDEO_PATH = DATA_DIR / "video.mp4"
OUT_PNG = DATA_DIR / "floor_plan_first_detections_only.png"
OUT_CSV = DATA_DIR / "first_seen_detections.csv"   # new CSV

YOLO_MODEL = "models/yolo11n.pt"
YOLO_DEVICE = "cpu"
YOLO_CONF = 0.80
YOLO_IOU = 0.45

TARGET_CLASSES = [
  "person","chair","couch","bed","dining table","laptop","monitor",
  "microwave","refrigerator","oven","toaster","sink","potted plant","book"
]

PROJECTION = "x,-z"
CONVERT_M_TO_FT = False
M_TO_FT = 3.280839895013123

CONTROL_POINTS = []

# ----------------- Helper functions (same as before) -----------------
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
    except:
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
    meta = [{"frameNumber": e.get("frameNumber"), "frameTimestamp": e.get("frameTimestamp"), "method": e.get("method"), "raw": e.get("raw")} for e in entries]
    return positions, meta

def project_3d_to_2d(positions3d, projection):
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
    pts = np.array(points2d)
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

# ----------------- Frame mapping logic -----------------
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

# ----------------- Video processing: first occurrence per class (same as before) -----------------
def process_video_first_per_class(video_path, detector, meta, mapped_plot, target_classes, save_detected=True):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"Video opened: {video_path}, frames: {total}")
    map_fn = build_arkit_frame_index_map(meta, num_video_frames=total)

    if target_classes is None:
        target_set = None
    else:
        target_set = {c.lower() for c in target_classes}

    first_seen = {}
    vf_idx = 0
    saved_frame_idxs = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        dets = detector.detect_frame(frame)
        if not dets:
            vf_idx += 1
            continue

        for d in dets:
            cls = d['class_name']
            cls_l = cls.lower()
            if target_set is not None and cls_l not in target_set:
                continue
            if cls_l in first_seen:
                continue

            a_idx = map_fn(vf_idx)
            a_idx = max(0, min(a_idx, mapped_plot.shape[0]-1))
            mx, my = float(mapped_plot[a_idx,0]), float(mapped_plot[a_idx,1])

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
                        cv2.putText(frame_copy, f"{dd['class_name']} {dd['conf']:.2f}", (xx1, max(0, yy1-8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    annotated_frame = DATA_DIR / f"annotated_first_vf{vf_idx}.jpg"
                    cv2.imwrite(str(annotated_frame), frame_copy)
                    annotated_frame = str(annotated_frame)
                    saved_frame_idxs.add(vf_idx)
                except Exception as e:
                    print("Warning: failed to save annotated frame:", e)
                    annotated_frame = None

            info = {
                'class_name': cls,
                'video_frame_index': vf_idx,
                'arkit_index': a_idx,
                'mapped_x': mx,
                'mapped_y': my,
                'bbox': d['xyxy'],
                'conf': d['conf'],
                'crop_path': crop_path,
                'annotated_frame': annotated_frame
            }
            first_seen[cls_l] = info
            print(f"DETECTED-START: {cls} at vf{vf_idx} ar{a_idx} mapped=({mx:.1f},{my:.1f}) conf={d['conf']:.3f} crop={crop_path}")

            if target_set is not None and set(first_seen.keys()) >= target_set:
                break

        if target_set is not None and set(first_seen.keys()) >= target_set:
            break

        vf_idx += 1

    cap.release()
    return first_seen

# ----------------- Main pipeline -----------------
def main():
    print("Loading and extracting ARKit positions...")
    positions3d, meta = extract_positions_list()
    print(f"Extracted {len(positions3d)} frames (3D positions).")

    if CONVERT_M_TO_FT:
        positions3d = positions3d * M_TO_FT
        print("Converted ARKit positions meters -> feet using factor", M_TO_FT)

    proj2 = project_3d_to_2d(positions3d, PROJECTION)
    print("Projected to 2D (projection=", PROJECTION, "), sample:", proj2[:6])

    spaces, floor_min, floor_max = load_floor_polygons()
    floor_center = (floor_min + floor_max) / 2.0
    print("Floor bounds:", floor_min, floor_max)

    if CONTROL_POINTS and len(CONTROL_POINTS) >= 2:
        print("Using CONTROL_POINTS for exact alignment (Umeyama similarity).")
        src = []; dst = []
        for (fi, fx, fy) in CONTROL_POINTS:
            if fi < 0 or fi >= len(proj2):
                raise ValueError(f"Control point frame index {fi} out of range (0..{len(proj2)-1})")
            src.append(proj2[fi]); dst.append([fx, fy])
        src = np.vstack(src); dst = np.vstack(dst)
        s, R, t = umeyama_2d(src, dst, with_scaling=True)
        mapped_all = apply_similarity_to_points(proj2, s, R, t)
        chosen_method = f"umeyama_{len(CONTROL_POINTS)}pts"
        rotation_used = 0.0
        chosen_score = count_points_inside_polygons(mapped_all, spaces)
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
        proj2 = best_choice["p2"]
        mapped_all = best_choice["mapped"]
        chosen_method = f"auto_proj_{best_choice['proj']}"
        rotation_used = best_choice["rot_deg"]
        chosen_score = best_score
        print("Auto-chosen projection:", best_choice["proj"], "rotation:", rotation_used, "score:", chosen_score)

    mapped_plot = interp_missing(mapped_all)

    print("Initializing YOLO detector (this may load weights and take a moment)...")
    detector = YoloDetector(model_path=YOLO_MODEL, device=YOLO_DEVICE, conf=YOLO_CONF, iou=YOLO_IOU)

    found_first = {}
    if VIDEO_PATH.exists():
        found_first = process_video_first_per_class(VIDEO_PATH, detector, meta, mapped_plot, TARGET_CLASSES, save_detected=True)
        print(f"First-seen classes collected: {len(found_first)}")
    else:
        print(f"Video not found at {VIDEO_PATH} -> skipping detection")

    # Save detections CSV
    if found_first:
        rows = []
        for cls_l, info in found_first.items():
            rows.append({
                "class": info['class_name'],
                "video_frame_index": info['video_frame_index'],
                "arkit_index": info['arkit_index'],
                "mapped_x": info['mapped_x'],
                "mapped_y": info['mapped_y'],
                "conf": info['conf'],
                "crop_path": info['crop_path'],
                "annotated_frame": info['annotated_frame']
            })
        df = pd.DataFrame(rows)
        df.to_csv(OUT_CSV, index=False)
        print("Saved detections CSV:", OUT_CSV)
    else:
        print("No first-seen detections to save to CSV.")

    # ---------- PLOT ONLY the detected first-seen points ----------
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
        ax.scatter([info['mapped_x']], [info['mapped_y']], s=220, marker=m, zorder=15)
        ax.text(info['mapped_x']+6, info['mapped_y']+6, f"{cls} vf{info['video_frame_index']}", fontsize=9, fontweight='bold')

    ax.set_xlim(floor_min[0]-10, floor_max[0]+10)
    ax.set_ylim(floor_min[1]-10, floor_max[1]+10)
    ax.set_aspect('equal', adjustable='box')
    plt.gca().invert_yaxis()
    plt.title(f"Floor plan — first-seen per-class (found {len(found_first)} of {len(TARGET_CLASSES)}) method {chosen_method} rot {rotation_used:.1f}° score {chosen_score}")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)
    print("Saved overlay image:", OUT_PNG)
    print("Done.")

if __name__ == "__main__":
    main()
