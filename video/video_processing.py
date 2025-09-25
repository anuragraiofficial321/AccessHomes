#!/usr/bin/env python3
import cv2
from config import logger
import numpy as np
import pandas as pd
import math
from config.config import VERBOSE
from pathlib import Path
from matplotlib.path import Path as MplPath
from utils.intrinsics_utils import compute_real_size_from_bbox

_clamp = lambda v, a, b: max(a, min(b, v))

#---------------Setup Logger-----------------
logger = logger.setup_logger()

# --------------- Clamp helper -----------------
def clamp_object_to_floor(spaces, cam_x, cam_y, obj_x, obj_y, eps=1e-2):
    """
    If (obj_x,obj_y) lies outside all floor polygons:
      - Find intersections of the segment cam->obj with polygon edges and clamp
        to the last intersection (just inside boundary).
      - If no intersections, snap to the closest boundary point of the nearest polygon.
    Returns (x_clamped, y_clamped, was_clamped: bool, used_poly_index: int or None).
    """
    P_cam = np.array([float(cam_x), float(cam_y)], dtype=float)
    P_obj = np.array([float(obj_x), float(obj_y)], dtype=float)
    d = P_obj - P_cam
    d_norm = np.linalg.norm(d) + 1e-12

    # Inside any polygon?
    for i, sp in enumerate(spaces):
        poly = np.asarray(sp.get("poly"), dtype=float)
        if poly.ndim == 2 and poly.shape[1] == 2:
            if MplPath(poly).contains_point((P_obj[0], P_obj[1])):
                return float(P_obj[0]), float(P_obj[1]), False, i

    # Line segment intersection helper
    def _seg_seg_intersection(p, r, q, s):
        def _cross(a, b): return a[0]*b[1] - a[1]*b[0]
        rxs = _cross(r, s)
        qmp = q - p
        if abs(rxs) < 1e-12:
            return None
        t = _cross(qmp, s) / (rxs + 1e-18)
        u = _cross(qmp, r) / (rxs + 1e-18)
        if 0 <= t <= 1 and 0 <= u <= 1:
            return t, u
        return None

    def _closest_point_on_segment(a, b, p):
        ab = b - a
        t = np.dot(p - a, ab) / (np.dot(ab, ab) + 1e-18)
        t = max(0.0, min(1.0, t))
        return a + t * ab

    # Collect intersections
    intersections = []
    for i, sp in enumerate(spaces):
        poly = np.asarray(sp.get("poly"), dtype=float)
        if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 2:
            continue
        ring = poly if np.allclose(poly[0], poly[-1]) else np.vstack([poly, poly[0]])
        for e in range(len(ring) - 1):
            a = ring[e]; b = ring[e+1]
            res = _seg_seg_intersection(P_cam, d, a, (b - a))
            if res is not None:
                t, u = res
                P_int = P_cam + t * d
                intersections.append((t, P_int, i))

    if intersections:
        intersections.sort(key=lambda x: x[0])
        t_last, P_last, poly_i = intersections[-1]
        # step slightly inward
        P_in = P_cam + (t_last - max(eps / (d_norm + 1e-12), 1e-6)) * d
        return float(P_in[0]), float(P_in[1]), True, poly_i

    # No intersections → snap to closest boundary
    best_dist = 1e18; best_pt = None; best_idx = None
    for i, sp in enumerate(spaces):
        poly = np.asarray(sp.get("poly"), dtype=float)
        if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 2:
            continue
        ring = poly if np.allclose(poly[0], poly[-1]) else np.vstack([poly, poly[0]])
        for e in range(len(ring) - 1):
            a = ring[e]; b = ring[e+1]
            cp = _closest_point_on_segment(a, b, P_obj)
            dist = float(np.hypot(*(cp - P_obj)))
            if dist < best_dist:
                best_dist, best_pt, best_idx = dist, cp, i
    if best_pt is None:
        return float(P_cam[0]), float(P_cam[1]), True, None
    v = best_pt - P_cam
    vn = np.linalg.norm(v) + 1e-12
    P_in = best_pt - (eps * v / vn)
    return float(P_in[0]), float(P_in[1]), True, best_idx

# --------------- build_arkit_frame_index_map stays the same ---------------
def build_arkit_frame_index_map(meta_list, num_video_frames=None):
    frame_nums = []
    for m in meta_list:
        fn = None
        if isinstance(m, dict):
            fn = m.get('frameNumber')
            if isinstance(fn, str):
                try: fn = int(fn)
                except: fn = None
        frame_nums.append(fn)
    valid = [fn for fn in frame_nums if fn is not None]
    if len(valid) >= 10:
        fn_to_idx = {fn: idx for idx, fn in enumerate(frame_nums) if fn is not None}
        keys_sorted = sorted(fn_to_idx.keys())
        def map_fn(vf):
            if vf in fn_to_idx: return fn_to_idx[vf]
            nearest = min(keys_sorted, key=lambda x: abs(x - vf))
            return fn_to_idx[nearest]
        return map_fn
    else:
        def map_fn(vf): return min(vf, len(meta_list)-1)
        return map_fn


def process_video_first_per_class(video_path, detector, meta, mapped_plot, positions3d,
                                  spaces,   # <-- add this
                                  s_map, R_map, t_map, chosen_proj,
                                  target_classes, save_detected=True, debug_reproject=False, reproject_csv=None,
                                  global_use_transpose=None, global_z_sign=None, SAVE_DETECTED_CROPS=True,
                                  OUT_DEBUG_CSV=None, M_TO_FT=3.280839895013123):

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    logger.debug(f"Video opened: {video_path}, frames: {total} size: {vid_w}x{vid_h}")
    map_fn = build_arkit_frame_index_map(meta, num_video_frames=total)

    target_set = None if target_classes is None else {c.lower() for c in target_classes}
    first_seen = {}
    vf_idx = 0
    saved_frame_idxs = set()
    depth_cache = {}
    repro_records = []
    debug_rows = []
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
            mx, my = float(mapped_plot[a_idx, 0]), float(mapped_plot[a_idx, 1])

            cam_pos3d = positions3d[a_idx]
            cam_rot = None
            try:
                if isinstance(meta[a_idx], dict):
                    if meta[a_idx].get("rot") is not None:
                        cam_rot = np.array(meta[a_idx].get("rot"), dtype=float)
                    else:
                        raw = meta[a_idx].get("raw") if isinstance(meta[a_idx].get("raw"), dict) else None
                        if raw is not None:
                            rawmat = raw.get("cameraTransform") or raw.get("transform") or raw.get("matrix") or None
                            if rawmat is None:
                                for k, v in raw.items():
                                    if isinstance(v, list) and len(v) in (12, 16):
                                        rawmat = v; break
                            if rawmat is not None:
                                from utils.matrix_utils import parse_homogeneous_matrix
                                p, R = parse_homogeneous_matrix(rawmat)
                                if p is not None and R is not None:
                                    cam_rot = R
                if cam_rot is not None:
                    cam_rot = np.array(cam_rot, dtype=float)
                    if cam_rot.shape != (3, 3):
                        if cam_rot.T.shape == (3, 3):
                            cam_rot = cam_rot.T
                        else:
                            print("Warning: cam_rot found but has unexpected shape", cam_rot.shape, "-> ignoring")
                            cam_rot = None
            except Exception as e:
                print("Warning: error extracting cam_rot:", e)
                cam_rot = None

            # global override or per-frame vote
            z_sign = 1.0
            chosen_R = cam_rot
            orientation_conf = None
            if cam_rot is not None:
                if global_use_transpose is not None or global_z_sign is not None:
                    if global_use_transpose is True:
                        chosen_R = cam_rot.T
                    elif global_use_transpose is False:
                        chosen_R = cam_rot
                    else:
                        chosen_R = cam_rot
                    z_sign = float(global_z_sign) if global_z_sign is not None else 1.0
                    orientation_conf = 1.0
                else:
                    neighbor_idx = None
                    if a_idx + 1 < len(positions3d): neighbor_idx = a_idx + 1
                    elif a_idx - 1 >= 0: neighbor_idx = a_idx - 1
                    if neighbor_idx is not None:
                        from utils.orientation_utils import choose_best_rotation_and_sign
                        Rtry, sign_try, score_try = choose_best_rotation_and_sign(cam_pos3d, cam_rot, positions3d[neighbor_idx])
                        chosen_R = Rtry; z_sign = sign_try; orientation_conf = score_try
                    else:
                        chosen_R = cam_rot; z_sign = 1.0; orientation_conf = 0.0
            else:
                orientation_conf = None

            # save crop
            crop_path = None
            if save_detected:
                try:
                    Path(str(video_path)).parent.mkdir(parents=True, exist_ok=True)
                    h_img, w_img = frame.shape[:2]
                    x1, y1, x2, y2 = [int(round(v)) for v in d['xyxy']]
                    x1 = _clamp(x1, 0, w_img-1); y1 = _clamp(y1, 0, h_img-1)
                    x2 = _clamp(x2, 0, w_img-1); y2 = _clamp(y2, 0, h_img-1)
                    if x2 > x1 and y2 > y1:
                        crop = frame[y1:y2, x1:x2]
                        crop_path = Path(str(video_path)).parent / f"first_{cls_l}_vf{vf_idx}.jpg"
                        cv2.imwrite(str(crop_path), crop); crop_path = str(crop_path)
                except Exception as e:
                    print("Warning: failed to save crop:", e); crop_path = None

            annotated_frame = None
            if save_detected and vf_idx not in saved_frame_idxs:
                try:
                    frame_copy = frame.copy()
                    for dd in dets:
                        xx1, yy1, xx2, yy2 = [int(round(v)) for v in dd['xyxy']]
                        cv2.rectangle(frame_copy, (xx1, yy1), (xx2, yy2), (0,255,0), 2)
                        cv2.putText(frame_copy, f"{dd['class_name']} {dd['conf']:.2f}",
                                    (xx1, max(0, yy1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    annotated_frame = Path(str(video_path)).parent / f"annotated_first_vf{vf_idx}.jpg"
                    cv2.imwrite(str(annotated_frame), frame_copy); annotated_frame = str(annotated_frame)
                    saved_frame_idxs.add(vf_idx)
                except Exception as e:
                    print("Warning: failed to save annotated frame:", e); annotated_frame = None

            # depth via Zoe
            depth_val, distance_m, distance_ft = None, None, None
            try:
                if vf_idx in depth_cache:
                    depth_map = depth_cache[vf_idx]
                else:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    from detectors.zoe_depth import init_zoe, get_zoe_depth_map, _zoe_model, _zoe_device
                    if _zoe_model is None:
                        ok = init_zoe()
                        if not ok:
                            raise RuntimeError("Failed to initialize ZoeDepth model.")
                    depth_map = get_zoe_depth_map(rgb)
                    depth_cache[vf_idx] = depth_map

                x1, y1, x2, y2 = [int(round(v)) for v in d['xyxy']]
                cx_bbox = int(round((x1 + x2) / 2)); cy_bbox = int(round((y1 + y2) / 2))
                h_img, w_img = depth_map.shape[:2]
                cx_bbox = _clamp(cx_bbox, 0, w_img-1); cy_bbox = _clamp(cy_bbox, 0, h_img-1)
                patch = depth_map[max(0, cy_bbox-1):min(h_img-1, cy_bbox+1)+1,
                                  max(0, cx_bbox-1):min(w_img-1, cx_bbox+1)+1]
                depth_val = float(np.median(patch.astype(float))) if patch.size > 0 else float(depth_map[cy_bbox, cx_bbox])
                distance_m = depth_val; distance_ft = distance_m * M_TO_FT
            except Exception as e:
                print("Warning: ZoeDepth failed for vf", vf_idx, "->", e)

            # compute object world & mapped
            obj_x, obj_y, obj_yaw = None, None, None
            p_world = None
            real_w_m, real_h_m = None, None
            try:
                from utils.intrinsics_utils import get_intrinsics_from_meta
                intr = None
                try:
                    intr = get_intrinsics_from_meta(meta[a_idx])
                    if intr is None and isinstance(meta[a_idx].get("raw"), dict):
                        intr = get_intrinsics_from_meta(meta[a_idx].get("raw"))
                except Exception:
                    intr = None

                if chosen_R is not None and distance_m is not None:
                    frame_size = (frame_w, frame_h)
                    # compute_object_world_and_mapped copied locally to avoid cross-file heavy deps
                    def compute_object_world_and_mapped_local(cam_pos3d, cam_rot, intrinsics_tuple, bbox_xyxy, depth_val,
                                                           chosen_proj, s_map, R_map, t_map, frame_size=None, z_sign=1.0):
                        try:
                            if cam_pos3d is None or cam_rot is None or depth_val is None:
                                return None, None, None, None
                            x1, y1, x2, y2 = bbox_xyxy
                            u = (x1 + x2) * 0.5; v = (y1 + y2) * 0.5
                            if intrinsics_tuple is None:
                                fw, fh = (frame_size if frame_size is not None else (640, 480))
                                fx = fy = 0.8 * max(fw, fh); cx = fw / 2.0; cy = fh / 2.0
                            else:
                                fx, fy, cx, cy, K_img_w, K_img_h = intrinsics_tuple
                                if frame_size is not None and K_img_w is not None and K_img_h is not None:
                                    frame_w, frame_h = frame_size
                                    try:
                                        K_w = int(K_img_w); K_h = int(K_img_h)
                                    except Exception:
                                        K_w = None; K_h = None
                                    if K_w is not None and K_h is not None and (K_w != frame_w or K_h != frame_h):
                                        sx = float(frame_w) / float(K_w); sy = float(frame_h) / float(K_h)
                                        fx = fx * sx; fy = fy * sy; cx = cx * sx; cy = cy * sy
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
                            from utils.projection_utils import project_3d_to_2d, apply_similarity_to_points
                            obj_proj2 = project_3d_to_2d(p_obj_world.reshape(1, 3), chosen_proj)
                            obj_mapped = apply_similarity_to_points(obj_proj2, s_map, R_map, t_map)
                            obj_x = float(obj_mapped[0, 0]); obj_y = float(obj_mapped[0, 1])
                            from utils.orientation_utils import compute_yaw_from_direction
                            yaw_deg = compute_yaw_from_direction(d_world_dir)
                            return obj_x, obj_y, yaw_deg, p_obj_world
                        except Exception as e:
                            print("compute_object_world_and_mapped error:", e)
                            return None, None, None, None

                    objx, objy, objyaw, pworld = compute_object_world_and_mapped_local(cam_pos3d, chosen_R, intr, d['xyxy'],
                                                                                distance_m, chosen_proj, s_map, R_map, t_map,
                                                                                frame_size=frame_size, z_sign=z_sign)
                    if objx is not None:
                        # Clamp to floor polygons along the ray camera->object
                        clamped_x, clamped_y, was_clamped, _poly_i = clamp_object_to_floor(
                                spaces,
                                mx, my,            # camera mapped position for this frame
                                objx, objy,
                                eps=0.02           # ~2cm; tweak to your unit scale
                            )
                        if was_clamped and VERBOSE:
                            print(f"  (clamped to floor) ({objx:.3f},{objy:.3f}) -> ({clamped_x:.3f},{clamped_y:.3f})")
                        obj_x, obj_y, obj_yaw = clamped_x, clamped_y, objyaw
                        p_world = pworld

                    #-------------Uncomment this for without boundary validation-----------------#
                    # if objx is not None:
                    #     obj_x, obj_y, obj_yaw = objx, objy, objyaw
                    #     p_world = pworld


                        # compute real-world width & height (meters) from bbox + depth + intrinsics
                        try:
                            real_w_m, real_h_m = compute_real_size_from_bbox(d['xyxy'], distance_m, intr, frame_size=(frame_w, frame_h))
                        except Exception as e:
                            if VERBOSE:
                                print("Warning: compute_real_size_from_bbox failed:", e)
                            real_w_m, real_h_m = None, None

                        if debug_reproject:
                            x1, y1, x2, y2 = d['xyxy']
                            u = (x1 + x2) * 0.5; v = (y1 + y2) * 0.5
                            if intr is not None:
                                fx, fy, cx, cy, K_w, K_h = intr
                                if (K_w is not None and K_h is not None) and (frame_w != K_w or frame_h != K_h):
                                    sx = float(frame_w) / float(K_w); sy = float(frame_h) / float(K_h)
                                    fx = fx * sx; fy = fy * sy; cx = cx * sx; cy = cy * sy
                            else:
                                fx = fy = 0.8 * max(frame_w, frame_h); cx = frame_w / 2.0; cy = frame_h / 2.0
                            d_cam = np.array([(u - cx) / fx, (v - cy) / fy, float(z_sign)], dtype=float)
                            d_cam_n = d_cam / np.linalg.norm(d_cam) if np.linalg.norm(d_cam) != 0 else d_cam
                            R = np.array(chosen_R, dtype=float)
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
                            debug_rows.append({
                                "class": cls,
                                "video_frame_index": vf_idx,
                                "arkit_index": a_idx,
                                "cam_pos_x": float(cam_pos3d[0]), "cam_pos_y": float(cam_pos3d[1]), "cam_pos_z": float(cam_pos3d[2]),
                                "p_world_x": float(p_world[0]) if p_world is not None else None,
                                "p_world_y": float(p_world[1]) if p_world is not None else None,
                                "p_world_z": float(p_world[2]) if p_world is not None else None,
                                "mapped_x": obj_x, "mapped_y": obj_y,
                                "u": u, "v": v, "u_est": u_est, "v_est": v_est,
                                "depth_m": distance_m,
                                "orientation_confidence": orientation_conf,
                                "real_width_m": real_w_m,
                                "real_height_m": real_h_m
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
                'real_width_m': real_w_m,
                'real_height_m': real_h_m,
            }
            first_seen[cls_l] = info

            dist_print = f"{distance_m:.3f}m" if distance_m is not None else "N/A"
            size_print = f"{real_w_m:.2f}m x {real_h_m:.2f}m" if (real_w_m is not None and real_h_m is not None) else "N/A"
            print(f"DETECTED-START: {cls} vf{vf_idx} cam=({mx:.1f},{my:.1f}) obj=({obj_x if obj_x else 'N/A'},{obj_y if obj_y else 'N/A'}) dist={dist_print} size={size_print} yaw={obj_yaw if obj_yaw else 'N/A'} z_sign={z_sign} orient_conf={orientation_conf}")

            if target_set is not None and set(first_seen.keys()) >= target_set:
                break

        if target_set is not None and set(first_seen.keys()) >= target_set:
            break

        vf_idx += 1

    cap.release()

    if debug_reproject and reproject_csv and repro_records:
        try:
            pd.DataFrame(repro_records).to_csv(reproject_csv, index=False)
            print("Saved reprojection debug CSV:", reproject_csv)
        except Exception as e:
            print("Warning: failed to save reprojection CSV:", e)

    if debug_rows and OUT_DEBUG_CSV is not None:
        try:
            pd.DataFrame(debug_rows).to_csv(OUT_DEBUG_CSV, index=False)
            print("Saved per-detection debug CSV:", OUT_DEBUG_CSV)
        except Exception as e:
            print("Warning: failed to save per-detection debug CSV:", e)

    return first_seen
