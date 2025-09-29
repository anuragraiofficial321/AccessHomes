"""
Video mapping and per-class first-seen detection processing.

This module contains the `process_video_first_per_class` function and helpers.
It mirrors the behavior of the monolithic script.
"""

import cv2
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from pathlib import Path

from detectors.yolo_detector import YoloDetector
from detectors.zoe_depth import init_zoe, get_zoe_depth_map, ZoeDepthForDepthEstimation

#!/usr/bin/env python3
"""
Video mapping and per-class first-seen detection processing.

This module contains the `process_video_first_per_class` function and helpers.
It mirrors the behavior of the monolithic script with added saving of:
 - annotated frames for detections (bbox + label)
 - accepted-frame images when a detection is accepted (room membership)
"""

import cv2
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from pathlib import Path
import os

from detectors.yolo_detector import YoloDetector
from detectors.zoe_depth import init_zoe, get_zoe_depth_map, ZoeDepthForDepthEstimation

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

def _clamp(v, a, b): return max(a, min(b, v))

def point_in_space(pt, spoly):
    try:
        if spoly is None:
            return False
        mp = MplPath(np.asarray(spoly, dtype=float))
        return bool(mp.contains_point((float(pt[0]), float(pt[1]))))
    except Exception:
        return False

def find_space_for_point(pt, spaces, margin=0.0):
    for sp in (spaces or []):
        poly = sp.get("poly") if isinstance(sp, dict) else None
        if poly is None:
            continue
        try:
            if point_in_space(pt, poly):
                return sp, (sp.get("id") or sp.get("name") or None)
        except Exception:
            continue
    if margin is not None and margin > 0.0:
        px, py = float(pt[0]), float(pt[1])
        margin2 = float(margin) * float(margin)
        closest_sp = None
        closest_d2 = float("inf")
        for sp in (spaces or []):
            try:
                poly = np.asarray(sp.get("poly"), dtype=float)
            except Exception:
                continue
            if poly.size == 0:
                continue
            pts = poly.reshape(-1, 2)
            for i in range(len(pts)):
                a = pts[i]
                b = pts[(i + 1) % len(pts)]
                ax, ay = float(a[0]), float(a[1])
                bx, by = float(b[0]), float(b[1])
                vx, vy = bx - ax, by - ay
                wx, wy = px - ax, py - ay
                denom = vx*vx + vy*vy
                if denom == 0:
                    t = 0.0
                else:
                    t = (wx*vx + wy*vy) / denom
                    t = max(0.0, min(1.0, t))
                projx = ax + t * vx
                projy = ay + t * vy
                dx = px - projx; dy = py - projy
                d2 = dx*dx + dy*dy
                if d2 < closest_d2:
                    closest_d2 = d2
                    closest_sp = sp
        if closest_d2 <= margin2:
            return closest_sp, (closest_sp.get("id") or closest_sp.get("name") or None)
    return None, None

def _ensure_dir(p: Path):
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

def _save_annotated_frame(frame_bgr, bbox, label_text, out_path: Path):
    """Draw bbox + label on a copy of frame and save to out_path (BGR)."""
    try:
        ann = frame_bgr.copy()
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        # rectangle
        cv2.rectangle(ann, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        # label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        ((tw, th), _) = cv2.getTextSize(label_text, font, scale, thickness)
        cv2.rectangle(ann, (x1, y1 - th - 6), (x1 + tw + 6, y1), (0, 255, 0), -1)
        cv2.putText(ann, label_text, (x1 + 3, y1 - 4), font, scale, (0, 0, 0), thickness, lineType=cv2.LINE_AA)
        # save
        cv2.imwrite(str(out_path), ann)
    except Exception as e:
        # don't break pipeline for save errors
        print("Warning: failed to save annotated frame:", e)

def _save_accepted_frame(frame_bgr, bbox, label_text, out_path: Path):
    """Save a frame indicating acceptance. Draw thicker bbox + ACCEPTED banner."""
    try:
        ann = frame_bgr.copy()
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
        # thicker rectangle to denote accepted
        cv2.rectangle(ann, (x1, y1), (x2, y2), color=(0, 128, 255), thickness=3)
        # ACCEPTED banner at top-left
        banner = "ACCEPTED"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        thickness = 2
        ((tw, th), _) = cv2.getTextSize(banner, font, scale, thickness)
        cv2.rectangle(ann, (x1, y1 - th - 12), (x1 + tw + 12, y1), (0, 128, 255), -1)
        cv2.putText(ann, banner, (x1 + 6, y1 - 6), font, scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
        # also write the class label
        cv2.putText(ann, label_text, (x1, y2 + 20), font, 0.6, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        cv2.imwrite(str(out_path), ann)
    except Exception as e:
        print("Warning: failed to save accepted frame:", e)


def process_video_first_per_class(video_path, detector, meta, mapped_plot, positions3d,
                                  s_map, R_map, t_map, chosen_proj,
                                  target_classes, spaces=None, save_detected=True,
                                  debug_reproject=False, reproject_csv=None,
                                  global_use_transpose=None, global_z_sign=None,
                                  room_margin=0.0, inside_push=3.0,
                                  verbose=True):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if verbose:
        print(f"Video opened: {video_path}, frames: {total} size: {vid_w}x{vid_h}")
    map_fn = build_arkit_frame_index_map(meta, num_video_frames=total)

    # output dirs for annotated & accepted frames
    out_base = Path("output_data")
    ann_dir = out_base / "annotated_frames"
    acc_dir = out_base / "accepted_frames"
    _ensure_dir(ann_dir); _ensure_dir(acc_dir)

    target_set = None if target_classes is None else {c.lower() for c in target_classes}
    first_seen = {}
    vf_idx = 0
    depth_cache = {}
    repro_records = []
    debug_rows = []
    rejected_rows = []

    def _snap_point_to_poly(px, py, poly):
        try:
            pts = np.asarray(poly, dtype=float).reshape(-1, 2)
        except Exception:
            return px, py
        closest_d2 = float("inf")
        closest_xy = (px, py)
        for i in range(len(pts)):
            a = pts[i]
            b = pts[(i + 1) % len(pts)]
            ax, ay = float(a[0]), float(a[1])
            bx, by = float(b[0]), float(b[1])
            vx, vy = bx - ax, by - ay
            wx, wy = px - ax, py - ay
            denom = vx * vx + vy * vy
            if denom == 0:
                t = 0.0
            else:
                t = (wx * vx + wy * vy) / denom
                t = max(0.0, min(1.0, t))
            projx = ax + t * vx
            projy = ay + t * vy
            dx = px - projx; dy = py - projy
            d2 = dx * dx + dy * dy
            if d2 < closest_d2:
                closest_d2 = d2
                closest_xy = (projx, projy)
        return float(closest_xy[0]), float(closest_xy[1])

    def _nudge_point_inside(px, py, poly, move_dist):
        try:
            pts = np.asarray(poly, dtype=float).reshape(-1, 2)
        except Exception:
            return px, py
        centroid = pts.mean(axis=0)
        vec = centroid - np.array([px, py], dtype=float)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            return px, py
        desired = min(move_dist, norm * 0.9)
        new = np.array([px, py], dtype=float) + (vec / norm) * desired
        tries = 10
        curr_move = desired
        while tries > 0:
            if point_in_space(new, pts):
                return float(new[0]), float(new[1])
            curr_move *= 0.5
            if curr_move < 1e-3:
                break
            new = np.array([px, py], dtype=float) + (vec / norm) * curr_move
            tries -= 1
        sx, sy = _snap_point_to_poly(px, py, pts)
        v2 = centroid - np.array([sx, sy], dtype=float)
        n2 = float(np.linalg.norm(v2))
        if n2 > 1e-6:
            tiny = min(0.5, n2 * 0.5)
            sx = sx + (v2[0] / n2) * tiny
            sy = sy + (v2[1] / n2) * tiny
        return float(sx), float(sy)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_h, frame_w = frame.shape[:2]
        dets = []
        try:
            dets = detector.detect_frame(frame)
        except Exception as e:
            if verbose:
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
                            if verbose:
                                print("Warning: cam_rot found but has unexpected shape", cam_rot.shape, "-> ignoring")
                            cam_rot = None
            except Exception as e:
                if verbose:
                    print("Warning: error extracting cam_rot:", e)
                cam_rot = None

            # orientation
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

            # prepare annotated frame save (if requested)
            if save_detected:
                try:
                    bbox = [int(round(v)) for v in d['xyxy']]
                    label_text = f"{cls} {d.get('conf',0):.2f}"
                    ann_fname = f"vf{vf_idx}_arkit{a_idx}_{cls_l}.jpg"
                    ann_path = ann_dir / ann_fname
                    _save_annotated_frame(frame, bbox, label_text, ann_path)
                except Exception as e:
                    if verbose:
                        print("Warning: failed to create/save annotated frame:", e)

            crop_path = None
            annotated_frame = None

            depth_val, distance_m, distance_ft = None, None, None
            try:
                if vf_idx in depth_cache:
                    depth_map = depth_cache[vf_idx]
                else:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if ZoeDepthForDepthEstimation is None:
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
                distance_m = depth_val; distance_ft = distance_m * 3.280839895013123
            except Exception as e:
                if verbose:
                    print("Warning: ZoeDepth failed for vf", vf_idx, "->", e)

            obj_x, obj_y, obj_yaw = None, None, None
            p_world = None
            try:
                intr = None
                try:
                    from utils.intrinsics_utils import get_intrinsics_from_meta as gk
                    intr = gk(meta[a_idx])
                    if intr is None and isinstance(meta[a_idx].get("raw"), dict):
                        intr = gk(meta[a_idx].get("raw"))
                except Exception:
                    intr = None

                if chosen_R is not None and distance_m is not None:
                    from utils.reprojection_utils import compute_object_world_and_mapped
                    frame_size = (frame_w, frame_h)
                    objx, objy, objyaw, pworld = compute_object_world_and_mapped(cam_pos3d, chosen_R, intr, d['xyxy'],
                                                                                distance_m, chosen_proj, s_map, R_map, t_map,
                                                                                frame_size=frame_size, z_sign=z_sign)
                    if objx is not None:
                        obj_x, obj_y, obj_yaw = objx, objy, objyaw
                        p_world = pworld

                        if debug_reproject:
                            x1, y1, x2, y2 = d['xyxy']
                            u = (x1 + x2) * 0.5; v = (y1 + y2) * 0.5
                            repro_records.append({
                                "vf_idx": vf_idx, "class": cls, "u": u, "v": v, "depth": distance_m, "z_sign": z_sign
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
                                "u": u, "v": v, "depth_m": distance_m,
                                "orientation_confidence": orientation_conf
                            })
            except Exception as e:
                if verbose:
                    print("Warning: failed to compute object true position:", e)

            space_cam = None; space_obj = None; space_cam_id = None; space_obj_id = None
            accepted_flag = False
            try:
                if spaces is not None:
                    space_cam, space_cam_id = find_space_for_point((mx, my), spaces, margin=room_margin)
                    if obj_x is not None and obj_y is not None:
                        space_obj, space_obj_id = find_space_for_point((obj_x, obj_y), spaces, margin=room_margin)
                    else:
                        space_obj, space_obj_id = None, None
                if spaces is not None:
                    same_space = (space_cam is not None and space_obj is not None and (space_cam_id == space_obj_id))
                    if not same_space:
                        rejected_rows.append({
                            "class": cls,
                            "video_frame_index": vf_idx,
                            "arkit_index": a_idx,
                            "cam_space": space_cam_id,
                            "obj_space": space_obj_id,
                            "cam_mapped_x": mx, "cam_mapped_y": my,
                            "object_x": obj_x, "object_y": obj_y,
                            "distance_m": distance_m,
                            "conf": d.get("conf")
                        })
                        if verbose:
                            print(f"REJECTED (room mismatch): {cls} vf{vf_idx} cam_space={space_cam_id} obj_space={space_obj_id} obj=({obj_x},{obj_y})")
                        continue

                    if obj_x is not None and obj_y is not None and space_obj is not None:
                        try:
                            poly = np.asarray(space_obj.get("poly"), dtype=float)
                            if not point_in_space((obj_x, obj_y), poly):
                                sx, sy = _snap_point_to_poly(obj_x, obj_y, poly)
                                obj_x, obj_y = sx, sy
                            if inside_push is not None and inside_push > 0:
                                nx, ny = _nudge_point_inside(obj_x, obj_y, poly, float(inside_push))
                                obj_x, obj_y = nx, ny
                        except Exception as e:
                            if verbose:
                                print("Warning: inside-push step failed:", e)

                # If we reach here, detection is accepted (room membership ok or spaces is None)
                accepted_flag = True
            except Exception as e:
                if verbose:
                    print("Warning: room-membership check failed:", e)
                continue

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
                'crop_path': None,
                'annotated_frame': None,
                'depth_model_units': depth_val,
                'distance_m': distance_m,
                'distance_ft': distance_ft,
                'cam_space': (space_cam_id if space_cam_id is not None else None),
                'obj_space': (space_obj_id if space_obj_id is not None else None),
            }
            # Save accepted-frame image (a special visual) if detection accepted and save_detected True
            if accepted_flag and save_detected:
                try:
                    bbox_int = [int(round(v)) for v in d['xyxy']]
                    label_text = f"{cls} {d.get('conf',0):.2f}"
                    acc_fname = f"vf{vf_idx}_arkit{a_idx}_{cls_l}_ACCEPT.jpg"
                    acc_path = acc_dir / acc_fname
                    _save_accepted_frame(frame, bbox_int, label_text, acc_path)
                    info['annotated_frame'] = str(acc_path)
                except Exception as e:
                    if verbose:
                        print("Warning: failed to save accepted-frame image:", e)

            # add to first_seen
            first_seen[cls_l] = info

            dist_print = f"{distance_m:.3f}m" if distance_m is not None else "N/A"
            if verbose:
                print(f"DETECTED-ACCEPT: {cls} vf{vf_idx} cam=({mx:.1f},{my:.1f}) obj=({obj_x if obj_x else 'N/A'},{obj_y if obj_y else 'N/A'}) dist={dist_print} yaw={obj_yaw if obj_yaw else 'N/A'} z_sign={z_sign} ")

            if target_set is not None and set(first_seen.keys()) >= target_set:
                break

        if target_set is not None and set(first_seen.keys()) >= target_set:
            break

        vf_idx += 1

    cap.release()

    if rejected_rows:
        try:
            rej_df = pd.DataFrame(rejected_rows)
            rej_path = Path("output_data") / "rejected_detections_room_mismatch.csv"
            rej_df.to_csv(str(rej_path), index=False)
            if verbose:
                print("Saved rejected detections (room mismatch) to:", rej_path)
        except Exception as e:
            if verbose:
                print("Warning: failed to save rejected detections CSV:", e)

    if debug_reproject and reproject_csv is not None and repro_records:
        try:
            pd.DataFrame(repro_records).to_csv(reproject_csv, index=False)
            if verbose:
                print("Saved reprojection debug CSV:", reproject_csv)
        except Exception as e:
            if verbose:
                print("Warning: failed to save reprojection CSV:", e)

    if debug_rows:
        try:
            pd.DataFrame(debug_rows).to_csv(Path("output_data") / "first_seen_detections_debug.csv", index=False)
            if verbose:
                print("Saved per-detection debug CSV:", Path("output_data") / "first_seen_detections_debug.csv")
        except Exception as e:
            if verbose:
                print("Warning: failed to save per-detection debug CSV:", e)

    return first_seen
